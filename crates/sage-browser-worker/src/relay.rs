//! Bounded, full-duplex native messaging. Control remains readable while the
//! extension is awaiting an OS/browser operation. Only correlated replies cross
//! back into the broker; late replies retain their original request identity.
use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use sage_core::ipc::{read_frame, write_frame};
use sage_protocol::{PROTOCOL_VERSION, sage::ipc::v2 as wire};
use serde_json::{Value, json};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::sync::mpsc;
use tokio::time::{Instant, timeout};

use super::{HostResult, frame};

const WRITE_LIMIT: Duration = Duration::from_secs(1);

pub(super) async fn session<S, I, O>(
    stream: S,
    mut input: I,
    mut output: O,
    session_id: String,
    mut received_sequence: u64,
    mut sent_sequence: u64,
) -> HostResult<()>
where
    S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
    I: AsyncRead + Unpin + Send + 'static,
    O: AsyncWrite + Unpin,
{
    let (mut reader, mut writer) = tokio::io::split(stream);
    // read_exact must never be dropped halfway through a frame by select!.
    // JoinSet aborts both readers when the session returns or is cancelled.
    let mut readers = tokio::task::JoinSet::new();
    let (frames_tx, mut frames_rx) = mpsc::channel(4);
    readers.spawn(async move {
        loop {
            let frame = read_frame(&mut reader).await;
            let failed = frame.is_err();
            if frames_tx.send(frame).await.is_err() || failed {
                break;
            }
        }
    });
    let (browser_tx, mut browser_rx) = mpsc::channel(4);
    readers.spawn(async move {
        loop {
            let message = read_message(&mut input).await;
            let failed = message.is_err();
            if browser_tx.send(message).await.is_err() || failed {
                break;
            }
        }
    });
    let mut pending = HashMap::<String, Instant>::new();
    let mut cancellations = HashMap::<String, Instant>::new();
    let mut expiry_check = tokio::time::interval(Duration::from_secs(1));
    loop {
        let outgoing = tokio::select! {
            incoming = frames_rx.recv() => {
                let incoming = incoming.ok_or("Core reader disconnected")??;
                if incoming.protocol_version != PROTOCOL_VERSION || incoming.sequence <= received_sequence {
                    return Err("Invalid or replayed core frame".into());
                }
                received_sequence = incoming.sequence;
                match incoming.payload {
                    Some(wire::frame::Payload::AdapterRequest(request)) => {
                        if pending.contains_key(&request.request_id) { return Err("Duplicate browser dispatch".into()); }
                        if pending.len() >= 32 {
                            Some(wire::frame::Payload::AdapterResult(wire::AdapterResult {
                                request_id: request.request_id, error: "Browser worker capacity exceeded".into(), ..Default::default()
                            }))
                        } else {
                            let deadline = retirement_deadline(request.expires_at_unix_ms)?;
                            let message = browser_payload(&request, &session_id)?;
                            pending.insert(request.request_id, deadline);
                            timeout(WRITE_LIMIT, write_message(&mut output, &message)).await??;
                            None
                        }
                    }
                    Some(wire::frame::Payload::AdapterCancel(cancel)) => {
                        if cancellations.len() >= 128 || cancellations.contains_key(&cancel.request_id) {
                            return Err("Invalid or excessive browser cancellation".into());
                        }
                        // A cancellation can precede an unsent request. Its ACK
                        // is correlated separately from final effect receipts.
                        cancellations.insert(cancel.request_id.clone(), retirement_deadline(cancel.expires_at_unix_ms)?);
                        let message = json!({"request_id":cancel.request_id,"operation":"cancel","expires_at_unix_ms":cancel.expires_at_unix_ms});
                        timeout(WRITE_LIMIT, write_message(&mut output, &message)).await??;
                        None
                    }
                    Some(wire::frame::Payload::Pong(_)) => None,
                    _ => return Err("Unexpected core message for browser worker".into()),
                }
            }
            reply = browser_rx.recv() => {
                let reply = reply.ok_or("Browser reader disconnected")??;
                let id = reply["request_id"].as_str().ok_or("Missing browser response identity")?;
                if reply["operation"].as_str() == Some("cancel_ack") {
                    if cancellations.remove(id).is_none() { return Err("Uncorrelated browser cancellation acknowledgement".into()); }
                    Some(wire::frame::Payload::AdapterCancelAcknowledged(wire::AdapterCancelAcknowledged { request_id: id.into() }))
                } else {
                    if pending.remove(id).is_none() { return Err("Uncorrelated browser response".into()); }
                    let success = reply["success"].as_bool().ok_or("Missing browser response outcome")?;
                    let error = reply["error"].as_str().unwrap_or(if success { "" } else { "Browser action failed" });
                    if error.len() > 4096 { return Err("Browser error exceeds bounds".into()); }
                    Some(wire::frame::Payload::AdapterResult(wire::AdapterResult {
                        request_id: id.into(), success, json: serde_json::to_string(&reply["data"])?, error: error.into(), ..Default::default()
                    }))
                }
            }
            _ = expiry_check.tick() => {
                // Do not synthesize success or discard a possible late effect.
                // A missing final receipt ends the worker session as uncertain.
                let now = Instant::now();
                if pending.values().chain(cancellations.values()).any(|deadline| *deadline <= now) {
                    return Err("Browser worker did not settle its expired requests".into());
                }
                None
            }
        };
        if let Some(payload) = outgoing {
            sent_sequence = sent_sequence
                .checked_add(1)
                .ok_or("IPC sequence exhausted")?;
            timeout(
                WRITE_LIMIT,
                write_frame(&mut writer, &frame(sent_sequence, payload)),
            )
            .await??;
        }
    }
}

fn retirement_deadline(expires: i64) -> HostResult<Instant> {
    let now = i64::try_from(SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis())?;
    let remaining = expires.saturating_sub(now);
    if remaining > 60_000 {
        return Err("Browser request expiry exceeds bounds".into());
    }
    Ok(Instant::now() + Duration::from_millis(remaining.max(0) as u64) + Duration::from_secs(60))
}

fn browser_payload(request: &wire::AdapterRequest, session: &str) -> HostResult<Value> {
    let mut body: Value = serde_json::from_str(&request.json)?;
    if let Some(target) = &request.browser_target {
        body.as_object_mut().ok_or("Invalid action body")?.insert(
            "browser_target".into(),
            serde_json::to_value(sage_core::browser_target::BrowserTarget::from_wire(target)?)?,
        );
    }
    if request.operation == "execute" {
        let grant = request
            .grant
            .as_ref()
            .ok_or("Missing typed execution grant")?;
        if grant.worker_session != session || grant.domain != "browser" {
            return Err("Grant belongs to another worker".into());
        }
        body.as_object_mut().ok_or("Invalid action body")?.insert(
            "capability".into(),
            sage_core::capability::wire_grant_json(grant)?,
        );
    }
    Ok(
        json!({"request_id":request.request_id,"operation":request.operation,"payload":body,"expires_at_unix_ms":request.expires_at_unix_ms}),
    )
}

async fn read_message<I: AsyncRead + Unpin>(input: &mut I) -> HostResult<Value> {
    let length = input.read_u32_le().await? as usize;
    if length == 0 || length > 256 * 1024 {
        return Err("Browser response exceeds bounds".into());
    }
    let mut bytes = vec![0; length];
    input.read_exact(&mut bytes).await?;
    Ok(serde_json::from_slice(&bytes)?)
}

async fn write_message<O: AsyncWrite + Unpin>(output: &mut O, message: &Value) -> HostResult<()> {
    let bytes = serde_json::to_vec(message)?;
    if bytes.len() > 1024 * 1024 {
        return Err("Browser request exceeds bounds".into());
    }
    output.write_u32_le(bytes.len() as u32).await?;
    output.write_all(&bytes).await?;
    output.flush().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn cancellation_bypasses_pending_browser_work_and_keeps_late_replies_correlated() {
        let (mut core, host) = tokio::io::duplex(8192);
        let (mut browser, native) = tokio::io::duplex(8192);
        let (input, output) = tokio::io::split(native);
        let worker = tokio::spawn(session(host, input, output, "test-session".into(), 0, 0));
        timeout(Duration::from_secs(2), async {
            let expiry = i64::try_from(SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis()).unwrap() + 30_000;
            let request = wire::AdapterRequest { request_id: "pending".into(), operation: "binding".into(), json: "{}".into(), expires_at_unix_ms: expiry, ..Default::default() };
            write_frame(&mut core, &frame(1, wire::frame::Payload::AdapterRequest(Box::new(request)))).await.unwrap();
            assert_eq!(read_message(&mut browser).await.unwrap()["request_id"], "pending");
            write_frame(&mut core, &frame(2, wire::frame::Payload::AdapterCancel(wire::AdapterCancel { request_id: "pending".into(), expires_at_unix_ms: expiry }))).await.unwrap();
            assert_eq!(read_message(&mut browser).await.unwrap()["operation"], "cancel");
            write_message(&mut browser, &json!({"request_id":"pending","operation":"cancel_ack"})).await.unwrap();
            let acknowledgement = read_frame(&mut core).await.unwrap();
            assert_eq!(acknowledgement.sequence, 1);
            assert!(matches!(acknowledgement.payload, Some(wire::frame::Payload::AdapterCancelAcknowledged(_))));
            write_message(&mut browser, &json!({"request_id":"pending","success":true,"data":{"url":"already-changed"}})).await.unwrap();
            let receipt = read_frame(&mut core).await.unwrap();
            assert_eq!(receipt.sequence, 2);
            let Some(wire::frame::Payload::AdapterResult(result)) = receipt.payload else { panic!("Expected late adapter result"); };
            assert!(result.success);
            assert_eq!(result.request_id, "pending");
            // Browser data cannot promote itself to a trusted UI command.
            write_message(&mut browser, &json!({"request_id":"unrelated","success":true,"command":{"submit_task":"untrusted"}})).await.unwrap();
        }).await.expect("Stop waited for the browser result");
        assert!(
            timeout(Duration::from_secs(2), worker)
                .await
                .unwrap()
                .unwrap()
                .unwrap_err()
                .to_string()
                .contains("Uncorrelated")
        );
    }
}
