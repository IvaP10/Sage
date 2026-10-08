#![forbid(unsafe_code)]

use std::io::{self, Read, Write};

use serde::{Deserialize, Serialize};

const PROTOCOL_VERSION: u16 = 1;
const MAX_FRAME_BYTES: usize = 4 * 1024;
const MAX_MESSAGES_PER_PROCESS: usize = 64;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WorkerRequest {
    protocol_version: u16,
    kind: String,
    challenge: String,
}

#[derive(Debug, Serialize)]
struct WorkerResponse<'a> {
    protocol_version: u16,
    kind: &'a str,
    challenge: &'a str,
    process_id: u32,
    readiness: &'a str,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("sage-inference-worker: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args_os().skip(1);
    match arguments.next().as_deref() {
        Some(argument) if argument == "--version" => {
            println!(
                "sage-inference-worker {} protocol {PROTOCOL_VERSION}",
                env!("CARGO_PKG_VERSION")
            );
            Ok(())
        }
        #[cfg(all(feature = "sandbox-probe", target_os = "macos"))]
        Some(argument) if argument == "--sandbox-probe" => sandbox_probe(arguments.collect()),
        None => run_stdio_protocol(),
        _ => Err("unsupported command line".into()),
    }
}

fn run_stdio_protocol() -> Result<(), Box<dyn std::error::Error>> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut input = stdin.lock();
    let mut output = stdout.lock();

    for _ in 0..MAX_MESSAGES_PER_PROCESS {
        let Some(request_bytes) = read_frame(&mut input)? else {
            return Ok(());
        };
        let request: WorkerRequest = serde_json::from_slice(&request_bytes)?;
        validate_request(&request)?;
        let response = WorkerResponse {
            protocol_version: PROTOCOL_VERSION,
            kind: "hello",
            challenge: &request.challenge,
            process_id: std::process::id(),
            readiness: "model_not_admitted",
        };
        write_frame(&mut output, &serde_json::to_vec(&response)?)?;
    }

    Err("worker message limit reached".into())
}

fn validate_request(request: &WorkerRequest) -> Result<(), Box<dyn std::error::Error>> {
    if request.kind != "hello" {
        return Err("unsupported worker request".into());
    }
    if request.protocol_version != PROTOCOL_VERSION {
        return Err("unsupported worker protocol version".into());
    }
    if request.challenge.len() != 64
        || !request
            .challenge
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        return Err("worker challenge must be 32 bytes encoded as hexadecimal".into());
    }
    Ok(())
}

fn read_frame<R: Read>(reader: &mut R) -> io::Result<Option<Vec<u8>>> {
    let mut length = [0_u8; 4];
    let first = reader.read(&mut length[..1])?;
    if first == 0 {
        return Ok(None);
    }
    reader.read_exact(&mut length[1..])?;
    let size = u32::from_be_bytes(length) as usize;
    if size == 0 || size > MAX_FRAME_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "worker frame exceeds its size bound",
        ));
    }
    let mut frame = vec![0; size];
    reader.read_exact(&mut frame)?;
    Ok(Some(frame))
}

fn write_frame<W: Write>(writer: &mut W, frame: &[u8]) -> io::Result<()> {
    if frame.is_empty() || frame.len() > MAX_FRAME_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "worker response exceeds its size bound",
        ));
    }
    let length = u32::try_from(frame.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "worker frame is too large"))?;
    writer.write_all(&length.to_be_bytes())?;
    writer.write_all(frame)?;
    writer.flush()
}

#[cfg(all(feature = "sandbox-probe", target_os = "macos"))]
fn sandbox_probe(arguments: Vec<std::ffi::OsString>) -> Result<(), Box<dyn std::error::Error>> {
    use std::{
        fs,
        net::{SocketAddr, TcpStream},
        os::fd::RawFd,
        path::PathBuf,
        time::Duration,
    };

    let mut read_path = None;
    let mut write_path = None;
    let mut connect_address = None;
    let mut inherited_fd = None;
    let mut values = arguments.into_iter();
    while let Some(option) = values.next() {
        let value = values.next().ok_or("missing sandbox probe option value")?;
        match option.to_str() {
            Some("--read-path") => read_path = Some(PathBuf::from(value)),
            Some("--write-path") => write_path = Some(PathBuf::from(value)),
            Some("--connect") => {
                connect_address = Some(value.to_string_lossy().parse::<SocketAddr>()?)
            }
            Some("--read-fd") => inherited_fd = Some(value.to_string_lossy().parse::<RawFd>()?),
            _ => return Err("unsupported sandbox probe option".into()),
        }
    }
    let read_path = read_path.ok_or("missing --read-path")?;
    let write_path = write_path.ok_or("missing --write-path")?;
    let connect_address = connect_address.ok_or("missing --connect")?;
    let inherited_fd = inherited_fd.ok_or("missing --read-fd")?;
    if !(3..=4096).contains(&inherited_fd) {
        return Err("inherited descriptor is outside the accepted range".into());
    }

    require_denied(fs::read(read_path), "filesystem read")?;
    require_denied(
        fs::write(write_path, b"sandbox violation"),
        "filesystem write",
    )?;
    match TcpStream::connect_timeout(&connect_address, Duration::from_millis(500)) {
        Err(error) if error.kind() == io::ErrorKind::PermissionDenied => {}
        Err(error) => return Err(format!("network denial was not established: {error}").into()),
        Ok(_) => return Err("sandbox permitted a network connection".into()),
    }

    let descriptor_path = PathBuf::from(format!("/dev/fd/{inherited_fd}"));
    let descriptor_bytes = fs::read(descriptor_path)?;
    if descriptor_bytes != b"sage-inherited-read-only-descriptor\n" {
        return Err("inherited descriptor contents did not match".into());
    }

    println!(
        "{{\"read_path\":\"denied\",\"write_path\":\"denied\",\"network\":\"denied\",\"inherited_fd\":\"readable\"}}"
    );
    Ok(())
}

#[cfg(all(feature = "sandbox-probe", target_os = "macos"))]
fn require_denied<T>(
    result: io::Result<T>,
    operation: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    match result {
        Err(error) if error.kind() == io::ErrorKind::PermissionDenied => Ok(()),
        Err(error) => Err(format!("{operation} denial was not established: {error}").into()),
        Ok(_) => Err(format!("sandbox permitted {operation}").into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn frames_round_trip_and_reject_zero_or_oversized_lengths() {
        let payload = br#"{"protocol_version":1,"kind":"hello","challenge":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"}"#;
        let mut encoded = Vec::new();
        write_frame(&mut encoded, payload).expect("write bounded frame");
        let mut input = Cursor::new(encoded);
        assert_eq!(
            read_frame(&mut input).expect("read frame"),
            Some(payload.to_vec())
        );
        assert_eq!(read_frame(&mut input).expect("clean eof"), None);

        for size in [0_u32, (MAX_FRAME_BYTES + 1) as u32] {
            let mut input = Cursor::new(size.to_be_bytes().to_vec());
            assert_eq!(
                read_frame(&mut input).unwrap_err().kind(),
                io::ErrorKind::InvalidData
            );
        }
    }

    #[test]
    fn handshake_rejects_unknown_fields_wrong_version_and_malformed_challenge() {
        let valid = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let accepted: WorkerRequest = serde_json::from_str(&format!(
            "{{\"protocol_version\":1,\"kind\":\"hello\",\"challenge\":\"{valid}\"}}"
        ))
        .expect("valid handshake schema");
        validate_request(&accepted).expect("valid handshake");

        let unknown = serde_json::from_str::<WorkerRequest>(&format!(
            "{{\"protocol_version\":1,\"kind\":\"hello\",\"challenge\":\"{valid}\",\"authority\":true}}"
        ));
        assert!(unknown.is_err());

        let wrong_version = WorkerRequest {
            protocol_version: 2,
            kind: "hello".into(),
            challenge: valid.into(),
        };
        assert!(validate_request(&wrong_version).is_err());
        let malformed = WorkerRequest {
            protocol_version: 1,
            kind: "hello".into(),
            challenge: "short".into(),
        };
        assert!(validate_request(&malformed).is_err());
    }
}
