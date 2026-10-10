//! Passive, bounded UPnP-AV renderer discovery for Sage's private-LAN device
//! fabric. Discovery results are untrusted observations; this crate grants no
//! pairing, disclosure, or control authority.

#![forbid(unsafe_code)]

use std::{
    collections::BTreeSet,
    net::{IpAddr, Ipv4Addr, SocketAddr},
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};

use anyhow::{Context, bail};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::{
    net::UdpSocket,
    task::JoinSet,
    time::{Instant, timeout_at},
};
use url::Url;

const SSDP_GROUP: SocketAddr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(239, 255, 255, 250)), 1900);
const MAX_SSDP_PACKET_BYTES: usize = 8 * 1024;
const MAX_DESCRIPTION_BYTES: usize = 64 * 1024;
const MAX_SOAP_RESPONSE_BYTES: usize = 16 * 1024;
const MAX_PROTOCOL_INFO_TEXT_BYTES: usize = 8 * 1024;
const MAX_PROTOCOL_INFO_ENTRIES: usize = 16;
const MAX_PROTOCOL_INFO_COMPONENT_BYTES: usize = 256;
const MAX_XML_NODES: usize = 4096;
const MAX_XML_DEPTH: usize = 32;
const MAX_DEVICES: usize = 16;
const MAX_FETCHES_IN_FLIGHT: usize = 4;
const DESCRIPTION_TIMEOUT: Duration = Duration::from_secs(2);
const MEDIA_RENDERER_PREFIX: &str = "urn:schemas-upnp-org:device:MediaRenderer:";
const AV_TRANSPORT_PREFIX: &str = "urn:schemas-upnp-org:service:AVTransport:";
const CONNECTION_MANAGER_PREFIX: &str = "urn:schemas-upnp-org:service:ConnectionManager:";

static DISCOVERY_ACTIVE: AtomicBool = AtomicBool::new(false);

const M_SEARCH: &[u8] = b"M-SEARCH * HTTP/1.1\r\nHOST: 239.255.255.250:1900\r\nMAN: \"ssdp:discover\"\r\nMX: 2\r\nST: ssdp:all\r\nUSER-AGENT: Sage/1.0 UPnP/1.1\r\n\r\n";

/// A passive LAN observation. Every device-provided field and endpoint stays
/// untrusted until a separate pairing and policy flow establishes authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RendererCandidate {
    source: SocketAddr,
    unique_device_name: String,
    device_type: String,
    friendly_name: String,
    manufacturer: Option<String>,
    model_name: Option<String>,
    av_transport_service_type: String,
    #[serde(skip_serializing)]
    av_transport_control_url: String,
    connection_manager_service_type: Option<String>,
    #[serde(skip_serializing)]
    connection_manager_control_url: Option<String>,
    description_sha256: String,
    observed_at: DateTime<Utc>,
    evidence_scope: String,
}

impl RendererCandidate {
    pub fn unique_device_name(&self) -> &str {
        &self.unique_device_name
    }

    pub fn description_sha256(&self) -> &str {
        &self.description_sha256
    }

    pub fn friendly_name(&self) -> &str {
        &self.friendly_name
    }
}

/// A read-only AVTransport observation. Values are device-reported facts, not
/// authenticated identity or permission to issue playback mutations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransportObservation {
    pub unique_device_name: String,
    pub description_sha256: String,
    pub state: TransportState,
    pub status: TransportStatus,
    pub current_speed: String,
    pub observed_at: DateTime<Utc>,
    pub evidence_scope: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransportState {
    Stopped,
    Playing,
    Transitioning,
    PausedPlayback,
    PausedRecording,
    Recording,
    NoMediaPresent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransportStatus {
    Ok,
    ErrorOccurred,
}

/// One untrusted protocol-info tuple reported by a renderer's ConnectionManager.
/// Fields are preserved verbatim after grammar and size checks; they are not a
/// promise that a stream will succeed or permission to send content.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProtocolInfo {
    pub protocol: String,
    pub network: String,
    pub content_format: String,
    pub additional_info: String,
}

impl ProtocolInfo {
    /// Check the standard protocol, network, and content-format fields.
    ///
    /// The UPnP ConnectionManager matching rule treats `*` as matching any
    /// value and recommends case-insensitive comparisons. Additional
    /// information is intentionally excluded from this basic comparison; it
    /// can carry setup or protection requirements that need separate policy.
    pub fn matches_standard_fields(&self, other: &Self) -> bool {
        let valid = |value: &str| {
            !value.is_empty()
                && value.len() <= MAX_PROTOCOL_INFO_COMPONENT_BYTES
                && !value.chars().any(char::is_control)
        };
        [
            (&self.protocol, &other.protocol),
            (&self.network, &other.network),
            (&self.content_format, &other.content_format),
        ]
        .into_iter()
        .all(|(left, right)| {
            valid(left)
                && valid(right)
                && (left == "*" || right == "*" || left.eq_ignore_ascii_case(right))
        })
    }
}

/// Read-only ConnectionManager capability evidence from one exact renderer
/// description. `sink` lists formats the renderer advertises as receivers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProtocolInfoObservation {
    pub unique_device_name: String,
    pub description_sha256: String,
    pub source: Vec<ProtocolInfo>,
    pub sink: Vec<ProtocolInfo>,
    pub observed_at: DateTime<Utc>,
    pub evidence_scope: String,
}

impl ProtocolInfoObservation {
    /// Return whether any currently advertised renderer sink tuple matches a
    /// proposed source tuple's standard fields. This is compatibility
    /// evidence only; it does not prove the source is reachable or that a
    /// stream can be established.
    pub fn sink_advertises_source(&self, source: &ProtocolInfo) -> bool {
        self.sink
            .iter()
            .any(|sink| source.matches_standard_fields(sink))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SsdpLocation {
    source: SocketAddr,
    unique_device_name: String,
    device_type: String,
    location: Url,
}

#[derive(Debug)]
struct XmlNode {
    name: String,
    text: String,
    children: Vec<usize>,
}

/// Send one IPv4 SSDP search, collect a bounded response set, and fetch only
/// same-host HTTP device descriptions from private or link-local addresses.
/// The result is discovery evidence only; no device command is issued.
pub async fn discover_media_renderers(
    discovery_window: Duration,
) -> anyhow::Result<Vec<RendererCandidate>> {
    discover_media_renderers_at(SSDP_GROUP, discovery_window).await
}

async fn discover_media_renderers_at(
    search_address: SocketAddr,
    discovery_window: Duration,
) -> anyhow::Result<Vec<RendererCandidate>> {
    if DISCOVERY_ACTIVE
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        bail!("another UPnP discovery is already active");
    }
    let _slot = DiscoverySlot;
    if discovery_window.is_zero() || discovery_window > Duration::from_secs(6) {
        bail!("UPnP discovery window must be between 1 ms and 6 seconds");
    }

    let standard_socket =
        std::net::UdpSocket::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 0))
            .context("bind bounded UPnP discovery socket")?;
    standard_socket
        .set_multicast_ttl_v4(2)
        .context("set SSDP multicast TTL")?;
    standard_socket
        .set_nonblocking(true)
        .context("configure nonblocking SSDP socket")?;
    let socket = UdpSocket::from_std(standard_socket).context("register SSDP socket")?;
    socket
        .send_to(M_SEARCH, search_address)
        .await
        .context("send local SSDP search")?;

    let deadline = Instant::now() + discovery_window;
    let mut locations = Vec::new();
    let mut seen = BTreeSet::new();
    let mut packet = [0u8; MAX_SSDP_PACKET_BYTES + 1];
    while locations.len() < MAX_DEVICES {
        let received = match timeout_at(deadline, socket.recv_from(&mut packet)).await {
            Ok(Ok(received)) => received,
            Ok(Err(error)) => return Err(error).context("receive SSDP response"),
            Err(_) => break,
        };
        let (length, source) = received;
        if let Some(location) = parse_ssdp_response(&packet[..length], source)
            && seen.insert((
                location.unique_device_name.clone(),
                location.location.as_str().to_owned(),
            ))
        {
            locations.push(location);
        }
    }

    let client = reqwest::Client::builder()
        .connect_timeout(DESCRIPTION_TIMEOUT)
        .timeout(DESCRIPTION_TIMEOUT)
        .redirect(reqwest::redirect::Policy::none())
        .no_proxy()
        .build()
        .context("build private-LAN-only description client")?;
    let observed_at = Utc::now();
    let mut pending = locations.into_iter();
    let mut fetches = JoinSet::new();
    for _ in 0..MAX_FETCHES_IN_FLIGHT {
        if let Some(location) = pending.next() {
            let client = client.clone();
            fetches.spawn(async move { fetch_candidate(&client, location, observed_at).await });
        }
    }

    let mut candidates = Vec::new();
    while let Some(result) = fetches.join_next().await {
        match result {
            Ok(Ok(Some(candidate))) => candidates.push(candidate),
            Ok(Ok(None)) => {}
            Ok(Err(_)) => {}
            Err(_) => {}
        }
        if let Some(location) = pending.next() {
            let client = client.clone();
            fetches.spawn(async move { fetch_candidate(&client, location, observed_at).await });
        }
    }
    candidates.sort_by(|left, right| left.unique_device_name.cmp(&right.unique_device_name));
    Ok(candidates)
}

/// Ask one previously discovered candidate for its current transport state.
/// This performs a read-only SOAP action; it does not pair with the renderer,
/// control playback, or create reusable authority. The renderer response is
/// bounded and remains an untrusted observation.
pub async fn observe_transport(
    candidate: &RendererCandidate,
) -> anyhow::Result<TransportObservation> {
    if !is_private_lan(candidate.source.ip())
        || !valid_av_transport_type(&candidate.av_transport_service_type)
    {
        bail!("renderer candidate is outside Sage's bounded private-LAN profile");
    }
    let control_url =
        parse_same_host_http_url(&candidate.av_transport_control_url, candidate.source.ip())
            .context("renderer AVTransport endpoint is not same-host private HTTP")?;
    let action = format!("{}#GetTransportInfo", candidate.av_transport_service_type);
    let body = format!(
        "<s:Envelope xmlns:s=\"http://schemas.xmlsoap.org/soap/envelope/\" s:encodingStyle=\"http://schemas.xmlsoap.org/soap/encoding/\"><s:Body><u:GetTransportInfo xmlns:u=\"{}\"><InstanceID>0</InstanceID></u:GetTransportInfo></s:Body></s:Envelope>",
        candidate.av_transport_service_type
    );
    let client = reqwest::Client::builder()
        .connect_timeout(DESCRIPTION_TIMEOUT)
        .timeout(DESCRIPTION_TIMEOUT)
        .redirect(reqwest::redirect::Policy::none())
        .no_proxy()
        .build()
        .context("build private-LAN-only AVTransport client")?;
    let mut response = client
        .post(control_url)
        .header(reqwest::header::CONTENT_TYPE, "text/xml; charset=\"utf-8\"")
        .header("SOAPACTION", action)
        .body(body)
        .send()
        .await
        .context("request renderer transport state")?;
    if !response.status().is_success() {
        bail!(
            "renderer AVTransport request returned HTTP {}",
            response.status()
        );
    }
    if response
        .content_length()
        .is_some_and(|length| length > MAX_SOAP_RESPONSE_BYTES as u64)
    {
        bail!("renderer AVTransport response exceeds Sage's size bound");
    }
    let mut response_body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .context("read renderer AVTransport response")?
    {
        if response_body.len().saturating_add(chunk.len()) > MAX_SOAP_RESPONSE_BYTES {
            bail!("renderer AVTransport response exceeds Sage's size bound");
        }
        response_body.extend_from_slice(&chunk);
    }
    let (state, status, current_speed) =
        parse_transport_response(&response_body).context("parse renderer AVTransport response")?;
    Ok(TransportObservation {
        unique_device_name: candidate.unique_device_name.clone(),
        description_sha256: candidate.description_sha256.clone(),
        state,
        status,
        current_speed,
        observed_at: Utc::now(),
        evidence_scope: "private_lan_untrusted_transport_observation".into(),
    })
}

/// Ask a previously discovered renderer for its advertised source and sink
/// protocol-info lists. This is a read-only SOAP action; the result is
/// untrusted compatibility evidence, not proof that a particular media path
/// works or permission to transmit content.
pub async fn observe_protocol_info(
    candidate: &RendererCandidate,
) -> anyhow::Result<ProtocolInfoObservation> {
    let (service_type, control_endpoint) = candidate
        .connection_manager_service_type
        .as_deref()
        .zip(candidate.connection_manager_control_url.as_deref())
        .context("renderer does not advertise a usable ConnectionManager service")?;
    if !is_private_lan(candidate.source.ip()) || !valid_connection_manager_type(service_type) {
        bail!("renderer candidate is outside Sage's bounded private-LAN profile");
    }
    let control_url = parse_same_host_http_url(control_endpoint, candidate.source.ip())
        .context("renderer ConnectionManager endpoint is not same-host private HTTP")?;
    let action = format!("{}#GetProtocolInfo", service_type);
    let body = format!(
        "<s:Envelope xmlns:s=\"http://schemas.xmlsoap.org/soap/envelope/\" s:encodingStyle=\"http://schemas.xmlsoap.org/soap/encoding/\"><s:Body><u:GetProtocolInfo xmlns:u=\"{}\"/></s:Body></s:Envelope>",
        service_type
    );
    let client = reqwest::Client::builder()
        .connect_timeout(DESCRIPTION_TIMEOUT)
        .timeout(DESCRIPTION_TIMEOUT)
        .redirect(reqwest::redirect::Policy::none())
        .no_proxy()
        .build()
        .context("build private-LAN ConnectionManager client")?;
    let mut response = client
        .post(control_url)
        .header(reqwest::header::CONTENT_TYPE, "text/xml; charset=\"utf-8\"")
        .header("SOAPACTION", action)
        .body(body)
        .send()
        .await
        .context("request renderer protocol information")?;
    if !response.status().is_success() {
        bail!(
            "renderer ConnectionManager request returned HTTP {}",
            response.status()
        );
    }
    if response
        .content_length()
        .is_some_and(|length| length > MAX_SOAP_RESPONSE_BYTES as u64)
    {
        bail!("renderer ConnectionManager response exceeds Sage's size bound");
    }
    let mut response_body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .context("read renderer ConnectionManager response")?
    {
        if response_body.len().saturating_add(chunk.len()) > MAX_SOAP_RESPONSE_BYTES {
            bail!("renderer ConnectionManager response exceeds Sage's size bound");
        }
        response_body.extend_from_slice(&chunk);
    }
    let (source, sink) = parse_protocol_info_response(&response_body)
        .context("parse renderer ConnectionManager response")?;
    Ok(ProtocolInfoObservation {
        unique_device_name: candidate.unique_device_name.clone(),
        description_sha256: candidate.description_sha256.clone(),
        source,
        sink,
        observed_at: Utc::now(),
        evidence_scope: "private_lan_untrusted_protocol_info".into(),
    })
}

fn parse_protocol_info_response(
    bytes: &[u8],
) -> anyhow::Result<(Vec<ProtocolInfo>, Vec<ProtocolInfo>)> {
    let Some(document) = parse_xml_document(bytes)? else {
        bail!("renderer response is not bounded well-formed XML");
    };
    let root = &document[document[0].children[0]];
    if local_name(&root.name) != "Envelope" {
        bail!("renderer response has no SOAP envelope");
    }
    let body = unique_child_named(root, &document, "Body")
        .context("renderer SOAP envelope has no unique body")?;
    let response = unique_child_named(body, &document, "GetProtocolInfoResponse")
        .context("renderer SOAP body has no GetProtocolInfo response")?;
    let source = unique_protocol_info_child(response, &document, "Source")?;
    let sink = unique_protocol_info_child(response, &document, "Sink")?;
    Ok((source, sink))
}

fn unique_protocol_info_child(
    response: &XmlNode,
    document: &[XmlNode],
    name: &str,
) -> anyhow::Result<Vec<ProtocolInfo>> {
    let field = unique_child_named(response, document, name)
        .with_context(|| format!("renderer protocol-info response has no unique {name}"))?;
    if !field.children.is_empty() || field.text.len() > MAX_PROTOCOL_INFO_TEXT_BYTES {
        bail!("renderer protocol-info {name} field is nested or oversized");
    }
    parse_protocol_info_list(field.text.trim())
        .with_context(|| format!("renderer protocol-info {name} list is malformed"))
}

fn parse_protocol_info_list(value: &str) -> anyhow::Result<Vec<ProtocolInfo>> {
    if value.is_empty() {
        return Ok(Vec::new());
    }
    let mut protocols = Vec::new();
    for item in split_protocol_info_csv(value)? {
        let mut fields = item.split(':');
        let protocol = bounded_protocol_component(fields.next().unwrap_or_default())?;
        let network = bounded_protocol_component(fields.next().unwrap_or_default())?;
        let content_format = bounded_protocol_component(fields.next().unwrap_or_default())?;
        let additional_info = bounded_protocol_component(fields.next().unwrap_or_default())?;
        if fields.next().is_some() {
            bail!("renderer protocol-info entry has too many fields");
        }
        protocols.push(ProtocolInfo {
            protocol,
            network,
            content_format,
            additional_info,
        });
    }
    Ok(protocols)
}

fn split_protocol_info_csv(value: &str) -> anyhow::Result<Vec<String>> {
    let mut entries = Vec::new();
    let mut entry = String::new();
    let mut characters = value.chars();
    while let Some(character) = characters.next() {
        match character {
            ',' => {
                if entry.is_empty() {
                    bail!("renderer protocol-info CSV contains an empty entry");
                }
                entries.push(std::mem::take(&mut entry));
            }
            '\\' => match characters.next() {
                Some('\\') => entry.push('\\'),
                Some(',') => entry.push(','),
                _ => bail!("renderer protocol-info CSV contains an invalid escape"),
            },
            _ => entry.push(character),
        }
        if entry.len() > MAX_PROTOCOL_INFO_COMPONENT_BYTES * 4 {
            bail!("renderer protocol-info entry exceeds Sage's component bound");
        }
    }
    if entry.is_empty() {
        bail!("renderer protocol-info CSV contains an empty entry");
    }
    entries.push(entry);
    if entries.len() > MAX_PROTOCOL_INFO_ENTRIES {
        bail!("renderer protocol-info list exceeds Sage's entry bound");
    }
    Ok(entries)
}

fn bounded_protocol_component(value: &str) -> anyhow::Result<String> {
    let value = value.trim();
    if value.is_empty()
        || value.len() > MAX_PROTOCOL_INFO_COMPONENT_BYTES
        || value.chars().any(char::is_control)
    {
        bail!("renderer protocol-info component is empty, oversized, or contains controls");
    }
    Ok(value.to_owned())
}

fn parse_transport_response(
    bytes: &[u8],
) -> anyhow::Result<(TransportState, TransportStatus, String)> {
    let Some(document) = parse_xml_document(bytes)? else {
        bail!("renderer response is not bounded well-formed XML");
    };
    let root = &document[document[0].children[0]];
    if local_name(&root.name) != "Envelope" {
        bail!("renderer response has no SOAP envelope");
    }
    let body = unique_child_named(root, &document, "Body")
        .context("renderer SOAP envelope has no unique body")?;
    let response = unique_child_named(body, &document, "GetTransportInfoResponse")
        .context("renderer SOAP body has no GetTransportInfo response")?;
    let state = match unique_bounded_child_text(response, &document, "CurrentTransportState", 32)
        .as_deref()
    {
        Some("STOPPED") => TransportState::Stopped,
        Some("PLAYING") => TransportState::Playing,
        Some("TRANSITIONING") => TransportState::Transitioning,
        Some("PAUSED_PLAYBACK") => TransportState::PausedPlayback,
        Some("PAUSED_RECORDING") => TransportState::PausedRecording,
        Some("RECORDING") => TransportState::Recording,
        Some("NO_MEDIA_PRESENT") => TransportState::NoMediaPresent,
        _ => bail!("renderer reported an unsupported transport state"),
    };
    let status = match unique_bounded_child_text(response, &document, "CurrentTransportStatus", 32)
        .as_deref()
    {
        Some("OK") => TransportStatus::Ok,
        Some("ERROR_OCCURRED") => TransportStatus::ErrorOccurred,
        _ => bail!("renderer reported an unsupported transport status"),
    };
    let current_speed = unique_bounded_child_text(response, &document, "CurrentSpeed", 16)
        .filter(|speed| valid_transport_speed(speed))
        .context("renderer reported an invalid transport speed")?;
    Ok((state, status, current_speed))
}

fn valid_transport_speed(speed: &str) -> bool {
    fn decimal(value: &str) -> bool {
        let value = value.strip_prefix('-').unwrap_or(value);
        if value.is_empty() {
            return false;
        }
        let mut pieces = value.split('.');
        let whole = pieces.next().unwrap_or_default();
        let fraction = pieces.next();
        !whole.is_empty()
            && whole.bytes().all(|byte| byte.is_ascii_digit())
            && fraction.is_none_or(|fraction| {
                !fraction.is_empty() && fraction.bytes().all(|byte| byte.is_ascii_digit())
            })
            && pieces.next().is_none()
    }

    let mut pieces = speed.split('/');
    let Some(numerator) = pieces.next() else {
        return false;
    };
    match pieces.next() {
        None => decimal(numerator),
        Some(denominator) => {
            decimal(numerator)
                && !denominator.is_empty()
                && denominator.bytes().all(|byte| byte.is_ascii_digit())
                && denominator.parse::<u32>().is_ok_and(|value| value != 0)
                && pieces.next().is_none()
        }
    }
}

struct DiscoverySlot;

impl Drop for DiscoverySlot {
    fn drop(&mut self) {
        DISCOVERY_ACTIVE.store(false, Ordering::Release);
    }
}

fn parse_ssdp_response(bytes: &[u8], source: SocketAddr) -> Option<SsdpLocation> {
    if bytes.is_empty() || bytes.len() > MAX_SSDP_PACKET_BYTES || !is_private_lan(source.ip()) {
        return None;
    }
    let text = std::str::from_utf8(bytes).ok()?;
    if text
        .chars()
        .any(|character| character.is_control() && !matches!(character, '\r' | '\n' | '\t'))
    {
        return None;
    }
    let mut lines = text.split("\r\n");
    if lines.next()? != "HTTP/1.1 200 OK" {
        return None;
    }
    let mut headers = BTreeSet::new();
    let mut location = None;
    let mut search_target = None;
    let mut unique_device_name = None;
    let mut terminated = false;
    for line in lines {
        if line.is_empty() {
            terminated = true;
            break;
        }
        if line.starts_with([' ', '\t']) {
            return None;
        }
        let (name, value) = line.split_once(':')?;
        if name.is_empty()
            || !name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
        {
            return None;
        }
        let canonical_name = name.to_ascii_lowercase();
        if !headers.insert(canonical_name.clone()) {
            return None;
        }
        let value = value.trim();
        if value.is_empty() || value.bytes().any(|byte| byte.is_ascii_control()) {
            return None;
        }
        match canonical_name.as_str() {
            "location" => location = Some(value),
            "st" => search_target = Some(value),
            "usn" => unique_device_name = Some(value),
            _ => {}
        }
    }
    if !terminated {
        return None;
    }
    let search_target = search_target?;
    let device_type = search_target
        .strip_prefix(MEDIA_RENDERER_PREFIX)
        .filter(|version| {
            !version.is_empty() && version.bytes().all(|byte| byte.is_ascii_digit())
        })?;
    let device_type = format!("{MEDIA_RENDERER_PREFIX}{device_type}");
    let unique_device_name = unique_device_name?
        .split_once("::")
        .map_or(unique_device_name?, |(udn, _)| udn);
    if !unique_device_name
        .get(..5)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("uuid:"))
        || unique_device_name.len() > 256
        || unique_device_name
            .bytes()
            .any(|byte| byte.is_ascii_control())
    {
        return None;
    }
    let location = parse_same_host_http_url(location?, source.ip())?;
    Some(SsdpLocation {
        source,
        unique_device_name: unique_device_name.to_owned(),
        device_type,
        location,
    })
}

async fn fetch_candidate(
    client: &reqwest::Client,
    location: SsdpLocation,
    observed_at: DateTime<Utc>,
) -> anyhow::Result<Option<RendererCandidate>> {
    let mut response = client
        .get(location.location.clone())
        .header(reqwest::header::ACCEPT, "application/xml, text/xml")
        .send()
        .await
        .context("fetch local UPnP device description")?;
    if !response.status().is_success() {
        return Ok(None);
    }
    if response
        .content_length()
        .is_some_and(|length| length > MAX_DESCRIPTION_BYTES as u64)
    {
        return Ok(None);
    }
    let mut description = Vec::new();
    while let Some(chunk) = response.chunk().await? {
        if description.len().saturating_add(chunk.len()) > MAX_DESCRIPTION_BYTES {
            return Ok(None);
        }
        description.extend_from_slice(&chunk);
    }
    let digest = Sha256::digest(&description);
    let Some(parsed) = parse_device_description(&description, &location)? else {
        return Ok(None);
    };
    Ok(Some(RendererCandidate {
        source: location.source,
        unique_device_name: location.unique_device_name,
        device_type: location.device_type,
        friendly_name: parsed.friendly_name,
        manufacturer: parsed.manufacturer,
        model_name: parsed.model_name,
        av_transport_service_type: parsed.service_type,
        av_transport_control_url: parsed.control_url,
        connection_manager_service_type: parsed.connection_manager_service_type,
        connection_manager_control_url: parsed.connection_manager_control_url,
        description_sha256: encode_hex(&digest),
        observed_at,
        evidence_scope: "private_lan_passive_discovery".into(),
    }))
}

struct ParsedDeviceDescription {
    friendly_name: String,
    manufacturer: Option<String>,
    model_name: Option<String>,
    service_type: String,
    control_url: String,
    connection_manager_service_type: Option<String>,
    connection_manager_control_url: Option<String>,
}

fn parse_device_description(
    bytes: &[u8],
    location: &SsdpLocation,
) -> anyhow::Result<Option<ParsedDeviceDescription>> {
    let Some(document) = parse_xml_document(bytes)? else {
        return Ok(None);
    };
    let root = &document[document[0].children[0]];
    if local_name(&root.name) != "root" {
        return Ok(None);
    }
    let Some(device) = child_named(root, &document, "device") else {
        return Ok(None);
    };
    let device_type = child_text(device, &document, "deviceType");
    let udn = child_text(device, &document, "UDN");
    if device_type.as_deref() != Some(location.device_type.as_str())
        || udn.as_deref() != Some(location.unique_device_name.as_str())
    {
        return Ok(None);
    }
    let service_base = if let Some(value) = child_text(root, &document, "URLBase") {
        let Some(base) = resolve_same_host_url(&location.location, &value, location.source.ip())
        else {
            return Ok(None);
        };
        base
    } else {
        location.location.clone()
    };
    let Some(friendly_name) =
        child_text(device, &document, "friendlyName").and_then(|text| bounded_text(&text, 256))
    else {
        return Ok(None);
    };
    let manufacturer =
        child_text(device, &document, "manufacturer").and_then(|text| bounded_text(&text, 256));
    let model_name =
        child_text(device, &document, "modelName").and_then(|text| bounded_text(&text, 256));
    let Some(service_list) = child_named(device, &document, "serviceList") else {
        return Ok(None);
    };
    let mut av_transport = None;
    let mut connection_manager = None;
    for service in children_named(service_list, &document, "service") {
        let service_type = child_text(service, &document, "serviceType");
        let Some(service_type) = service_type else {
            continue;
        };
        let is_av_transport = valid_av_transport_type(&service_type);
        let is_connection_manager = valid_connection_manager_type(&service_type);
        if !is_av_transport && !is_connection_manager {
            continue;
        }
        let Some(control_path) = child_text(service, &document, "controlURL")
            .and_then(|value| bounded_text(&value, 2048))
        else {
            continue;
        };
        let Some(control_url) =
            resolve_same_host_url(&service_base, &control_path, location.source.ip())
        else {
            continue;
        };
        if is_av_transport && av_transport.is_none() {
            av_transport = Some((service_type.clone(), control_url.to_string()));
        }
        if is_connection_manager && connection_manager.is_none() {
            connection_manager = Some((service_type, control_url.to_string()));
        }
    }
    let Some((service_type, control_url)) = av_transport else {
        return Ok(None);
    };
    let (connection_manager_service_type, connection_manager_control_url) = connection_manager
        .map_or((None, None), |(service_type, control_url)| {
            (Some(service_type), Some(control_url))
        });
    Ok(Some(ParsedDeviceDescription {
        friendly_name,
        manufacturer,
        model_name,
        service_type,
        control_url,
        connection_manager_service_type,
        connection_manager_control_url,
    }))
}

// Parse once into a small, closed element arena. DTDs and custom entities are
// rejected, preventing external fetches and entity-expansion attacks.
fn parse_xml_document(bytes: &[u8]) -> anyhow::Result<Option<Vec<XmlNode>>> {
    if bytes.is_empty() || bytes.len() > MAX_DESCRIPTION_BYTES {
        return Ok(None);
    }
    let text = match std::str::from_utf8(bytes) {
        Ok(text) => text,
        Err(_) => return Ok(None),
    };
    if text
        .chars()
        .any(|character| !valid_xml_character(character))
    {
        return Ok(None);
    }
    let mut nodes = vec![XmlNode {
        name: String::new(),
        text: String::new(),
        children: Vec::new(),
    }];
    let mut stack = vec![0usize];
    let mut cursor = 0usize;
    let mut saw_root = false;
    while cursor < text.len() {
        if text.as_bytes()[cursor] != b'<' {
            let end = text[cursor..]
                .find('<')
                .map_or(text.len(), |offset| cursor + offset);
            let decoded = match decode_xml_text(&text[cursor..end]) {
                Ok(decoded) => decoded,
                Err(_) => return Ok(None),
            };
            if stack.len() == 1 && !decoded.trim().is_empty() {
                return Ok(None);
            }
            if let Some(node) = nodes.get_mut(*stack.last().expect("synthetic root")) {
                if node.text.len().saturating_add(decoded.len()) > MAX_DESCRIPTION_BYTES {
                    return Ok(None);
                }
                node.text.push_str(&decoded);
            }
            cursor = end;
            continue;
        }

        if text[cursor..].starts_with("<!--") {
            let Some(end) = text[cursor + 4..].find("-->") else {
                return Ok(None);
            };
            if text[cursor + 4..cursor + 4 + end].contains("--") {
                return Ok(None);
            }
            cursor += 4 + end + 3;
            continue;
        }
        if text[cursor..].starts_with("<![CDATA[") {
            let Some(end) = text[cursor + 9..].find("]]>") else {
                return Ok(None);
            };
            let start = cursor + 9;
            let content = &text[start..start + end];
            if stack.len() == 1 {
                return Ok(None);
            }
            let node = nodes
                .get_mut(*stack.last().expect("open root"))
                .expect("current XML element");
            if node.text.len().saturating_add(content.len()) > MAX_DESCRIPTION_BYTES {
                return Ok(None);
            }
            node.text.push_str(content);
            cursor = start + end + 3;
            continue;
        }
        if text[cursor..].starts_with("<?") {
            let Some(end) = text[cursor + 2..].find("?>") else {
                return Ok(None);
            };
            cursor += 2 + end + 2;
            continue;
        }
        if text[cursor..].starts_with("<!") {
            return Ok(None);
        }

        let Some(end) = find_tag_end(text, cursor + 1) else {
            return Ok(None);
        };
        let token = text[cursor + 1..end].trim();
        if let Some(closing) = token.strip_prefix('/') {
            let name = closing.trim();
            if name.is_empty()
                || name.bytes().any(|byte| byte.is_ascii_whitespace())
                || stack.len() <= 1
            {
                return Ok(None);
            }
            let node_index = *stack.last().expect("open element");
            if nodes[node_index].name != name {
                return Ok(None);
            }
            stack.pop();
        } else {
            let self_closing = token.ends_with('/');
            let token = token.strip_suffix('/').unwrap_or(token).trim_end();
            let Some((name, rest)) = split_xml_name(token) else {
                return Ok(None);
            };
            if !valid_xml_name(name) || !valid_xml_attributes(rest) || nodes.len() >= MAX_XML_NODES
            {
                return Ok(None);
            }
            if stack.len() > MAX_XML_DEPTH {
                return Ok(None);
            }
            if stack.len() == 1 {
                if saw_root {
                    return Ok(None);
                }
                saw_root = true;
            }
            let parent = *stack.last().expect("synthetic root");
            let index = nodes.len();
            nodes.push(XmlNode {
                name: name.to_owned(),
                text: String::new(),
                children: Vec::new(),
            });
            nodes[parent].children.push(index);
            if !self_closing {
                stack.push(index);
            }
        }
        cursor = end + 1;
    }
    if stack.len() != 1 || !saw_root || nodes[0].children.len() != 1 {
        return Ok(None);
    }
    Ok(Some(nodes))
}

// Kept as a tiny wrapper to make malformed documents a normal non-match.
fn find_tag_end(text: &str, start: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut quote = None;
    for (offset, byte) in bytes.iter().copied().enumerate().skip(start) {
        match (quote, byte) {
            (Some(active), value) if active == value => quote = None,
            (None, b'\'' | b'"') => quote = Some(byte),
            (None, b'>') => return Some(offset),
            _ => {}
        }
    }
    None
}

fn split_xml_name(token: &str) -> Option<(&str, &str)> {
    let end = token
        .find(|character: char| character.is_ascii_whitespace())
        .unwrap_or(token.len());
    let name = &token[..end];
    (!name.is_empty()).then_some((name, &token[end..]))
}

fn valid_xml_name(name: &str) -> bool {
    let mut bytes = name.bytes();
    let Some(first) = bytes.next() else {
        return false;
    };
    (first.is_ascii_alphabetic() || matches!(first, b'_' | b':'))
        && bytes
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b':' | b'-' | b'.'))
        && name.matches(':').count() <= 1
}

fn valid_xml_attributes(mut input: &str) -> bool {
    let mut seen = BTreeSet::new();
    loop {
        input = input.trim_start();
        if input.is_empty() {
            return true;
        }
        let Some(end) =
            input.find(|character: char| character.is_ascii_whitespace() || character == '=')
        else {
            return false;
        };
        let name = &input[..end];
        if !valid_xml_name(name) || !seen.insert(name) {
            return false;
        }
        input = input[end..].trim_start();
        let Some(rest) = input.strip_prefix('=') else {
            return false;
        };
        input = rest.trim_start();
        let Some(quote) = input.as_bytes().first().copied() else {
            return false;
        };
        if !matches!(quote, b'\'' | b'"') {
            return false;
        }
        let Some(end) = input[1..].find(char::from(quote)) else {
            return false;
        };
        let value = &input[1..1 + end];
        if value.contains('<') || decode_xml_text(value).is_err() {
            return false;
        }
        input = &input[2 + end..];
    }
}

fn decode_xml_text(input: &str) -> anyhow::Result<String> {
    let mut result = String::with_capacity(input.len());
    let mut rest = input;
    while let Some(index) = rest.find('&') {
        result.push_str(&rest[..index]);
        let reference_start = index + 1;
        let after_ampersand = &rest[reference_start..];
        let Some(end) = after_ampersand.find(';') else {
            bail!("unterminated XML entity reference");
        };
        let entity = &after_ampersand[..end];
        let character = match entity {
            "amp" => '&',
            "lt" => '<',
            "gt" => '>',
            "quot" => '"',
            "apos" => '\'',
            _ if entity.starts_with("#x") => char::from_u32(u32::from_str_radix(&entity[2..], 16)?)
                .filter(|character| valid_xml_character(*character))
                .context("invalid XML character reference")?,
            _ if entity.starts_with('#') => char::from_u32(entity[1..].parse::<u32>()?)
                .filter(|character| valid_xml_character(*character))
                .context("invalid XML character reference")?,
            _ => bail!("custom XML entities are not supported"),
        };
        result.push(character);
        rest = &after_ampersand[end + 1..];
    }
    result.push_str(rest);
    Ok(result)
}

fn valid_xml_character(character: char) -> bool {
    matches!(character as u32, 0x9 | 0xA | 0xD | 0x20..=0xD7FF | 0xE000..=0xFFFD | 0x10000..=0x10FFFF)
}

fn local_name(name: &str) -> &str {
    name.rsplit_once(':').map_or(name, |(_, local)| local)
}

fn child_named<'a>(node: &'a XmlNode, arena: &'a [XmlNode], name: &str) -> Option<&'a XmlNode> {
    node.children
        .iter()
        .map(|index| &arena[*index])
        .find(|child| local_name(&child.name) == name)
}

fn unique_child_named<'a>(
    node: &'a XmlNode,
    arena: &'a [XmlNode],
    name: &str,
) -> Option<&'a XmlNode> {
    let mut matching = children_named(node, arena, name);
    let first = matching.next()?;
    matching.next().is_none().then_some(first)
}

fn unique_bounded_child_text(
    node: &XmlNode,
    arena: &[XmlNode],
    name: &str,
    maximum_bytes: usize,
) -> Option<String> {
    let child = unique_child_named(node, arena, name)?;
    child
        .children
        .is_empty()
        .then(|| bounded_text(&child.text, maximum_bytes))?
}

fn children_named<'a>(
    node: &'a XmlNode,
    arena: &'a [XmlNode],
    name: &str,
) -> impl Iterator<Item = &'a XmlNode> {
    node.children
        .iter()
        .map(|index| &arena[*index])
        .filter(move |child| local_name(&child.name) == name)
}

fn child_text(node: &XmlNode, arena: &[XmlNode], name: &str) -> Option<String> {
    let child = child_named(node, arena, name)?;
    child
        .children
        .is_empty()
        .then(|| child.text.trim().to_owned())
}

fn bounded_text(text: &str, maximum_bytes: usize) -> Option<String> {
    let text = text.trim();
    if text.is_empty()
        || text.len() > maximum_bytes
        || text
            .chars()
            .any(|character| character.is_control() && !matches!(character, '\t' | '\n' | '\r'))
    {
        None
    } else {
        Some(text.to_owned())
    }
}

fn valid_av_transport_type(service_type: &str) -> bool {
    valid_upnp_service_type(service_type, AV_TRANSPORT_PREFIX)
}

fn valid_connection_manager_type(service_type: &str) -> bool {
    valid_upnp_service_type(service_type, CONNECTION_MANAGER_PREFIX)
}

fn valid_upnp_service_type(service_type: &str, prefix: &str) -> bool {
    service_type.strip_prefix(prefix).is_some_and(|version| {
        !version.is_empty()
            && version.bytes().all(|byte| byte.is_ascii_digit())
            && version.parse::<u32>().is_ok_and(|version| version > 0)
    })
}

fn parse_same_host_http_url(value: &str, source_ip: IpAddr) -> Option<Url> {
    let url = Url::parse(value).ok()?;
    let host_matches = url.host_str()? == source_ip.to_string();
    (url.scheme() == "http"
        && host_matches
        && url.username().is_empty()
        && url.password().is_none()
        && url.query().is_none()
        && url.fragment().is_none()
        && url.port_or_known_default().is_some()
        && !url.path().is_empty())
    .then_some(url)
}

fn resolve_same_host_url(base: &Url, reference: &str, source_ip: IpAddr) -> Option<Url> {
    let url = base.join(reference).ok()?;
    (url.scheme() == "http"
        && url.host_str() == Some(source_ip.to_string().as_str())
        && url.username().is_empty()
        && url.password().is_none()
        && url.query().is_none()
        && url.fragment().is_none())
    .then_some(url)
}

fn is_private_lan(address: IpAddr) -> bool {
    match address {
        IpAddr::V4(address) => {
            address.is_private() || address.is_link_local() || address.is_loopback()
        }
        IpAddr::V6(address) => {
            address.is_unique_local() || address.is_unicast_link_local() || address.is_loopback()
        }
    }
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        result.push(HEX[(byte >> 4) as usize] as char);
        result.push(HEX[(byte & 0x0f) as usize] as char);
    }
    result
}

#[cfg(test)]
mod tests {
    use std::net::{IpAddr, Ipv4Addr};

    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::{TcpListener, UdpSocket},
    };

    use super::*;

    const DEVICE: &str = r#"<?xml version="1.0" encoding="utf-8"?>
<root xmlns="urn:schemas-upnp-org:device-1-0"><device><deviceType>urn:schemas-upnp-org:device:MediaRenderer:1</deviceType><friendlyName>Living Room &amp; TV</friendlyName><manufacturer>Example</manufacturer><modelName>Renderer 1</modelName><UDN>uuid:01234567-89ab-cdef-0123-456789abcdef</UDN><serviceList><service><serviceType>urn:schemas-upnp-org:service:AVTransport:1</serviceType><controlURL>/upnp/control/av</controlURL></service><service><serviceType>urn:schemas-upnp-org:service:ConnectionManager:1</serviceType><controlURL>/upnp/control/cm</controlURL></service></serviceList></device></root>"#;

    fn location() -> SsdpLocation {
        SsdpLocation {
            source: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(192, 168, 1, 20)), 1900),
            unique_device_name: "uuid:01234567-89ab-cdef-0123-456789abcdef".into(),
            device_type: format!("{MEDIA_RENDERER_PREFIX}1"),
            location: Url::parse("http://192.168.1.20:1400/rootDesc.xml").unwrap(),
        }
    }

    fn candidate(control_url: String) -> RendererCandidate {
        RendererCandidate {
            source: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 1900),
            unique_device_name: "uuid:01234567-89ab-cdef-0123-456789abcdef".into(),
            device_type: format!("{MEDIA_RENDERER_PREFIX}1"),
            friendly_name: "Living Room & TV".into(),
            manufacturer: Some("Example".into()),
            model_name: Some("Renderer 1".into()),
            av_transport_service_type: format!("{AV_TRANSPORT_PREFIX}1"),
            av_transport_control_url: control_url,
            connection_manager_service_type: Some(format!("{CONNECTION_MANAGER_PREFIX}1")),
            connection_manager_control_url: Some("http://127.0.0.1:1400/upnp/control/cm".into()),
            description_sha256: "a".repeat(64),
            observed_at: Utc::now(),
            evidence_scope: "private_lan_passive_discovery".into(),
        }
    }

    #[test]
    fn ssdp_parser_accepts_only_private_renderer_responses_and_same_host_locations() {
        let packet = b"HTTP/1.1 200 OK\r\nCACHE-CONTROL: max-age=1800\r\nST: urn:schemas-upnp-org:device:MediaRenderer:1\r\nUSN: uuid:01234567-89ab-cdef-0123-456789abcdef::urn:schemas-upnp-org:device:MediaRenderer:1\r\nLOCATION: http://192.168.1.20:1400/rootDesc.xml\r\n\r\n";
        let source = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(192, 168, 1, 20)), 1900);
        let parsed = parse_ssdp_response(packet, source).unwrap();
        assert_eq!(
            parsed.location.as_str(),
            "http://192.168.1.20:1400/rootDesc.xml"
        );
        assert_eq!(parsed.source, source);
        assert!(parse_ssdp_response(packet, "8.8.8.8:1900".parse().unwrap()).is_none());

        let external = packet
            .windows(b"http://192.168.1.20".len())
            .position(|window| window == b"http://192.168.1.20")
            .map(|index| {
                let mut changed = packet.to_vec();
                changed.splice(
                    index..index + b"http://192.168.1.20".len(),
                    b"http://203.0.113.9".iter().copied(),
                );
                changed
            })
            .unwrap();
        assert!(parse_ssdp_response(&external, source).is_none());

        let duplicate = b"HTTP/1.1 200 OK\r\nST: urn:schemas-upnp-org:device:MediaRenderer:1\r\nST: urn:schemas-upnp-org:device:MediaRenderer:1\r\nUSN: uuid:x\r\nLOCATION: http://192.168.1.20/root.xml\r\n\r\n";
        assert!(parse_ssdp_response(duplicate, source).is_none());
    }

    #[test]
    fn description_parser_binds_device_identity_and_only_same_origin_av_transport() {
        let parsed = parse_device_description(DEVICE.as_bytes(), &location())
            .unwrap()
            .unwrap();
        assert_eq!(parsed.friendly_name, "Living Room & TV");
        assert_eq!(parsed.manufacturer.as_deref(), Some("Example"));
        assert_eq!(parsed.model_name.as_deref(), Some("Renderer 1"));
        assert_eq!(
            parsed.service_type,
            "urn:schemas-upnp-org:service:AVTransport:1"
        );
        assert_eq!(
            parsed.control_url,
            "http://192.168.1.20:1400/upnp/control/av"
        );

        let mut changed_identity =
            DEVICE.replace("uuid:01234567-89ab-cdef-0123-456789abcdef", "uuid:other");
        assert!(
            parse_device_description(changed_identity.as_bytes(), &location())
                .unwrap()
                .is_none()
        );

        changed_identity = DEVICE.replace("/upnp/control/av", "http://192.168.1.21:1400/control");
        assert!(
            parse_device_description(changed_identity.as_bytes(), &location())
                .unwrap()
                .is_none()
        );

        let with_url_base = DEVICE
            .replace(
                "<device>",
                "<URLBase>http://192.168.1.20:1401/media/</URLBase><device>",
            )
            .replace("/upnp/control/av", "control/av")
            .replace("/upnp/control/cm", "control/cm");
        let parsed = parse_device_description(with_url_base.as_bytes(), &location())
            .unwrap()
            .unwrap();
        assert_eq!(
            parsed.control_url,
            "http://192.168.1.20:1401/media/control/av"
        );
        assert_eq!(
            parsed.connection_manager_service_type.as_deref(),
            Some("urn:schemas-upnp-org:service:ConnectionManager:1")
        );
        assert_eq!(
            parsed.connection_manager_control_url.as_deref(),
            Some("http://192.168.1.20:1401/media/control/cm")
        );

        let external_base = with_url_base.replace("192.168.1.20:1401", "192.168.1.21:1401");
        assert!(
            parse_device_description(external_base.as_bytes(), &location())
                .unwrap()
                .is_none()
        );

        let nested_name = DEVICE.replace("Living Room &amp; TV", "Living <alias>Room</alias> TV");
        assert!(
            parse_device_description(nested_name.as_bytes(), &location())
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn renderer_summary_never_serializes_its_private_lan_control_endpoint() {
        let candidate = candidate("http://127.0.0.1:1400/upnp/control/av".into());
        let summary = serde_json::to_value(candidate).unwrap();
        assert!(summary.get("av_transport_control_url").is_none());
        assert!(summary.get("connection_manager_control_url").is_none());
    }

    #[tokio::test]
    async fn protocol_info_observation_uses_read_only_same_host_soap() {
        let server = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = server.local_addr().unwrap();
        let mut candidate = candidate(format!("http://{address}/upnp/control/av"));
        candidate.connection_manager_control_url =
            Some(format!("http://{address}/upnp/control/cm"));
        let server = tokio::spawn(async move {
            let (mut stream, _) = server.accept().await.unwrap();
            let mut request = Vec::new();
            let mut chunk = [0_u8; 1024];
            let body_start = loop {
                let length = stream.read(&mut chunk).await.unwrap();
                assert_ne!(length, 0, "client closed before sending SOAP request");
                request.extend_from_slice(&chunk[..length]);
                if let Some(end) = request.windows(4).position(|window| window == b"\r\n\r\n") {
                    break end + 4;
                }
                assert!(request.len() <= 8 * 1024, "request headers are bounded");
            };
            let header_end = body_start - 4;
            let headers = std::str::from_utf8(&request[..header_end]).unwrap();
            let content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().unwrap())
                })
                .unwrap();
            while request.len() < body_start + content_length {
                let length = stream.read(&mut chunk).await.unwrap();
                assert_ne!(length, 0, "client closed before sending the SOAP body");
                request.extend_from_slice(&chunk[..length]);
            }
            let headers = std::str::from_utf8(&request[..header_end]).unwrap();
            let body =
                std::str::from_utf8(&request[body_start..body_start + content_length]).unwrap();
            assert!(headers.starts_with("POST /upnp/control/cm HTTP/1.1\r\n"));
            assert!(headers.lines().any(|line| {
                line.eq_ignore_ascii_case(&format!(
                    "SOAPACTION: {CONNECTION_MANAGER_PREFIX}1#GetProtocolInfo"
                ))
            }));
            assert!(body.contains("<u:GetProtocolInfo"));
            assert!(!body.contains("SetAVTransportURI"));
            assert!(!body.contains("<InstanceID>"));
            let response_body = b"<s:Envelope xmlns:s=\"http://schemas.xmlsoap.org/soap/envelope/\"><s:Body><u:GetProtocolInfoResponse xmlns:u=\"urn:schemas-upnp-org:service:ConnectionManager:1\"><Source>http-get:*:video/mp4:DLNA.ORG_OP=01</Source><Sink>http-get:*:video/mp4:*,http-get:*:audio/mpeg:DLNA.ORG_OP=01</Sink></u:GetProtocolInfoResponse></s:Body></s:Envelope>";
            let headers = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/xml\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                response_body.len()
            );
            stream.write_all(headers.as_bytes()).await.unwrap();
            stream.write_all(response_body).await.unwrap();
            stream.shutdown().await.unwrap();
        });

        let observation = observe_protocol_info(&candidate).await.unwrap();
        server.await.unwrap();
        assert_eq!(observation.unique_device_name, candidate.unique_device_name);
        assert_eq!(observation.description_sha256, candidate.description_sha256);
        assert_eq!(observation.source.len(), 1);
        assert_eq!(observation.sink.len(), 2);
        assert_eq!(observation.sink[0].content_format, "video/mp4");
        assert_eq!(observation.sink[0].additional_info, "*");
        assert_eq!(observation.sink[1].content_format, "audio/mpeg");
        assert_eq!(
            observation.evidence_scope,
            "private_lan_untrusted_protocol_info"
        );
    }

    #[test]
    fn protocol_info_parser_accepts_empty_lists_and_rejects_ambiguous_or_unbounded_lists() {
        let response = b"<s:Envelope><s:Body><u:GetProtocolInfoResponse><Source> \n </Source><Sink>http-get:*:video/mp4:DLNA.ORG_OP=01</Sink></u:GetProtocolInfoResponse></s:Body></s:Envelope>";
        let (source, sink) = parse_protocol_info_response(response).unwrap();
        assert!(source.is_empty());
        assert_eq!(sink.len(), 1);

        let duplicate_source = b"<s:Envelope><s:Body><u:GetProtocolInfoResponse><Source></Source><Source></Source><Sink></Sink></u:GetProtocolInfoResponse></s:Body></s:Envelope>";
        assert!(parse_protocol_info_response(duplicate_source).is_err());
        assert!(parse_protocol_info_list("http-get:*:video/mp4").is_err());
        assert!(parse_protocol_info_list("http-get:*::*").is_err());
        assert!(parse_protocol_info_list("http-get:*:video/mp4:info:extra").is_err());
        assert!(parse_protocol_info_list("http-get:*:video/mp4:*,").is_err());
        let escaped = parse_protocol_info_list(
            r"http-get:*:video/mp4:upnp.org_note=one\,two,http-get:*:audio/mpeg:*",
        )
        .unwrap();
        assert_eq!(escaped.len(), 2);
        assert_eq!(escaped[0].additional_info, "upnp.org_note=one,two");
        assert!(parse_protocol_info_list(r"http-get:*:video/mp4:bad\q").is_err());
        assert!(
            parse_protocol_info_list(&format!(
                "http-get:*:video/mp4:{}",
                "x".repeat(MAX_PROTOCOL_INFO_COMPONENT_BYTES + 1)
            ))
            .is_err()
        );
        let too_many = (0..=MAX_PROTOCOL_INFO_ENTRIES)
            .map(|_| "http-get:*:video/mp4:*")
            .collect::<Vec<_>>()
            .join(",");
        assert!(parse_protocol_info_list(&too_many).is_err());
        assert!(parse_protocol_info_list("http-get:*:video/mp4:bad\nvalue").is_err());
    }

    #[test]
    fn protocol_compatibility_matches_only_standard_fields_and_honors_wildcards() {
        let source = ProtocolInfo {
            protocol: "HTTP-GET".into(),
            network: "*".into(),
            content_format: "video/mp4".into(),
            additional_info: "upnp.org_profile=source-profile".into(),
        };
        let sink = ProtocolInfo {
            protocol: "http-get".into(),
            network: "https".into(),
            content_format: "VIDEO/MP4".into(),
            additional_info: "upnp.org_profile=sink-profile".into(),
        };
        assert!(source.matches_standard_fields(&sink));

        let mismatched = ProtocolInfo {
            content_format: "video/webm".into(),
            ..sink.clone()
        };
        assert!(!source.matches_standard_fields(&mismatched));
        let invalid = ProtocolInfo {
            protocol: String::new(),
            ..sink.clone()
        };
        assert!(!source.matches_standard_fields(&invalid));

        let observation = ProtocolInfoObservation {
            unique_device_name: "uuid:renderer".into(),
            description_sha256: "a".repeat(64),
            source: Vec::new(),
            sink: vec![sink],
            observed_at: Utc::now(),
            evidence_scope: "private_lan_untrusted_protocol_info".into(),
        };
        assert!(observation.sink_advertises_source(&source));
        assert!(!observation.sink_advertises_source(&mismatched));
    }

    #[tokio::test]
    async fn protocol_info_observation_rejects_missing_service_and_off_host_endpoint() {
        let mut no_service = candidate("http://127.0.0.1:9/control".into());
        no_service.connection_manager_service_type = None;
        assert!(observe_protocol_info(&no_service).await.is_err());

        let mut off_host = candidate("http://127.0.0.1:9/control".into());
        off_host.connection_manager_control_url = Some("http://8.8.8.8:80/control".into());
        assert!(observe_protocol_info(&off_host).await.is_err());
    }

    #[test]
    fn xml_parser_rejects_dtd_custom_entities_bad_nesting_and_unbounded_shapes() {
        let dtd = b"<!DOCTYPE root [<!ENTITY x SYSTEM 'file:///etc/passwd'>]><root/>";
        assert!(parse_xml_document(dtd).unwrap().is_none());
        assert!(
            parse_xml_document(b"<root><device></root>")
                .unwrap()
                .is_none()
        );
        assert!(
            parse_xml_document(b"<root>&custom;</root>")
                .unwrap()
                .is_none()
        );
        assert!(
            parse_xml_document(&vec![b'x'; MAX_DESCRIPTION_BYTES + 1])
                .unwrap()
                .is_none()
        );

        let deep = format!(
            "{}x{}",
            "<a>".repeat(MAX_XML_DEPTH + 2),
            "</a>".repeat(MAX_XML_DEPTH + 2)
        );
        assert!(parse_xml_document(deep.as_bytes()).unwrap().is_none());
    }

    #[test]
    fn url_validation_refuses_credentials_queries_fragments_and_origin_changes() {
        let source = IpAddr::V4(Ipv4Addr::new(192, 168, 1, 20));
        assert!(parse_same_host_http_url("http://user@192.168.1.20/a", source).is_none());
        assert!(parse_same_host_http_url("http://192.168.1.20/a?x=1", source).is_none());
        assert!(parse_same_host_http_url("https://192.168.1.20/a", source).is_none());
        assert!(parse_same_host_http_url("http://8.8.8.8/a", source).is_none());
    }

    #[tokio::test]
    async fn bounded_search_fetches_a_loopback_renderer_description() {
        let ssdp = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let ssdp_address = ssdp.local_addr().unwrap();
        let description_server = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let description_address = description_server.local_addr().unwrap();
        let location = format!("http://{description_address}/rootDesc.xml");
        let ssdp_response = format!(
            "HTTP/1.1 200 OK\r\nST: {MEDIA_RENDERER_PREFIX}1\r\nUSN: uuid:01234567-89ab-cdef-0123-456789abcdef::{MEDIA_RENDERER_PREFIX}1\r\nLOCATION: {location}\r\n\r\n"
        );

        let ssdp_server = tokio::spawn(async move {
            let mut request = [0_u8; 2048];
            let (length, peer) = ssdp.recv_from(&mut request).await.unwrap();
            assert!(request[..length].starts_with(b"M-SEARCH * HTTP/1.1\r\n"));
            ssdp.send_to(ssdp_response.as_bytes(), peer).await.unwrap();
        });
        let description_server = tokio::spawn(async move {
            let (mut stream, _) = description_server.accept().await.unwrap();
            let body = DEVICE.as_bytes();
            let headers = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/xml\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            stream.write_all(headers.as_bytes()).await.unwrap();
            stream.write_all(body).await.unwrap();
            stream.shutdown().await.unwrap();
        });

        let candidates = discover_media_renderers_at(ssdp_address, Duration::from_secs(1))
            .await
            .unwrap();
        ssdp_server.await.unwrap();
        description_server.await.unwrap();

        assert_eq!(candidates.len(), 1);
        let candidate = &candidates[0];
        assert_eq!(candidate.source.ip(), IpAddr::V4(Ipv4Addr::LOCALHOST));
        assert_eq!(
            candidate.unique_device_name,
            "uuid:01234567-89ab-cdef-0123-456789abcdef"
        );
        assert_eq!(candidate.friendly_name, "Living Room & TV");
        assert_eq!(
            candidate.connection_manager_service_type.as_deref(),
            Some("urn:schemas-upnp-org:service:ConnectionManager:1")
        );
        assert_eq!(candidate.evidence_scope, "private_lan_passive_discovery");
        assert_eq!(
            Url::parse(&candidate.av_transport_control_url)
                .unwrap()
                .host_str(),
            Some("127.0.0.1")
        );
    }

    #[tokio::test]
    async fn transport_observation_uses_same_host_soap_and_returns_untrusted_typed_state() {
        let server = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = server.local_addr().unwrap();
        let candidate = candidate(format!("http://{address}/upnp/control/av"));
        let server = tokio::spawn(async move {
            let (mut stream, _) = server.accept().await.unwrap();
            let mut request = Vec::new();
            let mut chunk = [0_u8; 1024];
            let body_start = loop {
                let length = stream.read(&mut chunk).await.unwrap();
                assert_ne!(length, 0, "client closed before sending SOAP request");
                request.extend_from_slice(&chunk[..length]);
                if let Some(end) = request.windows(4).position(|window| window == b"\r\n\r\n") {
                    break end + 4;
                }
                assert!(request.len() <= 8 * 1024, "request headers are bounded");
            };
            let header_end = body_start - 4;
            let headers = std::str::from_utf8(&request[..header_end]).unwrap();
            let content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().unwrap())
                })
                .unwrap();
            while request.len() < body_start + content_length {
                let length = stream.read(&mut chunk).await.unwrap();
                assert_ne!(length, 0, "client closed before sending the SOAP body");
                request.extend_from_slice(&chunk[..length]);
            }
            let headers = std::str::from_utf8(&request[..header_end]).unwrap();
            let body =
                std::str::from_utf8(&request[body_start..body_start + content_length]).unwrap();
            assert!(headers.starts_with("POST /upnp/control/av HTTP/1.1\r\n"));
            assert!(headers.lines().any(|line| {
                line.eq_ignore_ascii_case(&format!(
                    "SOAPACTION: {AV_TRANSPORT_PREFIX}1#GetTransportInfo"
                ))
            }));
            assert!(body.contains("<u:GetTransportInfo"));
            assert!(body.contains("<InstanceID>0</InstanceID>"));
            assert!(!body.contains("SetAVTransportURI"));
            let response_body = b"<s:Envelope xmlns:s=\"http://schemas.xmlsoap.org/soap/envelope/\"><s:Body><u:GetTransportInfoResponse xmlns:u=\"urn:schemas-upnp-org:service:AVTransport:1\"><CurrentTransportState>PLAYING</CurrentTransportState><CurrentTransportStatus>OK</CurrentTransportStatus><CurrentSpeed>1</CurrentSpeed></u:GetTransportInfoResponse></s:Body></s:Envelope>";
            let headers = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/xml\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                response_body.len()
            );
            stream.write_all(headers.as_bytes()).await.unwrap();
            stream.write_all(response_body).await.unwrap();
            stream.shutdown().await.unwrap();
        });

        let observation = observe_transport(&candidate).await.unwrap();
        server.await.unwrap();
        assert_eq!(observation.unique_device_name, candidate.unique_device_name);
        assert_eq!(observation.description_sha256, candidate.description_sha256);
        assert_eq!(observation.state, TransportState::Playing);
        assert_eq!(observation.status, TransportStatus::Ok);
        assert_eq!(observation.current_speed, "1");
        assert_eq!(
            observation.evidence_scope,
            "private_lan_untrusted_transport_observation"
        );
    }

    #[tokio::test]
    async fn transport_observation_rejects_off_host_endpoints_and_unknown_device_states() {
        let mut candidate = candidate("http://127.0.0.1:9/control".into());
        candidate.source = "8.8.8.8:1900".parse().unwrap();
        assert!(observe_transport(&candidate).await.is_err());

        let response = b"<s:Envelope><s:Body><u:GetTransportInfoResponse><CurrentTransportState>UNRECOGNIZED</CurrentTransportState><CurrentTransportStatus>OK</CurrentTransportStatus><CurrentSpeed>1</CurrentSpeed></u:GetTransportInfoResponse></s:Body></s:Envelope>";
        assert!(parse_transport_response(response).is_err());
        let duplicate_state = b"<s:Envelope><s:Body><u:GetTransportInfoResponse><CurrentTransportState>PLAYING</CurrentTransportState><CurrentTransportState>STOPPED</CurrentTransportState><CurrentTransportStatus>OK</CurrentTransportStatus><CurrentSpeed>1</CurrentSpeed></u:GetTransportInfoResponse></s:Body></s:Envelope>";
        assert!(parse_transport_response(duplicate_state).is_err());
        let invalid_speed = b"<s:Envelope><s:Body><u:GetTransportInfoResponse><CurrentTransportState>PLAYING</CurrentTransportState><CurrentTransportStatus>OK</CurrentTransportStatus><CurrentSpeed>1//2</CurrentSpeed></u:GetTransportInfoResponse></s:Body></s:Envelope>";
        assert!(parse_transport_response(invalid_speed).is_err());
        assert!(valid_transport_speed("1/2"));
        assert!(valid_transport_speed("-0.5"));
        assert!(!valid_transport_speed("1//2"));
        assert!(!valid_transport_speed("1/0"));
    }
}
