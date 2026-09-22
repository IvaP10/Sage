//! Egress policy for configured inference routes. Resolve once, validate every
//! address, and pin those addresses in the client that performs the request.
use crate::{CoreError, CoreResult};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};

pub fn endpoint(value: &str) -> CoreResult<url::Url> {
    let url = url::Url::parse(value.trim()).map_err(|_| denied("Invalid provider URL"))?;
    if !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || url.host().is_none()
    {
        return Err(denied(
            "Provider URL must have a host and no credentials, query, or fragment",
        ));
    }
    let loopback = exact_loopback(&url);
    if url.scheme() != "https" && !(url.scheme() == "http" && loopback) {
        return Err(denied("Use HTTPS, or HTTP with an exact loopback host"));
    }
    if let Some(host) = url.host() {
        let address = match host {
            url::Host::Ipv4(ip) => Some(IpAddr::V4(ip)),
            url::Host::Ipv6(ip) => Some(IpAddr::V6(ip)),
            _ => None,
        };
        if address.is_some_and(|ip| !public_address(ip) && !loopback) {
            return Err(denied(
                "Private, link-local, metadata, and special-use provider addresses are blocked",
            ));
        }
    }
    Ok(url)
}

pub fn exact_loopback(url: &url::Url) -> bool {
    matches!(url.host(), Some(url::Host::Domain("localhost")))
        || matches!(url.host(), Some(url::Host::Ipv4(ip)) if ip == Ipv4Addr::LOCALHOST)
        || matches!(url.host(), Some(url::Host::Ipv6(ip)) if ip == Ipv6Addr::LOCALHOST)
}

pub fn public_address(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => {
            let [a, b, c, _] = ip.octets();
            !ip.is_private()
                && !ip.is_loopback()
                && !ip.is_link_local()
                && !ip.is_broadcast()
                && !ip.is_documentation()
                && !ip.is_multicast()
                && a != 0
                && a < 240
                && !(a == 100 && (64..=127).contains(&b))
                && !(a == 192 && b == 0 && c == 0)
                && !(a == 198 && (b == 18 || b == 19))
                && !(a == 192 && b == 88 && c == 99)
        }
        IpAddr::V6(ip) => {
            // Only global-unicast space; exclude embedded IPv4 and transition
            // ranges whose ultimate connection destination is ambiguous.
            let s = ip.segments();
            (s[0] & 0xe000) == 0x2000
                && s[0] != 0x2002
                && !(s[0] == 0x2001 && s[1] < 0x0200)
                && !(s[0] == 0x2001 && s[1] == 0x0db8)
                && s[0] != 0x3fff
        }
    }
}

pub async fn provider_client(value: &str) -> CoreResult<reqwest::Client> {
    let url = endpoint(value)?;
    let host = url
        .host_str()
        .ok_or_else(|| denied("Missing provider host"))?
        .trim_matches(['[', ']']);
    let port = url
        .port_or_known_default()
        .ok_or_else(|| denied("Missing provider port"))?;
    let addresses: Vec<SocketAddr> = if host == "localhost" {
        vec![SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port)]
    } else {
        tokio::time::timeout(
            std::time::Duration::from_secs(10),
            tokio::net::lookup_host((host, port)),
        )
        .await
        .map_err(|_| denied("Provider DNS lookup timed out"))?
        .map_err(|_| denied("Provider DNS lookup failed"))?
        .take(32)
        .collect()
    };
    if addresses.is_empty()
        || addresses.iter().any(|addr| {
            if exact_loopback(&url) {
                !addr.ip().is_loopback()
            } else {
                !public_address(addr.ip())
            }
        })
    {
        return Err(denied("Provider DNS resolved to a disallowed destination"));
    }
    reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .resolve_to_addrs(host, &addresses)
        .connect_timeout(std::time::Duration::from_secs(15))
        .timeout(std::time::Duration::from_secs(90))
        .build()
        .map_err(|_| denied("Provider network client could not be created"))
}

fn denied(message: &str) -> CoreError {
    CoreError::InvalidAction(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn public_fetch_cannot_use_provider_loopback_exceptions_or_forged_evidence() {
        use sha2::{Digest, Sha256};
        for url in [
            "http://127.0.0.1:8080/",
            "https://localhost/",
            "https://api.example.com:8443/",
            "https://169.254.169.254/",
            "https://api.example.com/?private=value",
        ] {
            assert!(public_url(url).is_err(), "{url}");
        }
        let mut result = FetchedResource {
            url: "https://example.com/".into(),
            status: 200,
            media_type: "text/plain".into(),
            text: "document".into(),
            sha256: format!("{:x}", Sha256::digest(b"document")),
        };
        result.validate().unwrap();
        result.text = "different contents".into();
        assert!(result.validate().is_err());
        result.text = "document".into();
        result.status = 302;
        assert!(result.validate().is_err());
        result.status = 200;
        result.media_type = "application/pdf".into();
        assert!(result.validate().is_err());
    }
    #[test]
    fn blocks_private_and_ambiguous_destinations() {
        for address in [
            "0.0.0.0",
            "10.0.0.2",
            "100.100.100.200",
            "127.0.0.2",
            "169.254.169.254",
            "192.168.1.1",
            "198.19.1.1",
            "224.0.0.1",
            "::1",
            "::ffff:127.0.0.1",
            "fc00::1",
            "fe80::1",
            "2001:db8::1",
            "2002:7f00:1::",
        ] {
            assert!(!public_address(address.parse().unwrap()), "{address}");
        }
        for url in [
            "https://169.254.169.254/",
            "https://[::ffff:127.0.0.1]/",
            "http://localhost.attacker.test/",
            "https://user:pass@provider.test/",
        ] {
            assert!(endpoint(url).is_err(), "{url}");
        }
        assert!(endpoint("http://127.0.0.1:8080/v1").is_ok());
        assert!(endpoint("https://api.example.com/v1").is_ok());
    }
}

/// Public research never uses browser cookies, connector credentials, proxy
/// settings or the loopback exception available to configured model providers.
pub fn public_url(value: &str) -> CoreResult<url::Url> {
    let url = endpoint(value)?;
    if url.scheme() != "https" || exact_loopback(&url) || url.port_or_known_default() != Some(443) {
        return Err(denied(
            "Public research requires an HTTPS destination on port 443",
        ));
    }
    Ok(url)
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FetchedResource {
    pub url: String,
    pub status: u16,
    pub media_type: String,
    pub text: String,
    pub sha256: String,
}
impl FetchedResource {
    pub fn validate(&self) -> CoreResult<()> {
        use sha2::{Digest, Sha256};
        public_url(&self.url)?;
        if !(200..300).contains(&self.status)
            || !matches!(
                self.media_type.as_str(),
                "text/plain" | "text/html" | "application/json" | "application/ld+json"
            )
            || self.text.len() > 1_048_576
            || format!("{:x}", Sha256::digest(self.text.as_bytes())) != self.sha256
        {
            return Err(CoreError::VerificationFailed(
                "Public response evidence failed validation".into(),
            ));
        }
        Ok(())
    }
}

pub async fn fetch_public(value: &str, max_bytes: u64) -> CoreResult<FetchedResource> {
    use sha2::{Digest, Sha256};
    let url = public_url(value)?;
    if !(1..=1_048_576).contains(&max_bytes) {
        return Err(denied("Invalid download budget"));
    }
    let client = provider_client(url.as_str()).await?;
    let mut response = client
        .get(url.clone())
        .header(
            reqwest::header::ACCEPT,
            "text/plain, text/html, application/json",
        )
        .header(reqwest::header::USER_AGENT, "Sage/2 public-research")
        .send()
        .await
        .map_err(|_| CoreError::ExecutionFailed("Public HTTPS request failed".into()))?;
    let status = response.status().as_u16();
    if !response.status().is_success() || response.url() != &url {
        return Err(CoreError::ExecutionFailed(
            "Public request failed or redirected; approve the actual destination separately".into(),
        ));
    }
    if response
        .content_length()
        .is_some_and(|length| length > max_bytes)
    {
        return Err(CoreError::ExecutionFailed(
            "Download exceeds its byte budget".into(),
        ));
    }
    let media_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|h| h.to_str().ok())
        .unwrap_or("")
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    if !matches!(
        media_type.as_str(),
        "text/plain" | "text/html" | "application/json" | "application/ld+json"
    ) {
        return Err(CoreError::ExecutorUnavailable(
            "This document type needs a qualified isolated parser".into(),
        ));
    }
    let mut bytes = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|_| CoreError::ExecutionFailed("Download was interrupted".into()))?
    {
        if bytes.len() + chunk.len() > max_bytes as usize {
            return Err(CoreError::ExecutionFailed(
                "Download exceeded its byte budget".into(),
            ));
        }
        bytes.extend_from_slice(&chunk);
    }
    let text = String::from_utf8(bytes).map_err(|_| {
        CoreError::ExecutorUnavailable("This encoding needs an isolated document parser".into())
    })?;
    let document = FetchedResource {
        url: url.to_string(),
        status,
        media_type,
        sha256: format!("{:x}", Sha256::digest(text.as_bytes())),
        text,
    };
    document.validate()?;
    Ok(document)
}
