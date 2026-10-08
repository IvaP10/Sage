//! First-party signature and content verification for Sage's pinned Qwen3.5
//! weight package. This boundary is independent of Core and contains no task
//! broker, storage, networking, or host execution code, so a restricted
//! inference process can reuse the same verifier without linking those
//! authorities. The package format is local-file based and never downloads
//! weights or selects a model endpoint.

use std::{
    collections::{BTreeMap, BTreeSet},
    io::{Cursor, Read, Seek, SeekFrom},
};

use ed25519_dalek::{Signature, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use thiserror::Error;

pub mod safetensors;

pub type PackageResult<T> = Result<T, PackageError>;

#[derive(Debug, Error)]
pub enum PackageError {
    #[error("invalid signed model package: {0}")]
    Invalid(String),
    #[error("model package I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("model package serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),
}

pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

pub fn f16_to_f32(bits: u16) -> f32 {
    sage_kernels::f16_bits_to_f32(bits)
}

pub const QWEN35_CHECKPOINT_REVISION: &str = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a";
const MAX_MANIFEST_BYTES: usize = 64 * 1024;
const MAX_SMALL_ARTIFACT_BYTES: u64 = 64 * 1024 * 1024;
const MAX_WEIGHT_ARTIFACT_BYTES: u64 = 16 * 1024 * 1024 * 1024;
const HASH_CHUNK_BYTES: usize = 1024 * 1024;
const SIGNATURE_DOMAIN: &[u8] = b"sage:qwen35-package-manifest:v2\0";

pub const QWEN35_PACKAGE_ARTIFACTS: [&str; 6] = [
    "config.json",
    "model.safetensors-00001-of-00002.safetensors",
    "model.safetensors-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer.json",
];

/// Closed Sage package envelope. The signature covers its schema version,
/// immutable checkpoint revision, signer key ID, and exact artifact digests.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qwen35PackageManifest {
    pub schema_version: u32,
    pub checkpoint_revision: String,
    pub key_id: String,
    pub artifacts: BTreeMap<String, String>,
    pub signature_hex: String,
}

#[derive(Serialize)]
struct SigningBody<'a> {
    schema_version: u32,
    checkpoint_revision: &'a str,
    key_id: &'a str,
    artifacts: &'a BTreeMap<String, String>,
}

/// Signature-verified manifest. This proves only that a configured Sage key
/// signed the file digests; a model must still pass tensor validation,
/// numerical/task evaluation, memory and latency gates before availability.
#[derive(Debug, Clone)]
pub struct VerifiedQwen35Package {
    checkpoint_revision: String,
    key_id: String,
    artifacts: BTreeMap<String, String>,
    manifest_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedArtifactDigest {
    name: String,
    sha256: String,
    bytes: u64,
}

impl VerifiedArtifactDigest {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

#[derive(Debug)]
pub struct VerifiedPackageArtifact<R> {
    name: String,
    sha256: String,
    bytes: u64,
    reader: R,
}

impl<R> VerifiedPackageArtifact<R> {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Consume an owned verification result to obtain the exact reader that
    /// was hashed. The handle is the only way to construct this result type.
    pub fn into_verified_reader(self) -> R {
        self.reader
    }
}

impl VerifiedPackageArtifact<Cursor<Vec<u8>>> {
    /// Consume an owned verification result for a small artifact such as the
    /// config, tokenizer, or tensor index.
    pub fn into_verified_bytes(self) -> Vec<u8> {
        self.into_verified_reader().into_inner()
    }
}

impl Qwen35PackageManifest {
    /// Return stable Sage signing bytes so an offline release tool can sign
    /// the exact closed manifest body without signing its signature field.
    pub fn signing_bytes(&self) -> PackageResult<Vec<u8>> {
        self.validate_body()?;
        let body = serde_json::to_vec(&SigningBody {
            schema_version: self.schema_version,
            checkpoint_revision: &self.checkpoint_revision,
            key_id: &self.key_id,
            artifacts: &self.artifacts,
        })?;
        let mut message = Vec::with_capacity(SIGNATURE_DOMAIN.len() + body.len());
        message.extend_from_slice(SIGNATURE_DOMAIN);
        message.extend_from_slice(&body);
        Ok(message)
    }

    /// Verify the manifest against the application's compiled trusted key
    /// set. A key supplied by an untrusted package or user is not a trust root.
    pub fn verify_signature(
        &self,
        trusted_keys: &BTreeMap<String, VerifyingKey>,
    ) -> PackageResult<VerifiedQwen35Package> {
        self.validate_body()?;
        let signature_bytes = decode_signature(&self.signature_hex)?;
        let signature = Signature::from_bytes(&signature_bytes);
        let key = trusted_keys.get(&self.key_id).ok_or_else(|| {
            PackageError::Invalid("Qwen package signer is not in Sage's trusted key set".into())
        })?;
        key.verify_strict(&self.signing_bytes()?, &signature)
            .map_err(|_| PackageError::Invalid("Qwen package signature is invalid".into()))?;
        let serialized = serde_json::to_vec(self)?;
        Ok(VerifiedQwen35Package {
            checkpoint_revision: self.checkpoint_revision.clone(),
            key_id: self.key_id.clone(),
            artifacts: self.artifacts.clone(),
            manifest_sha256: format!("{:x}", Sha256::digest(serialized)),
        })
    }

    fn validate_body(&self) -> PackageResult<()> {
        if self.schema_version != 2
            || self.checkpoint_revision != QWEN35_CHECKPOINT_REVISION
            || self.key_id.trim().is_empty()
            || self.key_id.len() > 128
            || self.key_id.chars().any(char::is_control)
            || self.artifacts.len() != QWEN35_PACKAGE_ARTIFACTS.len()
            || self
                .artifacts
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>()
                != QWEN35_PACKAGE_ARTIFACTS.into_iter().collect()
            || self.artifacts.values().any(|digest| !valid_digest(digest))
        {
            return Err(PackageError::Invalid(
                "Qwen package manifest does not match Sage's pinned closed profile".into(),
            ));
        }
        if self.signature_hex.len() > 128
            || (!self.signature_hex.is_empty()
                && (self.signature_hex.len() != 128
                    || !self
                        .signature_hex
                        .bytes()
                        .all(|byte| byte.is_ascii_hexdigit())))
        {
            return Err(PackageError::Invalid(
                "Qwen package signature encoding is invalid".into(),
            ));
        }
        let encoded_size = serde_json::to_vec(self)?.len();
        if encoded_size > MAX_MANIFEST_BYTES {
            return Err(PackageError::Invalid(
                "Qwen package manifest exceeds Sage's 64 KiB bound".into(),
            ));
        }
        Ok(())
    }
}

impl VerifiedQwen35Package {
    pub fn checkpoint_revision(&self) -> &str {
        &self.checkpoint_revision
    }

    pub fn key_id(&self) -> &str {
        &self.key_id
    }

    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }

    pub fn require_owned_receipt<R>(
        &self,
        expected_name: &str,
        receipt: &VerifiedPackageArtifact<R>,
    ) -> PackageResult<()> {
        let expected_digest = self.artifacts.get(expected_name).ok_or_else(|| {
            PackageError::Invalid("Artifact name is not covered by the signed package".into())
        })?;
        let maximum_bytes = if expected_name.ends_with(".safetensors") {
            MAX_WEIGHT_ARTIFACT_BYTES
        } else {
            MAX_SMALL_ARTIFACT_BYTES
        };
        if receipt.name != expected_name
            || &receipt.sha256 != expected_digest
            || receipt.bytes == 0
            || receipt.bytes > maximum_bytes
        {
            return Err(PackageError::Invalid(
                "Package artifact receipt is not the owned signed artifact requested".into(),
            ));
        }
        Ok(())
    }

    /// Hash one already-open package artifact in bounded chunks and restore
    /// the caller's original file offset on both success and failure.
    pub fn verify_artifact<R: Read + Seek>(
        &self,
        name: &str,
        reader: &mut R,
    ) -> PackageResult<VerifiedArtifactDigest> {
        let expected = self.artifacts.get(name).ok_or_else(|| {
            PackageError::Invalid("Artifact name is not covered by the signed package".into())
        })?;
        let maximum = if name.ends_with(".safetensors") {
            MAX_WEIGHT_ARTIFACT_BYTES
        } else {
            MAX_SMALL_ARTIFACT_BYTES
        };
        let original_position = reader.stream_position()?;
        let result = hash_open_file(reader, maximum);
        let restore_result = reader.seek(SeekFrom::Start(original_position));
        let (actual, bytes) = result?;
        restore_result?;
        if &actual != expected {
            return Err(PackageError::Invalid(format!(
                "Signed Qwen package artifact digest mismatch: {name}"
            )));
        }
        Ok(VerifiedArtifactDigest {
            name: name.to_owned(),
            sha256: actual,
            bytes,
        })
    }

    /// Verify and retain ownership of the same open artifact handle. A future
    /// loader can consume the returned handle directly, rather than reopening
    /// a path after verification and creating a verify-then-swap gap.
    pub fn verify_owned_artifact<R: Read + Seek>(
        &self,
        name: &str,
        mut reader: R,
    ) -> PackageResult<VerifiedPackageArtifact<R>> {
        let receipt = self.verify_artifact(name, &mut reader)?;
        Ok(VerifiedPackageArtifact {
            name: receipt.name,
            sha256: receipt.sha256,
            bytes: receipt.bytes,
            reader,
        })
    }

    /// Read, hash, and retain one small artifact's exact bytes in a single
    /// pass. This avoids verifying a config/index/tokenizer and then reopening
    /// or rereading a potentially changed source before parsing it.
    pub fn read_verified_small_artifact<R: Read + Seek>(
        &self,
        name: &str,
        reader: &mut R,
    ) -> PackageResult<VerifiedPackageArtifact<Cursor<Vec<u8>>>> {
        let expected = self.artifacts.get(name).ok_or_else(|| {
            PackageError::Invalid("Artifact name is not covered by the signed package".into())
        })?;
        if name.ends_with(".safetensors") {
            return Err(PackageError::Invalid(
                "Weight shards must use bounded streaming verification".into(),
            ));
        }
        let original_position = reader.stream_position()?;
        let result = read_and_hash_open_file(reader, MAX_SMALL_ARTIFACT_BYTES);
        let restore_result = reader.seek(SeekFrom::Start(original_position));
        let (bytes, actual) = result?;
        restore_result?;
        if &actual != expected {
            return Err(PackageError::Invalid(format!(
                "Signed Qwen package artifact digest mismatch: {name}"
            )));
        }
        let byte_count = u64::try_from(bytes.len()).map_err(|_| {
            PackageError::Invalid("Package artifact size exceeds platform bounds".into())
        })?;
        Ok(VerifiedPackageArtifact {
            name: name.to_owned(),
            sha256: actual,
            bytes: byte_count,
            reader: Cursor::new(bytes),
        })
    }
}

fn hash_open_file<R: Read + Seek>(
    reader: &mut R,
    maximum_bytes: u64,
) -> PackageResult<(String, u64)> {
    reader.seek(SeekFrom::Start(0))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; HASH_CHUNK_BYTES];
    let mut total = 0_u64;
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        total = total
            .checked_add(u64::try_from(count).map_err(|_| {
                PackageError::Invalid("Package artifact read exceeded platform bounds".into())
            })?)
            .filter(|bytes| *bytes <= maximum_bytes)
            .ok_or_else(|| {
                PackageError::Invalid("Package artifact exceeds Sage's size limit".into())
            })?;
        hasher.update(&buffer[..count]);
    }
    if total == 0 {
        return Err(PackageError::Invalid("Package artifact is empty".into()));
    }
    Ok((format!("{:x}", hasher.finalize()), total))
}

fn read_and_hash_open_file<R: Read + Seek>(
    reader: &mut R,
    maximum_bytes: u64,
) -> PackageResult<(Vec<u8>, String)> {
    reader.seek(SeekFrom::Start(0))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; HASH_CHUNK_BYTES];
    let mut bytes = Vec::new();
    let mut total = 0_u64;
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        total = total
            .checked_add(u64::try_from(count).map_err(|_| {
                PackageError::Invalid("Package artifact read exceeded platform bounds".into())
            })?)
            .filter(|size| *size <= maximum_bytes)
            .ok_or_else(|| {
                PackageError::Invalid("Package artifact exceeds Sage's size limit".into())
            })?;
        hasher.update(&buffer[..count]);
        bytes.extend_from_slice(&buffer[..count]);
    }
    if total == 0 {
        return Err(PackageError::Invalid("Package artifact is empty".into()));
    }
    Ok((bytes, format!("{:x}", hasher.finalize())))
}

fn decode_signature(value: &str) -> PackageResult<[u8; 64]> {
    if value.len() != 128 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(PackageError::Invalid(
            "Qwen package signature must contain exactly 64 encoded bytes".into(),
        ));
    }
    let mut signature = [0_u8; 64];
    let (pairs, _) = value.as_bytes().as_chunks::<2>();
    for (index, pair) in pairs.iter().enumerate() {
        let pair = std::str::from_utf8(pair)
            .map_err(|_| PackageError::Invalid("Qwen signature hex is malformed".into()))?;
        signature[index] = u8::from_str_radix(pair, 16)
            .map_err(|_| PackageError::Invalid("Qwen signature hex is malformed".into()))?;
    }
    Ok(signature)
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use ed25519_dalek::{Signer, SigningKey};

    use super::*;

    fn fixture() -> (Qwen35PackageManifest, SigningKey) {
        let key = SigningKey::from_bytes(&[7; 32]);
        let artifacts = QWEN35_PACKAGE_ARTIFACTS
            .into_iter()
            .map(|name| {
                (
                    name.to_owned(),
                    format!("{:x}", Sha256::digest(name.as_bytes())),
                )
            })
            .collect();
        let mut manifest = Qwen35PackageManifest {
            schema_version: 2,
            checkpoint_revision: QWEN35_CHECKPOINT_REVISION.into(),
            key_id: "sage-test-2026".into(),
            artifacts,
            signature_hex: String::new(),
        };
        manifest.signature_hex = hex_signature(&key.sign(&manifest.signing_bytes().unwrap()));
        (manifest, key)
    }

    fn hex_signature(signature: &Signature) -> String {
        signature
            .to_bytes()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    #[test]
    fn signed_manifest_requires_pinned_profile_and_a_trusted_signer() {
        let (manifest, key) = fixture();
        let trusted = BTreeMap::from([(manifest.key_id.clone(), key.verifying_key())]);
        let verified = manifest
            .verify_signature(&trusted)
            .expect("valid signed Qwen package manifest");
        assert_eq!(verified.checkpoint_revision(), QWEN35_CHECKPOINT_REVISION);
        assert_eq!(verified.key_id(), "sage-test-2026");
        assert_eq!(verified.manifest_sha256().len(), 64);

        let unknown_trust = BTreeMap::new();
        assert!(manifest.verify_signature(&unknown_trust).is_err());
        let mut legacy_schema = manifest.clone();
        legacy_schema.schema_version = 1;
        assert!(legacy_schema.verify_signature(&trusted).is_err());
        let mut altered = manifest.clone();
        altered
            .artifacts
            .insert("config.json".into(), "c".repeat(64));
        assert!(altered.verify_signature(&trusted).is_err());
        let mut wrong_revision = manifest;
        wrong_revision.checkpoint_revision = "0".repeat(40);
        assert!(wrong_revision.verify_signature(&trusted).is_err());
    }

    #[test]
    fn verified_package_hashes_open_artifacts_in_bounded_chunks_and_restores_offset() {
        let (manifest, key) = fixture();
        let trusted = BTreeMap::from([(manifest.key_id.clone(), key.verifying_key())]);
        let bytes = b"small pinned model artifact".to_vec();
        let digest = format!("{:x}", Sha256::digest(&bytes));
        let mut altered = manifest.clone();
        altered
            .artifacts
            .insert("config.json".into(), digest.clone());
        altered.signature_hex = hex_signature(&key.sign(&altered.signing_bytes().unwrap()));
        let verified = altered
            .verify_signature(&trusted)
            .expect("signed artifact digest");
        let mut reader = Cursor::new(bytes);
        reader.set_position(3);
        let artifact = verified
            .verify_artifact("config.json", &mut reader)
            .expect("matching signed artifact");
        assert_eq!(artifact.sha256(), digest);
        assert_eq!(reader.position(), 3);
        assert!(
            verified
                .verify_artifact("unlisted.bin", &mut Cursor::new(b"unknown"))
                .is_err()
        );

        let mut tampered = Cursor::new(b"tampered artifact".to_vec());
        assert!(
            verified
                .verify_artifact("config.json", &mut tampered)
                .is_err()
        );
    }

    #[test]
    fn owned_verification_preserves_the_exact_hashed_reader_for_consumption() {
        let (manifest, key) = fixture();
        let trusted = BTreeMap::from([(manifest.key_id.clone(), key.verifying_key())]);
        let bytes = b"small pinned model artifact".to_vec();
        let digest = format!("{:x}", Sha256::digest(&bytes));
        let mut manifest = manifest;
        manifest
            .artifacts
            .insert("config.json".into(), digest.clone());
        manifest.signature_hex = hex_signature(&key.sign(&manifest.signing_bytes().unwrap()));
        let verified = manifest
            .verify_signature(&trusted)
            .expect("valid signed manifest");
        let artifact = verified
            .verify_owned_artifact("config.json", Cursor::new(bytes.clone()))
            .expect("signed owned artifact");
        assert_eq!(artifact.name(), "config.json");
        assert_eq!(artifact.sha256(), digest);
        assert_eq!(artifact.bytes(), bytes.len() as u64);
        let mut reader = artifact.into_verified_reader();
        let mut observed = Vec::new();
        reader.read_to_end(&mut observed).unwrap();
        assert_eq!(observed, bytes);
    }

    #[test]
    fn small_artifacts_are_hashed_and_retained_from_one_bounded_read() {
        let (manifest, key) = fixture();
        let trusted = BTreeMap::from([(manifest.key_id.clone(), key.verifying_key())]);
        let bytes = b"verified tokenizer bytes".to_vec();
        let digest = format!("{:x}", Sha256::digest(&bytes));
        let mut manifest = manifest;
        manifest
            .artifacts
            .insert("tokenizer.json".into(), digest.clone());
        manifest.signature_hex = hex_signature(&key.sign(&manifest.signing_bytes().unwrap()));
        let verified = manifest
            .verify_signature(&trusted)
            .expect("valid signed manifest");
        let mut source = Cursor::new(bytes.clone());
        source.set_position(4);
        let artifact = verified
            .read_verified_small_artifact("tokenizer.json", &mut source)
            .expect("matching small artifact");
        assert_eq!(artifact.sha256(), digest);
        assert_eq!(source.position(), 4);
        assert_eq!(artifact.into_verified_bytes(), bytes);

        let mut shard = Cursor::new(b"shard".to_vec());
        assert!(
            verified
                .read_verified_small_artifact(
                    "model.safetensors-00001-of-00002.safetensors",
                    &mut shard
                )
                .is_err()
        );
    }
}
