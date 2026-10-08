#![forbid(unsafe_code)]

//! Offline signer for Sage's exact pinned Qwen3.5 candidate artifacts.
//!
//! This tool creates a signed package manifest; it does not admit the model,
//! install a trust root, or enable product inference.

use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    path::{Path, PathBuf},
};

use anyhow::{Context, bail};
use ed25519_dalek::{Signer, SigningKey};
use sage_model_package::{
    QWEN35_CHECKPOINT_REVISION, QWEN35_PACKAGE_ARTIFACTS, Qwen35PackageManifest,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

const PINNED_CANDIDATE: &str = include_str!("../../../../evals/models/qwen3.5-4b-candidate.json");
const MANIFEST_FILE_NAME: &str = "sage-qwen35-package.json";
const HASH_BUFFER_BYTES: usize = 1024 * 1024;

#[derive(Debug, Deserialize)]
struct CandidateRecord {
    schema_version: u32,
    checkpoint: String,
    checkpoint_revision: String,
    source_artifacts: BTreeMap<String, ArtifactPin>,
}

#[derive(Debug, Deserialize)]
struct ArtifactPin {
    bytes: u64,
    sha256: String,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("sage-qwen35-package: {error:#}");
        std::process::exit(2);
    }
}

fn run() -> anyhow::Result<()> {
    let (artifact_dir, key_id) = parse_arguments()?;
    if !artifact_dir.is_absolute() {
        bail!("--artifact-dir must be an absolute path");
    }
    if !valid_key_id(&key_id) {
        bail!("--key-id must contain 1-128 ASCII letters, digits, '.', '_' or '-'");
    }

    let seed = read_signing_seed(io::stdin().lock())?;
    let signing_key = SigningKey::from_bytes(&seed);
    let candidate = pinned_candidate()?;
    let manifest = sign_candidate(&artifact_dir, &key_id, &signing_key, &candidate)
        .context("could not sign the pinned candidate package")?;
    let bytes = serde_json::to_vec_pretty(&manifest)?;
    let output_path = artifact_dir.join(MANIFEST_FILE_NAME);
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output_path)
        .with_context(|| {
            format!(
                "could not create {} without replacing an existing file",
                output_path.display()
            )
        })?;
    if let Err(error) = output.write_all(&bytes).and_then(|()| output.sync_all()) {
        drop(output);
        let _ = fs::remove_file(&output_path);
        return Err(error).with_context(|| format!("could not persist {}", output_path.display()));
    }
    eprintln!(
        "Signed candidate manifest written to {}",
        output_path.display()
    );
    Ok(())
}

fn parse_arguments() -> anyhow::Result<(PathBuf, String)> {
    let mut artifact_dir = None;
    let mut key_id = None;
    let mut signing_seed_stdin = false;
    let mut arguments = std::env::args_os().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--artifact-dir") => {
                if artifact_dir.is_some() {
                    bail!("--artifact-dir may be supplied only once");
                }
                let value = arguments.next().context("--artifact-dir requires a path")?;
                artifact_dir = Some(PathBuf::from(value));
            }
            Some("--key-id") => {
                if key_id.is_some() {
                    bail!("--key-id may be supplied only once");
                }
                let value = arguments.next().context("--key-id requires a value")?;
                key_id = Some(
                    value
                        .into_string()
                        .map_err(|_| anyhow::anyhow!("--key-id must be valid UTF-8"))?,
                );
            }
            Some("--signing-seed-stdin") => {
                if signing_seed_stdin {
                    bail!("--signing-seed-stdin may be supplied only once");
                }
                signing_seed_stdin = true;
            }
            Some("--help") | Some("-h") => {
                print_help();
                std::process::exit(0);
            }
            _ => bail!("unknown argument; use --help for usage"),
        }
    }
    let artifact_dir = artifact_dir.context("--artifact-dir is required")?;
    let key_id = key_id.context("--key-id is required")?;
    if !signing_seed_stdin {
        bail!("--signing-seed-stdin is required to make key handling explicit");
    }
    Ok((artifact_dir, key_id))
}

fn print_help() {
    println!(
        "Usage: sage-qwen35-package --artifact-dir ABSOLUTE_PATH --key-id ID --signing-seed-stdin\n\
         Reads exactly 32 raw Ed25519 seed bytes from stdin. Hashes and checks all six\n\
         artifacts against Sage's pinned candidate record, then creates\n\
         sage-qwen35-package.json without replacing an existing file. This signs a\n\
         candidate package only; it does not admit or enable the model."
    );
}

fn valid_key_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
}

fn read_signing_seed<R: Read>(mut reader: R) -> anyhow::Result<Zeroizing<[u8; 32]>> {
    let mut seed = Zeroizing::new([0_u8; 32]);
    reader
        .read_exact(&mut seed[..])
        .context("stdin must contain exactly 32 raw Ed25519 seed bytes")?;
    let mut extra = [0_u8; 1];
    if reader.read(&mut extra)? != 0 {
        bail!("stdin contains more than 32 Ed25519 seed bytes");
    }
    Ok(seed)
}

fn pinned_candidate() -> anyhow::Result<CandidateRecord> {
    let candidate: CandidateRecord = serde_json::from_str(PINNED_CANDIDATE)
        .context("the embedded Sage candidate record is invalid")?;
    if candidate.schema_version != 2
        || candidate.checkpoint != "Qwen/Qwen3.5-4B"
        || candidate.checkpoint_revision != QWEN35_CHECKPOINT_REVISION
        || candidate.source_artifacts.len() != QWEN35_PACKAGE_ARTIFACTS.len()
        || candidate
            .source_artifacts
            .keys()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>()
            != QWEN35_PACKAGE_ARTIFACTS.into_iter().collect()
        || candidate
            .source_artifacts
            .values()
            .any(|artifact| artifact.bytes == 0 || !valid_digest(&artifact.sha256))
    {
        bail!("embedded candidate record does not match Sage's pinned closed profile");
    }
    Ok(candidate)
}

fn sign_candidate(
    artifact_dir: &Path,
    key_id: &str,
    signing_key: &SigningKey,
    candidate: &CandidateRecord,
) -> anyhow::Result<Qwen35PackageManifest> {
    let mut artifacts = BTreeMap::new();
    for name in QWEN35_PACKAGE_ARTIFACTS {
        let pin = candidate
            .source_artifacts
            .get(name)
            .with_context(|| format!("pinned candidate is missing {name}"))?;
        let path = artifact_dir.join(name);
        let (sha256, bytes) = hash_pinned_artifact(&path, pin.bytes)
            .with_context(|| format!("candidate artifact {name} failed verification"))?;
        if bytes != pin.bytes || sha256 != pin.sha256 {
            bail!("candidate artifact {name} differs from Sage's pinned source digest");
        }
        artifacts.insert(name.to_owned(), sha256);
    }

    let mut manifest = Qwen35PackageManifest {
        schema_version: 2,
        checkpoint_revision: candidate.checkpoint_revision.clone(),
        key_id: key_id.to_owned(),
        artifacts,
        signature_hex: String::new(),
    };
    let signature = signing_key.sign(&manifest.signing_bytes()?);
    manifest.signature_hex = encode_hex(&signature.to_bytes());

    let trusted_keys = BTreeMap::from([(key_id.to_owned(), signing_key.verifying_key())]);
    manifest.verify_signature(&trusted_keys)?;
    Ok(manifest)
}

fn hash_pinned_artifact(path: &Path, expected_bytes: u64) -> anyhow::Result<(String, u64)> {
    let path_metadata = fs::symlink_metadata(path)
        .with_context(|| format!("could not inspect {}", path.display()))?;
    if !path_metadata.file_type().is_file() || path_metadata.len() != expected_bytes {
        bail!("artifact must be a regular file with the pinned exact size");
    }

    let mut file =
        File::open(path).with_context(|| format!("could not open {}", path.display()))?;
    let initial_metadata = file.metadata()?;
    if !initial_metadata.is_file() || initial_metadata.len() != expected_bytes {
        bail!("opened artifact does not have the pinned exact size");
    }

    let mut digest = Sha256::new();
    let mut buffer = [0_u8; HASH_BUFFER_BYTES];
    let mut bytes = 0_u64;
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        bytes = bytes
            .checked_add(read as u64)
            .context("artifact byte count overflow")?;
        if bytes > expected_bytes {
            bail!("artifact grew beyond its pinned size while hashing");
        }
        digest.update(&buffer[..read]);
    }
    if bytes != expected_bytes || file.metadata()?.len() != expected_bytes {
        bail!("artifact changed size while hashing");
    }
    Ok((format!("{:x}", digest.finalize()), bytes))
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, fs, io::Cursor};

    use ed25519_dalek::SigningKey;
    use sage_model_package::QWEN35_PACKAGE_ARTIFACTS;
    use sha2::{Digest, Sha256};

    use super::{
        ArtifactPin, CandidateRecord, hash_pinned_artifact, read_signing_seed, sign_candidate,
    };

    fn fixture_candidate(directory: &std::path::Path) -> CandidateRecord {
        let mut source_artifacts = BTreeMap::new();
        for name in QWEN35_PACKAGE_ARTIFACTS {
            let content = format!("fixture:{name}").into_bytes();
            fs::write(directory.join(name), &content).unwrap();
            source_artifacts.insert(
                name.to_owned(),
                ArtifactPin {
                    bytes: content.len() as u64,
                    sha256: format!("{:x}", Sha256::digest(&content)),
                },
            );
        }
        CandidateRecord {
            schema_version: 2,
            checkpoint: "Qwen/Qwen3.5-4B".into(),
            checkpoint_revision: sage_model_package::QWEN35_CHECKPOINT_REVISION.into(),
            source_artifacts,
        }
    }

    #[test]
    fn pinned_candidate_record_names_the_closed_package_artifacts() {
        let candidate = super::pinned_candidate().unwrap();
        assert_eq!(
            candidate.source_artifacts.len(),
            QWEN35_PACKAGE_ARTIFACTS.len()
        );
        for name in QWEN35_PACKAGE_ARTIFACTS {
            assert!(candidate.source_artifacts.contains_key(name));
        }
    }

    #[test]
    fn package_signing_binds_every_verified_artifact_to_the_key() {
        let directory = tempfile::tempdir().unwrap();
        let candidate = fixture_candidate(directory.path());
        let signing_key = SigningKey::from_bytes(&[7_u8; 32]);
        let manifest =
            sign_candidate(directory.path(), "test-key-1", &signing_key, &candidate).unwrap();
        let trusted = BTreeMap::from([("test-key-1".into(), signing_key.verifying_key())]);
        let verified = manifest.verify_signature(&trusted).unwrap();
        assert_eq!(
            verified.checkpoint_revision(),
            super::QWEN35_CHECKPOINT_REVISION
        );
        assert_eq!(manifest.artifacts.len(), QWEN35_PACKAGE_ARTIFACTS.len());

        let other_key = SigningKey::from_bytes(&[8_u8; 32]);
        let wrong_trust = BTreeMap::from([("test-key-1".into(), other_key.verifying_key())]);
        assert!(manifest.verify_signature(&wrong_trust).is_err());
    }

    #[test]
    fn package_signer_rejects_changed_or_symbolic_artifacts() {
        let directory = tempfile::tempdir().unwrap();
        let candidate = fixture_candidate(directory.path());
        let first = QWEN35_PACKAGE_ARTIFACTS[0];
        fs::write(directory.path().join(first), b"changed bytes").unwrap();
        let signing_key = SigningKey::from_bytes(&[9_u8; 32]);
        assert!(sign_candidate(directory.path(), "test-key", &signing_key, &candidate).is_err());

        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            let target = directory.path().join("real-artifact");
            let alias = directory.path().join("alias-artifact");
            fs::write(&target, b"a").unwrap();
            symlink(&target, &alias).unwrap();
            assert!(hash_pinned_artifact(&alias, 1).is_err());
        }
    }

    #[test]
    fn signing_seed_reader_requires_exactly_32_bytes() {
        let valid = read_signing_seed(Cursor::new([3_u8; 32])).unwrap();
        assert_eq!(&valid[..], &[3_u8; 32]);
        assert!(read_signing_seed(Cursor::new([3_u8; 31])).is_err());
        assert!(read_signing_seed(Cursor::new([3_u8; 33])).is_err());
    }

    #[test]
    fn digest_reader_rejects_wrong_size_before_reading() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("small");
        fs::write(&path, b"four").unwrap();
        assert!(hash_pinned_artifact(&path, 3).is_err());
    }
}
