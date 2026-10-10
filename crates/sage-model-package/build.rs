use std::{collections::BTreeSet, env, fs, path::PathBuf};

const TRUST_ENV: &str = "SAGE_QWEN35_PACKAGE_TRUSTED_KEYS";
const MAX_TRUST_ENV_BYTES: usize = 4096;
const MAX_TRUSTED_KEYS: usize = 16;

fn main() {
    println!("cargo:rerun-if-env-changed={TRUST_ENV}");

    let encoded = match env::var(TRUST_ENV) {
        Ok(encoded) => encoded,
        Err(env::VarError::NotPresent) => String::new(),
        Err(env::VarError::NotUnicode(_)) => {
            panic!("{TRUST_ENV} must be valid UTF-8")
        }
    };
    assert!(
        encoded.len() <= MAX_TRUST_ENV_BYTES,
        "{TRUST_ENV} exceeds its 4096-byte build-time bound"
    );

    let mut keys = Vec::new();
    let mut ids = BTreeSet::new();
    let mut public_keys = BTreeSet::new();
    for line in encoded
        .lines()
        .map(|line| line.strip_suffix('\r').unwrap_or(line))
        .filter(|line| !line.is_empty())
    {
        let (key_id, encoded_key) = line
            .split_once('=')
            .expect("trusted key entries must use key-id=64-lowercase-hex format");
        assert!(
            !key_id.is_empty()
                && key_id.len() <= 128
                && key_id.bytes().all(|byte| {
                    byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-')
                }),
            "{TRUST_ENV} contains an invalid key ID"
        );
        assert_eq!(
            encoded_key.len(),
            64,
            "{TRUST_ENV} public keys must be exactly 32 bytes encoded as lowercase hex"
        );
        assert!(
            encoded_key
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
            "{TRUST_ENV} public keys must use lowercase hexadecimal"
        );
        assert!(ids.insert(key_id.to_owned()), "duplicate trusted key ID");

        let decoded = decode_hex(encoded_key);
        assert!(
            public_keys.insert(decoded.clone()),
            "duplicate trusted public key"
        );
        let bytes = decoded
            .iter()
            .map(|byte| format!("0x{byte:02x}"))
            .collect::<Vec<_>>()
            .join(", ");
        keys.push((key_id.to_owned(), bytes));
        assert!(
            keys.len() <= MAX_TRUSTED_KEYS,
            "{TRUST_ENV} has more than {MAX_TRUSTED_KEYS} trusted keys"
        );
    }
    keys.sort_unstable_by(|left, right| left.0.cmp(&right.0));

    let generated = keys
        .iter()
        .map(|(key_id, bytes)| format!("    ({key_id:?}, [{bytes}]),\n"))
        .collect::<String>();
    let source = format!(
        "// Generated from the build-time Sage package trust configuration.\n\
         pub(super) const TRUSTED_QWEN35_PACKAGE_KEYS: &[(&str, [u8; 32])] = &[\n{generated}];\n"
    );
    let output = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo provides OUT_DIR"))
        .join("trusted_qwen35_package_keys.rs");
    fs::write(output, source).expect("write generated model package trust set");
}

fn decode_hex(encoded: &str) -> Vec<u8> {
    encoded
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| {
            let high = if pair[0].is_ascii_digit() {
                pair[0] - b'0'
            } else {
                pair[0] - b'a' + 10
            };
            let low = if pair[1].is_ascii_digit() {
                pair[1] - b'0'
            } else {
                pair[1] - b'a' + 10
            };
            high * 16 + low
        })
        .collect()
}
