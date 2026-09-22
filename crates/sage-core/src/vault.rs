//! SQLCipher storage and staged, verified migration. SQL strings containing key
//! material are never logged or included in error messages.
use crate::{CoreError, CoreResult, secrets::SecretBytes};
use rusqlite::{Connection, params};
use std::{io::Read, path::Path};

fn raw_key(key: &SecretBytes) -> zeroize::Zeroizing<String> {
    zeroize::Zeroizing::new(format!(
        "x'{}'",
        key.expose()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    ))
}

pub fn is_plaintext(path: &Path) -> CoreResult<bool> {
    let mut header = [0; 16];
    let count = std::fs::File::open(path)?.read(&mut header)?;
    Ok(count == 16 && &header == b"SQLite format 3\0")
}

pub fn open_encrypted(path: &Path, key: &SecretBytes) -> CoreResult<Connection> {
    let connection = Connection::open(path)?;
    let key = raw_key(key);
    connection
        .pragma_update(None, "key", &*key)
        .map_err(|_| failure("Database key could not be applied"))?;
    let cipher: String = connection
        .query_row("PRAGMA cipher_version", [], |row| row.get(0))
        .map_err(|_| failure("SQLCipher is required; plaintext fallback is disabled"))?;
    if cipher.is_empty() {
        return Err(failure("SQLCipher is unavailable"));
    }
    connection
        .query_row("SELECT count(*) FROM sqlite_master", [], |row| {
            row.get::<_, u64>(0)
        })
        .map_err(|_| failure("Database could not be decrypted"))?;
    connection.execute_batch("PRAGMA temp_store=MEMORY; PRAGMA cipher_memory_security=ON; PRAGMA foreign_keys=ON; PRAGMA busy_timeout=5000; PRAGMA journal_mode=WAL;")?;
    Ok(connection)
}

pub fn migrate_plaintext(path: &Path, key: &SecretBytes) -> CoreResult<()> {
    if !path.exists() || !is_plaintext(path)? {
        return Ok(());
    }
    let parent = path
        .parent()
        .ok_or_else(|| failure("Database has no parent"))?;
    let staged = parent.join(format!(".sage-encrypted-{}.db", uuid::Uuid::new_v4()));
    let backup = parent.join(format!(
        "sage-before-v2-{}.encrypted.db",
        uuid::Uuid::new_v4()
    ));
    let result = (|| {
        let source = Connection::open(path)?;
        source.execute_batch(
            "PRAGMA temp_store=MEMORY; PRAGMA busy_timeout=5000; PRAGMA wal_checkpoint(TRUNCATE);",
        )?;
        source
            .execute(
                "ATTACH DATABASE ?1 AS encrypted KEY ?2",
                params![staged.to_string_lossy(), &*raw_key(key)],
            )
            .map_err(|_| failure("Could not create encrypted migration destination"))?;
        source.execute_batch("BEGIN IMMEDIATE;")?;
        source
            .query_row("SELECT sqlcipher_export('encrypted')", [], |_| Ok(()))
            .map_err(|_| failure("Encrypted migration export failed"))?;
        source.execute_batch("COMMIT;")?;
        let integrity: String =
            source.query_row("PRAGMA encrypted.integrity_check", [], |row| row.get(0))?;
        if integrity != "ok" {
            return Err(failure("Encrypted migration integrity check failed"));
        }
        source.execute_batch("DETACH DATABASE encrypted;")?;
        drop(source);
        let check = open_encrypted(&staged, key)?;
        let faults = check
            .prepare("PRAGMA cipher_integrity_check")?
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        if !faults.is_empty() {
            return Err(failure("Encrypted page authentication failed"));
        }
        check.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")?;
        drop(check);
        // The rollback copy is encrypted before the original is replaced.
        std::fs::copy(&staged, &backup)?;
        std::fs::File::open(&backup)?.sync_all()?;
        std::fs::File::open(&staged)?.sync_all()?;
        std::fs::rename(&staged, path)?;
        #[cfg(unix)]
        std::fs::File::open(parent)?.sync_all()?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&staged);
    }
    result
}

fn failure(message: &str) -> CoreError {
    CoreError::Storage(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn migration_encrypts_history_and_rollback_before_replacement() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("legacy.db");
        let plain = Connection::open(&path).unwrap();
        plain
            .execute_batch(
                "CREATE TABLE note(content TEXT); INSERT INTO note VALUES('private canary text');",
            )
            .unwrap();
        drop(plain);
        let key = SecretBytes::new(vec![19; 32]);
        migrate_plaintext(&path, &key).unwrap();
        assert!(!is_plaintext(&path).unwrap());
        let database = open_encrypted(&path, &key).unwrap();
        assert_eq!(
            database
                .query_row("SELECT content FROM note", [], |r| r.get::<_, String>(0))
                .unwrap(),
            "private canary text"
        );
        assert!(open_encrypted(&path, &SecretBytes::new(vec![20; 32])).is_err());
        for entry in std::fs::read_dir(temp.path()).unwrap() {
            let bytes = std::fs::read(entry.unwrap().path()).unwrap();
            assert!(
                !bytes
                    .windows(b"private canary text".len())
                    .any(|part| part == b"private canary text")
            );
        }
    }
}
