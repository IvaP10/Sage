//! Durable, single-writer fencing for peer partition dispatch.
//!
//! The journal records only random job/lease identifiers, a paired-device
//! identity, and the plan digest. It never stores input or output bytes. A
//! broker must commit this record before exposing an authorized request to a
//! worker. A torn, reordered, or modified journal fails closed at startup.
//! The keyed chain detects modification, but cannot detect restoration of an
//! older valid file image; rollback protection requires a platform monotonic
//! store and remains a host integration responsibility.

use std::{
    collections::BTreeSet,
    fmt,
    fs::{File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
};

use hmac::Mac;
use sha2::Sha256;
use subtle::ConstantTimeEq;
use zeroize::Zeroizing;

use crate::{AuthorizedPartitionRequest, PeerError, PeerResult};

type HmacSha256 = hmac::Hmac<Sha256>;

const FILE_MAGIC: &[u8; 4] = b"SGDJ";
const FILE_VERSION: u16 = 1;
const HEADER_BYTES: u64 = 6;
const JOURNAL_DOMAIN: &[u8] = b"sage:peer:dispatch-journal:v1\0";
const RECORD_BODY_BYTES: usize = 138;
const RECORD_BYTES: usize = RECORD_BODY_BYTES + 32;
const MAX_DISPATCHES: usize = 1_000_000;

/// A keyed, durable set of job partitions whose dispatch outcome may now be
/// uncertain. Its exclusive file lock must be held by the broker for the
/// whole session. Supply an integrity key obtained from the native OS key
/// store; this journal does not persist or derive that key.
pub struct PeerDispatchJournal {
    file: File,
    integrity_key: Zeroizing<[u8; 32]>,
    last_mac: [u8; 32],
    sequence: u64,
    dispatched: BTreeSet<([u8; 16], u16)>,
    poisoned: bool,
}

impl PeerDispatchJournal {
    /// Open or create a journal and verify its complete authenticated history.
    /// Only one broker process can own the journal at a time.
    pub fn open(path: impl AsRef<Path>, integrity_key: Zeroizing<[u8; 32]>) -> PeerResult<Self> {
        if integrity_key.iter().all(|byte| *byte == 0) {
            return Err(PeerError::DispatchJournalKeyInvalid);
        }
        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(path)
            .map_err(|_| PeerError::DispatchJournalUnavailable)?;
        file.try_lock()
            .map_err(|_| PeerError::DispatchJournalInUse)?;

        let mut file_bytes = file
            .metadata()
            .map_err(|_| PeerError::DispatchJournalUnavailable)?
            .len();
        if file_bytes == 0 {
            file.write_all(&file_header())
                .and_then(|()| file.sync_all())
                .map_err(|_| PeerError::DispatchJournalUnavailable)?;
            file_bytes = HEADER_BYTES;
        }

        let mut journal = Self {
            file,
            integrity_key,
            last_mac: [0; 32],
            sequence: 0,
            dispatched: BTreeSet::new(),
            poisoned: false,
        };
        journal.recover(file_bytes)?;
        Ok(journal)
    }

    /// Persist a fence for this already-authorized exact partition. The caller
    /// may expose the request to its worker only after this method succeeds.
    pub fn record_dispatch(&mut self, request: &AuthorizedPartitionRequest) -> PeerResult<()> {
        self.append_record(
            *request.job_id(),
            request.partition_index(),
            *request.lease_id().as_bytes(),
            *request.peer().as_bytes(),
            *request.plan_digest(),
        )
    }

    fn append_record(
        &mut self,
        job_id: [u8; 16],
        partition_index: u16,
        lease_id: [u8; 16],
        peer_id: [u8; 32],
        plan_digest: [u8; 32],
    ) -> PeerResult<()> {
        if self.poisoned {
            return Err(PeerError::DispatchJournalUnavailable);
        }
        if self.dispatched.contains(&(job_id, partition_index)) {
            return Err(PeerError::DuplicateJob);
        }
        if self.dispatched.len() >= MAX_DISPATCHES {
            return Err(PeerError::DispatchJournalFull);
        }

        let next_sequence = self
            .sequence
            .checked_add(1)
            .ok_or(PeerError::DispatchJournalFull)?;
        let mut record = [0_u8; RECORD_BYTES];
        let mut offset = 0;
        put(&mut record, &mut offset, &next_sequence.to_be_bytes());
        put(&mut record, &mut offset, &job_id);
        put(&mut record, &mut offset, &partition_index.to_be_bytes());
        put(&mut record, &mut offset, &lease_id);
        put(&mut record, &mut offset, &peer_id);
        put(&mut record, &mut offset, &plan_digest);
        put(&mut record, &mut offset, &self.last_mac);
        debug_assert_eq!(offset, RECORD_BODY_BYTES);
        let mac = record_mac(&self.integrity_key, &record[..RECORD_BODY_BYTES])?;
        put(&mut record, &mut offset, &mac);
        debug_assert_eq!(offset, RECORD_BYTES);

        if self
            .file
            .write_all(&record)
            .and_then(|()| self.file.sync_data())
            .is_err()
        {
            // A failed append may have left a torn tail. Do not permit further
            // dispatches until a fresh open has validated the on-disk state.
            self.poisoned = true;
            return Err(PeerError::DispatchJournalUnavailable);
        }

        self.dispatched.insert((job_id, partition_index));
        self.sequence = next_sequence;
        self.last_mac = mac;
        Ok(())
    }

    /// Whether a partition identity has crossed the durable dispatch fence.
    pub fn contains(&self, job_id: &[u8; 16], partition_index: u16) -> bool {
        self.dispatched.contains(&(*job_id, partition_index))
    }

    pub fn len(&self) -> usize {
        self.dispatched.len()
    }

    pub fn is_empty(&self) -> bool {
        self.dispatched.is_empty()
    }

    fn recover(&mut self, file_bytes: u64) -> PeerResult<()> {
        if file_bytes < HEADER_BYTES {
            return Err(PeerError::DispatchJournalCorrupt);
        }
        self.file
            .seek(SeekFrom::Start(0))
            .map_err(|_| PeerError::DispatchJournalUnavailable)?;
        let mut header = [0_u8; HEADER_BYTES as usize];
        self.file
            .read_exact(&mut header)
            .map_err(|_| PeerError::DispatchJournalCorrupt)?;
        if header != file_header() {
            return Err(PeerError::DispatchJournalCorrupt);
        }

        let payload_bytes = file_bytes - HEADER_BYTES;
        if !payload_bytes.is_multiple_of(RECORD_BYTES as u64) {
            return Err(PeerError::DispatchJournalCorrupt);
        }
        let record_count = usize::try_from(payload_bytes / RECORD_BYTES as u64)
            .ok()
            .filter(|count| *count <= MAX_DISPATCHES)
            .ok_or(PeerError::DispatchJournalCorrupt)?;

        let mut record = [0_u8; RECORD_BYTES];
        for expected_sequence in 1..=record_count {
            self.file
                .read_exact(&mut record)
                .map_err(|_| PeerError::DispatchJournalCorrupt)?;
            let mut offset = 0;
            let sequence = take_u64(&record, &mut offset)?;
            let job_id = take_array::<16>(&record, &mut offset)?;
            let partition_index = take_u16(&record, &mut offset)?;
            let _lease_id = take_array::<16>(&record, &mut offset)?;
            let _peer_id = take_array::<32>(&record, &mut offset)?;
            let _plan_digest = take_array::<32>(&record, &mut offset)?;
            let previous_mac = take_array::<32>(&record, &mut offset)?;
            let stored_mac = take_array::<32>(&record, &mut offset)?;
            if offset != RECORD_BYTES
                || sequence != expected_sequence as u64
                || previous_mac != self.last_mac
            {
                return Err(PeerError::DispatchJournalCorrupt);
            }
            let actual_mac = record_mac(&self.integrity_key, &record[..RECORD_BODY_BYTES])?;
            if !bool::from(actual_mac.ct_eq(&stored_mac))
                || !self.dispatched.insert((job_id, partition_index))
            {
                return Err(PeerError::DispatchJournalCorrupt);
            }
            self.sequence = sequence;
            self.last_mac = stored_mac;
        }

        // Keep append writes explicit even though the file was opened in
        // append mode, and make accidental cursor assumptions impossible.
        self.file
            .seek(SeekFrom::End(0))
            .map_err(|_| PeerError::DispatchJournalUnavailable)?;
        Ok(())
    }
}

impl fmt::Debug for PeerDispatchJournal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PeerDispatchJournal")
            .field("sequence", &self.sequence)
            .field("dispatched_partitions", &self.dispatched.len())
            .field("integrity_key", &"[redacted]")
            .field("poisoned", &self.poisoned)
            .finish()
    }
}

fn file_header() -> [u8; HEADER_BYTES as usize] {
    let mut header = [0; HEADER_BYTES as usize];
    header[..FILE_MAGIC.len()].copy_from_slice(FILE_MAGIC);
    header[FILE_MAGIC.len()..].copy_from_slice(&FILE_VERSION.to_be_bytes());
    header
}

fn record_mac(key: &[u8; 32], record: &[u8]) -> PeerResult<[u8; 32]> {
    let mut mac = <HmacSha256 as Mac>::new_from_slice(key)
        .map_err(|_| PeerError::DispatchJournalUnavailable)?;
    mac.update(JOURNAL_DOMAIN);
    mac.update(record);
    Ok(mac.finalize().into_bytes().into())
}

fn put<const N: usize>(target: &mut [u8], offset: &mut usize, value: &[u8; N]) {
    target[*offset..*offset + N].copy_from_slice(value);
    *offset += N;
}

fn take<const N: usize>(source: &[u8], offset: &mut usize) -> PeerResult<[u8; N]> {
    let end = offset
        .checked_add(N)
        .ok_or(PeerError::DispatchJournalCorrupt)?;
    let value = source
        .get(*offset..end)
        .ok_or(PeerError::DispatchJournalCorrupt)?
        .try_into()
        .map_err(|_| PeerError::DispatchJournalCorrupt)?;
    *offset = end;
    Ok(value)
}

fn take_u16(source: &[u8], offset: &mut usize) -> PeerResult<u16> {
    Ok(u16::from_be_bytes(take(source, offset)?))
}

fn take_u64(source: &[u8], offset: &mut usize) -> PeerResult<u64> {
    Ok(u64::from_be_bytes(take(source, offset)?))
}

fn take_array<const N: usize>(source: &[u8], offset: &mut usize) -> PeerResult<[u8; N]> {
    take(source, offset)
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
    };

    use super::*;

    static NEXT_PATH: AtomicU64 = AtomicU64::new(0);

    struct ScratchFile(PathBuf);

    impl ScratchFile {
        fn new() -> Self {
            let id = NEXT_PATH.fetch_add(1, Ordering::Relaxed);
            Self(std::env::temp_dir().join(format!(
                "sage-peer-dispatch-journal-{}-{id}.bin",
                std::process::id()
            )))
        }
    }

    impl Drop for ScratchFile {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.0);
        }
    }

    fn key() -> Zeroizing<[u8; 32]> {
        Zeroizing::new([0x5a; 32])
    }

    #[test]
    fn reopen_recovers_fences_and_blocks_reuse_under_another_lease() {
        let path = ScratchFile::new();
        let job_id = [1; 16];
        {
            let mut journal = PeerDispatchJournal::open(&path.0, key()).unwrap();
            assert!(journal.is_empty());
            journal
                .append_record(job_id, 3, [2; 16], [3; 32], [4; 32])
                .unwrap();
            assert!(journal.contains(&job_id, 3));
        }
        let mut reopened = PeerDispatchJournal::open(&path.0, key()).unwrap();
        assert_eq!(reopened.len(), 1);
        assert!(reopened.contains(&job_id, 3));
        assert_eq!(
            reopened.append_record(job_id, 3, [8; 16], [9; 32], [7; 32]),
            Err(PeerError::DuplicateJob)
        );
    }

    #[test]
    fn wrong_integrity_key_and_modified_record_fail_closed() {
        let path = ScratchFile::new();
        {
            let mut journal = PeerDispatchJournal::open(&path.0, key()).unwrap();
            journal
                .append_record([1; 16], 0, [2; 16], [3; 32], [4; 32])
                .unwrap();
        }
        assert_eq!(
            PeerDispatchJournal::open(&path.0, Zeroizing::new([0x6b; 32])).err(),
            Some(PeerError::DispatchJournalCorrupt)
        );
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path.0)
            .unwrap();
        file.seek(SeekFrom::Start(HEADER_BYTES + 9)).unwrap();
        let mut byte = [0];
        file.read_exact(&mut byte).unwrap();
        file.seek(SeekFrom::Current(-1)).unwrap();
        file.write_all(&[byte[0] ^ 0x80]).unwrap();
        file.sync_all().unwrap();
        drop(file);
        assert_eq!(
            PeerDispatchJournal::open(&path.0, key()).err(),
            Some(PeerError::DispatchJournalCorrupt)
        );
    }

    #[test]
    fn all_zero_integrity_key_is_rejected_before_creating_a_file() {
        let path = ScratchFile::new();
        assert_eq!(
            PeerDispatchJournal::open(&path.0, Zeroizing::new([0; 32])).err(),
            Some(PeerError::DispatchJournalKeyInvalid)
        );
        assert!(!path.0.exists());
    }

    #[test]
    fn partial_header_or_torn_record_is_never_silently_truncated() {
        let path = ScratchFile::new();
        fs::write(&path.0, &file_header()[..3]).unwrap();
        assert_eq!(
            PeerDispatchJournal::open(&path.0, key()).err(),
            Some(PeerError::DispatchJournalCorrupt)
        );

        fs::write(&path.0, file_header()).unwrap();
        {
            let mut journal = PeerDispatchJournal::open(&path.0, key()).unwrap();
            journal
                .append_record([1; 16], 0, [2; 16], [3; 32], [4; 32])
                .unwrap();
        }
        let file = OpenOptions::new().write(true).open(&path.0).unwrap();
        file.set_len(HEADER_BYTES + RECORD_BYTES as u64 - 1)
            .unwrap();
        file.sync_all().unwrap();
        assert_eq!(
            PeerDispatchJournal::open(&path.0, key()).err(),
            Some(PeerError::DispatchJournalCorrupt)
        );
    }

    #[test]
    fn a_second_broker_cannot_open_the_same_journal() {
        let path = ScratchFile::new();
        let _first = PeerDispatchJournal::open(&path.0, key()).unwrap();
        assert_eq!(
            PeerDispatchJournal::open(&path.0, key()).err(),
            Some(PeerError::DispatchJournalInUse)
        );
    }
}
