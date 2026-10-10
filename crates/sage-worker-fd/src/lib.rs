//! Bounded handoff of read-only regular files to a directly spawned worker.
//!
//! Each source is duplicated with close-on-exec set. The caller explicitly
//! attaches only those duplicates to one child command; the parent copies keep
//! their close-on-exec flag and remain read-only. This is a narrow native OS
//! bridge for model artifacts, not a general descriptor inheritance API.

use std::fs::File;
use std::io;

pub const MAX_INHERITED_FILES: usize = 8;
pub const MAX_INHERITED_FILE_BYTES: u64 = 16 * 1024 * 1024 * 1024;

/// Parent-owned, close-on-exec duplicates for a bounded child handoff.
#[derive(Debug, Default)]
pub struct InheritedReadOnlyFiles {
    files: Vec<File>,
}

impl InheritedReadOnlyFiles {
    /// Duplicates exact open handles after checking they are nonempty regular
    /// files opened read-only. The input handles remain unchanged.
    #[cfg(unix)]
    pub fn duplicate(sources: &[&File]) -> io::Result<Self> {
        use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

        if sources.len() > MAX_INHERITED_FILES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "too many read-only worker files",
            ));
        }

        let mut files = Vec::with_capacity(sources.len());
        for (index, source) in sources.iter().enumerate() {
            let descriptor = source.as_raw_fd();
            if sources[..index]
                .iter()
                .any(|previous| previous.as_raw_fd() == descriptor)
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "duplicate source descriptor",
                ));
            }
            if !source.metadata()?.is_file() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "worker artifact handle is not a regular file",
                ));
            }
            let length = source.metadata()?.len();
            if length == 0 || length > MAX_INHERITED_FILE_BYTES {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "worker artifact size is outside its bound",
                ));
            }

            // SAFETY: fcntl only inspects flags on this live borrowed descriptor.
            let status_flags = unsafe { libc::fcntl(descriptor, libc::F_GETFL) };
            if status_flags < 0 {
                return Err(io::Error::last_os_error());
            }
            if status_flags & libc::O_ACCMODE != libc::O_RDONLY {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "worker artifact handle is not read-only",
                ));
            }

            // Refuse source descriptors that could leak into an unrelated exec.
            // SAFETY: fcntl only inspects flags on this live borrowed descriptor.
            let descriptor_flags = unsafe { libc::fcntl(descriptor, libc::F_GETFD) };
            if descriptor_flags < 0 {
                return Err(io::Error::last_os_error());
            }
            if descriptor_flags & libc::FD_CLOEXEC == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "source descriptor is already inheritable",
                ));
            }

            // SAFETY: F_DUPFD_CLOEXEC returns a new descriptor owned by this
            // process, or -1. The original remains owned by the caller.
            let duplicate =
                unsafe { libc::fcntl(descriptor, libc::F_DUPFD_CLOEXEC, libc::STDERR_FILENO + 1) };
            if duplicate < 0 {
                return Err(io::Error::last_os_error());
            }

            // SAFETY: `duplicate` is the fresh descriptor just returned by
            // F_DUPFD_CLOEXEC and is transferred exactly once into OwnedFd.
            let owned = unsafe { OwnedFd::from_raw_fd(duplicate) };
            files.push(File::from(owned));
        }
        Ok(Self { files })
    }

    /// Non-Unix platforms currently fail closed; their native handle
    /// inheritance implementation is a separate platform acceptance gate.
    #[cfg(not(unix))]
    pub fn duplicate(sources: &[&File]) -> io::Result<Self> {
        if sources.is_empty() {
            Ok(Self::default())
        } else {
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "read-only worker handle inheritance is not implemented on this platform",
            ))
        }
    }

    /// Child-local descriptor numbers in stable source order.
    #[cfg(unix)]
    pub fn raw_descriptors(&self) -> impl ExactSizeIterator<Item = i32> + '_ {
        use std::os::fd::AsRawFd;

        self.files.iter().map(AsRawFd::as_raw_fd)
    }

    /// Attaches only these duplicates to the next exec of this command. Keep
    /// `self` alive until `spawn` returns. Other close-on-exec descriptors are
    /// left untouched.
    #[cfg(unix)]
    pub fn configure_child(&self, command: &mut std::process::Command) -> io::Result<()> {
        use std::os::unix::process::CommandExt;

        let descriptors = self.raw_descriptors().collect::<Vec<_>>();
        // SAFETY: the pre-exec closure calls only async-signal-safe fcntl on
        // already-open descriptors and does not allocate or access shared state.
        unsafe {
            command.pre_exec(move || {
                for descriptor in descriptors.iter().copied() {
                    let flags = libc::fcntl(descriptor, libc::F_GETFD);
                    if flags < 0 {
                        return Err(io::Error::last_os_error());
                    }
                    if flags & libc::FD_CLOEXEC != 0
                        && libc::fcntl(descriptor, libc::F_SETFD, flags & !libc::FD_CLOEXEC) < 0
                    {
                        return Err(io::Error::last_os_error());
                    }
                }
                Ok(())
            });
        }
        Ok(())
    }

    #[cfg(not(unix))]
    pub fn raw_descriptors(&self) -> impl ExactSizeIterator<Item = i32> + '_ {
        self.files
            .iter()
            .map(|_| unreachable!("no handles can be duplicated"))
    }

    #[cfg(not(unix))]
    pub fn configure_child(&self, _command: &mut std::process::Command) -> io::Result<()> {
        if self.files.is_empty() {
            Ok(())
        } else {
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "read-only worker handle inheritance is not implemented on this platform",
            ))
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use std::fs::File;
    use std::io::Write;
    use std::process::Command;

    use super::InheritedReadOnlyFiles;

    #[test]
    fn child_reads_exact_inherited_handle_while_parent_descriptors_stay_close_on_exec() {
        let mut fixture = tempfile::NamedTempFile::new().unwrap();
        fixture.write_all(b"only this verified handle\n").unwrap();
        let source = File::open(fixture.path()).unwrap();
        let parent_flags =
            unsafe { libc::fcntl(std::os::fd::AsRawFd::as_raw_fd(&source), libc::F_GETFD) };
        assert_ne!(parent_flags & libc::FD_CLOEXEC, 0);

        let inherited = InheritedReadOnlyFiles::duplicate(&[&source]).unwrap();
        let child_descriptor = inherited.raw_descriptors().next().unwrap();
        let source_descriptor = std::os::fd::AsRawFd::as_raw_fd(&source);
        let mut command = Command::new("/bin/sh");
        command.args([
            "-c",
            &format!(
                "test ! -e /dev/fd/{source_descriptor} && exec /bin/cat /dev/fd/{child_descriptor}"
            ),
        ]);
        inherited.configure_child(&mut command).unwrap();
        let output = command.output().unwrap();
        assert!(output.status.success());
        assert_eq!(output.stdout, b"only this verified handle\n");

        let parent_flags = unsafe { libc::fcntl(child_descriptor, libc::F_GETFD) };
        assert_ne!(parent_flags & libc::FD_CLOEXEC, 0);
        let source_flags =
            unsafe { libc::fcntl(std::os::fd::AsRawFd::as_raw_fd(&source), libc::F_GETFD) };
        assert_ne!(source_flags & libc::FD_CLOEXEC, 0);
    }

    #[test]
    fn writable_files_and_duplicate_handles_are_rejected() {
        let mut fixture = tempfile::NamedTempFile::new().unwrap();
        fixture.write_all(b"read-only fixture").unwrap();
        assert_eq!(
            InheritedReadOnlyFiles::duplicate(&[fixture.as_file()])
                .unwrap_err()
                .kind(),
            std::io::ErrorKind::PermissionDenied
        );

        let readonly = File::open(fixture.path()).unwrap();
        assert_eq!(
            InheritedReadOnlyFiles::duplicate(&[&readonly, &readonly])
                .unwrap_err()
                .kind(),
            std::io::ErrorKind::InvalidInput
        );
    }
}
