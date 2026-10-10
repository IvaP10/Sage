//! First-party, allocation-light snapshots of memory and host CPU pressure.
//!
//! Each platform reads the narrowest native signal that can bound a new
//! allocation. The result is only a snapshot; callers must still handle later
//! allocation failure and operating-system memory pressure.

#![deny(unsafe_op_in_unsafe_fn)]

use std::io;

/// Return bytes currently available for another Sage workload.
///
/// On macOS this is the minimum of the current task's remaining dirty-memory
/// limit, when set, and reclaimable host pages. Windows reports available
/// physical memory. Linux reports `MemAvailable`, reduced by any visible
/// cgroup-v2 memory limit. Unsupported platforms fail closed.
pub fn available_bytes() -> io::Result<u64> {
    let available = platform::available_bytes()?;
    if available == 0 {
        return Err(io::Error::new(
            io::ErrorKind::OutOfMemory,
            "operating system reported no available memory",
        ));
    }
    Ok(available)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CpuCounterSource {
    #[cfg(any(target_os = "macos", test))]
    Mach,
    #[cfg(any(target_os = "windows", test))]
    Windows,
    #[cfg(any(target_os = "linux", test))]
    Linux,
}

/// A small cumulative OS CPU-time sample. It contains aggregate host counters
/// and the calling process's CPU counter, but no process or thread identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuTimeSnapshot {
    source: CpuCounterSource,
    counters: [u64; 8],
    process_cpu: u64,
}

impl CpuTimeSnapshot {
    #[cfg(any(target_os = "macos", test))]
    fn mach(counters: [u32; 4], process_cpu: u64) -> Self {
        Self {
            source: CpuCounterSource::Mach,
            counters: [
                u64::from(counters[0]),
                u64::from(counters[1]),
                u64::from(counters[2]),
                u64::from(counters[3]),
                0,
                0,
                0,
                0,
            ],
            process_cpu,
        }
    }

    #[cfg(any(target_os = "windows", test))]
    fn windows(idle: u64, kernel: u64, user: u64, process_cpu: u64) -> Self {
        Self {
            source: CpuCounterSource::Windows,
            counters: [idle, kernel, user, 0, 0, 0, 0, 0],
            process_cpu,
        }
    }

    #[cfg(any(target_os = "linux", test))]
    fn linux(counters: [u64; 8], process_cpu: u64) -> Self {
        Self {
            source: CpuCounterSource::Linux,
            counters,
            process_cpu,
        }
    }

    /// Busy CPU time in permille between these ordered cumulative samples.
    /// Returns `None` for mismatched sources, non-advancing counters, or
    /// malformed deltas rather than inventing a load estimate.
    pub fn busy_permille_since(self, previous: Self) -> Option<u16> {
        let (busy, total) = self.busy_and_total_since(previous)?;
        busy_time_permille(busy, total)
    }

    /// Busy CPU time outside this process, in permille between these ordered
    /// cumulative samples. Host and process counters use the same native time
    /// unit on each supported platform. Invalid or inconsistent deltas return
    /// `None`; the caller should take a new baseline instead of guessing.
    pub fn busy_permille_excluding_process_since(self, previous: Self) -> Option<u16> {
        let (busy, total) = self.busy_and_total_since(previous)?;
        let process = u128::from(self.process_cpu.checked_sub(previous.process_cpu)?);
        busy_time_permille(busy.checked_sub(process)?, total)
    }

    fn busy_and_total_since(self, previous: Self) -> Option<(u128, u128)> {
        if self.source != previous.source {
            return None;
        }
        let busy_and_total = match self.source {
            #[cfg(any(target_os = "macos", test))]
            CpuCounterSource::Mach => {
                let user = delta_u32(self.counters[0], previous.counters[0]);
                let system = delta_u32(self.counters[1], previous.counters[1]);
                let idle = delta_u32(self.counters[2], previous.counters[2]);
                let nice = delta_u32(self.counters[3], previous.counters[3]);
                let total =
                    u128::from(user) + u128::from(system) + u128::from(idle) + u128::from(nice);
                (total.checked_sub(u128::from(idle))?, total)
            }
            #[cfg(any(target_os = "windows", test))]
            CpuCounterSource::Windows => {
                let idle = self.counters[0].checked_sub(previous.counters[0])?;
                let kernel = self.counters[1].checked_sub(previous.counters[1])?;
                let user = self.counters[2].checked_sub(previous.counters[2])?;
                let total = u128::from(kernel) + u128::from(user);
                (total.checked_sub(u128::from(idle))?, total)
            }
            #[cfg(any(target_os = "linux", test))]
            CpuCounterSource::Linux => {
                let mut deltas = [0_u64; 8];
                for (index, delta) in deltas.iter_mut().enumerate() {
                    *delta = self.counters[index].checked_sub(previous.counters[index])?;
                }
                let total = deltas.iter().map(|value| u128::from(*value)).sum::<u128>();
                let idle = u128::from(deltas[3]) + u128::from(deltas[4]);
                (total.checked_sub(idle)?, total)
            }
        };
        Some(busy_and_total)
    }
}

/// Read cumulative native host CPU counters. Call on a blocking worker; pair
/// two snapshots around a bounded sampling interval before scheduling work.
pub fn cpu_time_snapshot() -> io::Result<CpuTimeSnapshot> {
    platform::cpu_time_snapshot()
}

fn busy_time_permille(busy: u128, total: u128) -> Option<u16> {
    if total == 0 || busy > total {
        return None;
    }
    u16::try_from(busy.checked_mul(1_000)? / total).ok()
}

#[cfg(any(target_os = "macos", test))]
fn delta_u32(current: u64, previous: u64) -> u64 {
    (current as u32).wrapping_sub(previous as u32) as u64
}

#[cfg(target_os = "macos")]
mod platform {
    use super::io;
    use std::mem::size_of;

    const KERN_SUCCESS: i32 = 0;
    const TASK_VM_INFO: u32 = 22;
    const HOST_VM_INFO64: i32 = 4;
    const HOST_CPU_LOAD_INFO: i32 = 3;

    // Prefix through `limit_bytes_remaining` in Apple's TASK_VM_INFO rev4.
    // The Mach headers pack this record to four-byte alignment.
    #[repr(C, packed(4))]
    struct TaskVmInfoRev4 {
        virtual_size: u64,
        region_count: i32,
        page_size: i32,
        resident_size: u64,
        resident_size_peak: u64,
        device: u64,
        device_peak: u64,
        internal: u64,
        internal_peak: u64,
        external: u64,
        external_peak: u64,
        reusable: u64,
        reusable_peak: u64,
        purgeable_volatile_pmap: u64,
        purgeable_volatile_resident: u64,
        purgeable_volatile_virtual: u64,
        compressed: u64,
        compressed_peak: u64,
        compressed_lifetime: u64,
        phys_footprint: u64,
        min_address: u64,
        max_address: u64,
        revision_three_ledgers: [i64; 21],
        limit_bytes_remaining: u64,
    }

    #[repr(C)]
    struct VmStatistics64Prefix {
        free_count: u32,
        active_count: u32,
        inactive_count: u32,
        wire_count: u32,
        zero_fill_count: u64,
        reactivations: u64,
        pageins: u64,
        pageouts: u64,
        faults: u64,
        cow_faults: u64,
        lookups: u64,
        hits: u64,
        purges: u64,
        purgeable_count: u32,
        speculative_count: u32,
    }

    #[repr(C)]
    struct HostCpuLoadInfo {
        cpu_ticks: [u32; 4],
    }

    #[link(name = "System")]
    unsafe extern "C" {
        static mach_task_self_: u32;
        fn mach_host_self() -> u32;
        fn mach_port_deallocate(task: u32, name: u32) -> i32;
        fn task_info(
            target_task: u32,
            flavor: u32,
            task_info_out: *mut i32,
            task_info_out_count: *mut u32,
        ) -> i32;
        fn host_statistics64(
            host: u32,
            flavor: i32,
            host_info_out: *mut i32,
            host_info_out_count: *mut u32,
        ) -> i32;
        fn host_statistics(
            host: u32,
            flavor: i32,
            host_info_out: *mut i32,
            host_info_out_count: *mut u32,
        ) -> i32;
        fn host_page_size(host: u32, page_size: *mut usize) -> i32;
    }

    pub(super) fn available_bytes() -> io::Result<u64> {
        let process_limit = task_memory_limit_remaining()?;
        let system_available = host_available_bytes()?;
        Ok(cap_by_process_limit(process_limit, system_available))
    }

    pub(super) fn cpu_time_snapshot() -> io::Result<super::CpuTimeSnapshot> {
        // SAFETY: mach_host_self acquires a host send right that HostPort
        // releases on every exit path.
        let host = unsafe { mach_host_self() };
        if host == 0 {
            return Err(io::Error::other("mach_host_self returned a null port"));
        }
        let _host_right = HostPort(host);
        let mut information = HostCpuLoadInfo { cpu_ticks: [0; 4] };
        let expected_count = u32::try_from(size_of::<HostCpuLoadInfo>() / size_of::<u32>())
            .map_err(|_| io::Error::other("HOST_CPU_LOAD_INFO count overflow"))?;
        let mut count = expected_count;
        // SAFETY: `information` matches the four-u32 host_cpu_load_info ABI,
        // and the guarded host port is valid for this read-only snapshot.
        let result = unsafe {
            host_statistics(
                host,
                HOST_CPU_LOAD_INFO,
                (&mut information as *mut HostCpuLoadInfo).cast::<i32>(),
                &mut count,
            )
        };
        if result != KERN_SUCCESS {
            return Err(io::Error::other(format!(
                "host_statistics(HOST_CPU_LOAD_INFO) failed with Mach status {result}"
            )));
        }
        if count < expected_count {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "host_statistics returned incomplete CPU counters",
            ));
        }
        let process_cpu = process_cpu_ticks()?;
        Ok(super::CpuTimeSnapshot::mach(
            information.cpu_ticks,
            process_cpu,
        ))
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct TimeVal {
        seconds: i64,
        microseconds: i32,
        padding: i32,
    }

    #[repr(C)]
    struct ResourceUsage {
        user_time: TimeVal,
        system_time: TimeVal,
        other: [i64; 14],
    }

    const RUSAGE_SELF: i32 = 0;
    const SC_CLK_TCK: i32 = 3;
    #[link(name = "System")]
    unsafe extern "C" {
        fn getrusage(who: i32, usage: *mut ResourceUsage) -> i32;
        fn sysconf(name: i32) -> i64;
    }

    fn process_cpu_ticks() -> io::Result<u64> {
        if size_of::<TimeVal>() != 16 || size_of::<ResourceUsage>() != 144 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "getrusage layout does not match the 64-bit macOS ABI",
            ));
        }
        let mut usage = ResourceUsage {
            user_time: TimeVal {
                seconds: 0,
                microseconds: 0,
                padding: 0,
            },
            system_time: TimeVal {
                seconds: 0,
                microseconds: 0,
                padding: 0,
            },
            other: [0; 14],
        };
        // SAFETY: `usage` has the SDK-checked 64-bit `struct rusage` layout
        // and RUSAGE_SELF requests cumulative CPU time for this process.
        if unsafe { getrusage(RUSAGE_SELF, &mut usage) } != 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: sysconf is read-only and SC_CLK_TCK reports the statistics
        // clock frequency used by HOST_CPU_LOAD_INFO's cumulative tick values.
        let ticks_per_second = unsafe { sysconf(SC_CLK_TCK) };
        if ticks_per_second <= 0 {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "statistics-clock frequency is unavailable",
            ));
        }
        let process_micros = timeval_micros(usage.user_time)?
            .checked_add(timeval_micros(usage.system_time)?)
            .ok_or_else(|| io::Error::other("process CPU time overflow"))?;
        u64::try_from(
            process_micros
                .checked_mul(ticks_per_second as u128)
                .ok_or_else(|| io::Error::other("process CPU tick conversion overflow"))?
                / 1_000_000,
        )
        .map_err(|_| io::Error::other("process CPU tick count exceeds u64"))
    }

    fn timeval_micros(value: TimeVal) -> io::Result<u128> {
        if value.seconds < 0 || !(0..1_000_000).contains(&value.microseconds) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "getrusage returned an invalid timeval",
            ));
        }
        u128::try_from(value.seconds)
            .ok()
            .and_then(|seconds| seconds.checked_mul(1_000_000))
            .and_then(|micros| micros.checked_add(value.microseconds as u128))
            .ok_or_else(|| io::Error::other("getrusage timeval overflow"))
    }

    fn cap_by_process_limit(process_limit: u64, system_available: u64) -> u64 {
        if process_limit == 0 {
            system_available
        } else {
            process_limit.min(system_available)
        }
    }

    fn task_memory_limit_remaining() -> io::Result<u64> {
        const REV4_BYTES: usize = 344;
        if size_of::<TaskVmInfoRev4>() != REV4_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "TASK_VM_INFO rev4 layout does not match the native ABI",
            ));
        }

        let mut info = TaskVmInfoRev4 {
            virtual_size: 0,
            region_count: 0,
            page_size: 0,
            resident_size: 0,
            resident_size_peak: 0,
            device: 0,
            device_peak: 0,
            internal: 0,
            internal_peak: 0,
            external: 0,
            external_peak: 0,
            reusable: 0,
            reusable_peak: 0,
            purgeable_volatile_pmap: 0,
            purgeable_volatile_resident: 0,
            purgeable_volatile_virtual: 0,
            compressed: 0,
            compressed_peak: 0,
            compressed_lifetime: 0,
            phys_footprint: 0,
            min_address: 0,
            max_address: 0,
            revision_three_ledgers: [0; 21],
            limit_bytes_remaining: 0,
        };
        let mut count = u32::try_from(size_of::<TaskVmInfoRev4>() / size_of::<u32>())
            .map_err(|_| io::Error::other("TASK_VM_INFO word count overflow"))?;

        // SAFETY: `mach_task_self_` is the current task. `info` is a writable,
        // four-byte-aligned buffer exactly matching Apple's rev4 prefix, and
        // `count` is the corresponding count of 32-bit natural_t words.
        let result = unsafe {
            task_info(
                mach_task_self_,
                TASK_VM_INFO,
                (&mut info as *mut TaskVmInfoRev4).cast::<i32>(),
                &mut count,
            )
        };
        if result != KERN_SUCCESS {
            return Err(io::Error::other(format!(
                "task_info(TASK_VM_INFO) failed with Mach status {result}"
            )));
        }
        if usize::try_from(count)
            .ok()
            .is_none_or(|words| words * size_of::<u32>() < size_of::<TaskVmInfoRev4>())
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "TASK_VM_INFO did not return the rev4 memory-limit field",
            ));
        }

        // Copy the packed field by value; do not create an unaligned reference.
        Ok(info.limit_bytes_remaining)
    }

    struct HostPort(u32);

    impl Drop for HostPort {
        fn drop(&mut self) {
            // SAFETY: this port right was returned to this task by
            // `mach_host_self` and is released exactly once here.
            unsafe {
                let _ = mach_port_deallocate(mach_task_self_, self.0);
            }
        }
    }

    fn host_available_bytes() -> io::Result<u64> {
        const VM_STATS_PREFIX_BYTES: usize = 96;
        if size_of::<VmStatistics64Prefix>() != VM_STATS_PREFIX_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "vm_statistics64 prefix layout does not match the native ABI",
            ));
        }

        // SAFETY: `mach_host_self` returns a send right owned by this task; the
        // `HostPort` guard releases it on every return path.
        let host = unsafe { mach_host_self() };
        if host == 0 {
            return Err(io::Error::other("mach_host_self returned a null port"));
        }
        let _host_right = HostPort(host);
        let mut statistics = VmStatistics64Prefix {
            free_count: 0,
            active_count: 0,
            inactive_count: 0,
            wire_count: 0,
            zero_fill_count: 0,
            reactivations: 0,
            pageins: 0,
            pageouts: 0,
            faults: 0,
            cow_faults: 0,
            lookups: 0,
            hits: 0,
            purges: 0,
            purgeable_count: 0,
            speculative_count: 0,
        };
        let mut count = u32::try_from(size_of::<VmStatistics64Prefix>() / size_of::<u32>())
            .map_err(|_| io::Error::other("vm_statistics64 word count overflow"))?;
        // SAFETY: `statistics` has the exact native prefix size, natural_t
        // alignment, and count required through `speculative_count`.
        let result = unsafe {
            host_statistics64(
                host,
                HOST_VM_INFO64,
                (&mut statistics as *mut VmStatistics64Prefix).cast::<i32>(),
                &mut count,
            )
        };
        if result != KERN_SUCCESS {
            return Err(io::Error::other(format!(
                "host_statistics64(HOST_VM_INFO64) failed with Mach status {result}"
            )));
        }
        if usize::try_from(count).ok().is_none_or(|words| {
            words
                .checked_mul(size_of::<u32>())
                .is_none_or(|bytes| bytes < size_of::<VmStatistics64Prefix>())
        }) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "host_statistics64 returned an incomplete memory snapshot",
            ));
        }

        let mut page_size = 0usize;
        // SAFETY: `page_size` points to writable vm_size_t storage for this
        // 64-bit target and `host` is the live host port guarded above.
        let result = unsafe { host_page_size(host, &mut page_size) };
        if result != KERN_SUCCESS || page_size == 0 {
            return Err(io::Error::other(format!(
                "host_page_size failed with Mach status {result}"
            )));
        }

        // Speculative pages are already included in free_count per Apple's
        // vm_statistics64 contract. Inactive pages can be reclaimed, so count
        // them once without double-counting speculative pages.
        let available_pages = u64::from(statistics.free_count)
            .checked_add(u64::from(statistics.inactive_count))
            .ok_or_else(|| io::Error::other("available Mach page count overflow"))?;
        available_pages
            .checked_mul(
                u64::try_from(page_size).map_err(|_| {
                    io::Error::other("Mach page size exceeds the supported byte count")
                })?,
            )
            .ok_or_else(|| io::Error::other("available Mach memory byte count overflow"))
    }

    #[cfg(test)]
    mod tests {
        use super::cap_by_process_limit;

        #[test]
        fn absent_task_limit_uses_host_snapshot_and_present_limit_caps_it() {
            assert_eq!(cap_by_process_limit(0, 5_000), 5_000);
            assert_eq!(cap_by_process_limit(7_000, 5_000), 5_000);
            assert_eq!(cap_by_process_limit(3_000, 5_000), 3_000);
        }
    }
}

#[cfg(target_os = "ios")]
mod platform {
    use super::io;

    #[link(name = "System")]
    unsafe extern "C" {
        fn os_proc_available_memory() -> usize;
    }

    pub(super) fn available_bytes() -> io::Result<u64> {
        // SAFETY: this process-local Apple system function takes no arguments.
        let bytes = unsafe { os_proc_available_memory() };
        u64::try_from(bytes).map_err(|_| io::Error::other("available-memory value exceeds u64"))
    }

    pub(super) fn cpu_time_snapshot() -> io::Result<super::CpuTimeSnapshot> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "native CPU load counters are unavailable on iOS",
        ))
    }
}

#[cfg(target_os = "windows")]
mod platform {
    use super::io;
    use std::ffi::c_void;
    use std::mem::size_of;

    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_physical: u64,
        available_physical: u64,
        total_page_file: u64,
        available_page_file: u64,
        total_virtual: u64,
        available_virtual: u64,
        available_extended_virtual: u64,
    }

    #[repr(C)]
    struct FileTime {
        low_date_time: u32,
        high_date_time: u32,
    }

    impl FileTime {
        fn ticks(&self) -> u64 {
            (u64::from(self.high_date_time) << 32) | u64::from(self.low_date_time)
        }
    }

    #[link(name = "Kernel32")]
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(status: *mut MemoryStatusEx) -> i32;
        fn GetSystemTimes(
            idle_time: *mut FileTime,
            kernel_time: *mut FileTime,
            user_time: *mut FileTime,
        ) -> i32;
        fn GetCurrentProcess() -> *mut c_void;
        fn GetProcessTimes(
            process: *mut c_void,
            creation_time: *mut FileTime,
            exit_time: *mut FileTime,
            kernel_time: *mut FileTime,
            user_time: *mut FileTime,
        ) -> i32;
    }

    pub(super) fn available_bytes() -> io::Result<u64> {
        let mut status = MemoryStatusEx {
            length: u32::try_from(size_of::<MemoryStatusEx>())
                .map_err(|_| io::Error::other("MEMORYSTATUSEX layout is too large"))?,
            memory_load: 0,
            total_physical: 0,
            available_physical: 0,
            total_page_file: 0,
            available_page_file: 0,
            total_virtual: 0,
            available_virtual: 0,
            available_extended_virtual: 0,
        };

        // SAFETY: `status` points to the initialized MEMORYSTATUSEX layout and
        // its `length` field is set to the exact buffer size required by Win32.
        let succeeded = unsafe { GlobalMemoryStatusEx(&mut status) };
        if succeeded == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(status.available_physical)
    }

    pub(super) fn cpu_time_snapshot() -> io::Result<super::CpuTimeSnapshot> {
        let mut idle = FileTime {
            low_date_time: 0,
            high_date_time: 0,
        };
        let mut kernel = FileTime {
            low_date_time: 0,
            high_date_time: 0,
        };
        let mut user = FileTime {
            low_date_time: 0,
            high_date_time: 0,
        };
        // SAFETY: each pointer references an initialized FILETIME output value;
        // GetSystemTimes only writes cumulative system CPU counters.
        let succeeded = unsafe { GetSystemTimes(&mut idle, &mut kernel, &mut user) };
        if succeeded == 0 {
            return Err(io::Error::last_os_error());
        }
        let mut creation = FileTime {
            low_date_time: 0,
            high_date_time: 0,
        };
        let mut exit = FileTime {
            low_date_time: 0,
            high_date_time: 0,
        };
        let mut process_kernel = FileTime {
            low_date_time: 0,
            high_date_time: 0,
        };
        let mut process_user = FileTime {
            low_date_time: 0,
            high_date_time: 0,
        };
        // SAFETY: GetCurrentProcess returns a pseudo-handle owned by the
        // current process; GetProcessTimes writes four initialized FILETIMEs.
        let succeeded = unsafe {
            GetProcessTimes(
                GetCurrentProcess(),
                &mut creation,
                &mut exit,
                &mut process_kernel,
                &mut process_user,
            )
        };
        if succeeded == 0 {
            return Err(io::Error::last_os_error());
        }
        let process_cpu = process_kernel
            .ticks()
            .checked_add(process_user.ticks())
            .ok_or_else(|| io::Error::other("process CPU counter overflow"))?;
        Ok(super::CpuTimeSnapshot::windows(
            idle.ticks(),
            kernel.ticks(),
            user.ticks(),
            process_cpu,
        ))
    }
}

#[cfg(target_os = "linux")]
mod platform {
    use super::io;
    use std::fs;

    pub(super) fn available_bytes() -> io::Result<u64> {
        let contents = fs::read_to_string("/proc/meminfo")?;
        let host_available = parse_mem_available(&contents)?;
        Ok(
            cgroup_v2_available().map_or(host_available, |limit_available| {
                host_available.min(limit_available)
            }),
        )
    }

    pub(super) fn cpu_time_snapshot() -> io::Result<super::CpuTimeSnapshot> {
        let contents = fs::read_to_string("/proc/stat")?;
        let mut snapshot = parse_cpu_time(&contents).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "aggregate CPU counters are missing or malformed",
            )
        })?;
        let process = fs::read_to_string("/proc/self/stat")?;
        snapshot.process_cpu = parse_process_cpu_time(&process).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "calling-process CPU counters are missing or malformed",
            )
        })?;
        Ok(snapshot)
    }

    fn parse_cpu_time(contents: &str) -> Option<super::CpuTimeSnapshot> {
        let mut counters = None;
        for line in contents.lines() {
            let mut fields = line.split_ascii_whitespace();
            if fields.next() != Some("cpu") {
                continue;
            }
            if counters.is_some() {
                return None;
            }
            let mut values = [0_u64; 8];
            let mut count = 0;
            for field in fields {
                if count == values.len() {
                    break;
                }
                values[count] = field.parse().ok()?;
                count += 1;
            }
            if count < 4 {
                return None;
            }
            counters = Some(super::CpuTimeSnapshot::linux(values, 0));
        }
        counters
    }

    fn parse_process_cpu_time(contents: &str) -> Option<u64> {
        // The command name is parenthesized and may itself contain spaces or
        // closing parentheses. Fields after the final ')' begin at field 3;
        // utime/stime are fields 14/15 (zero-based positions 11/12 here).
        let after_name = contents.get(contents.rfind(')')?.checked_add(1)?..)?;
        let mut fields = after_name.split_ascii_whitespace();
        let user = fields.nth(11)?.parse::<u64>().ok()?;
        let system = fields.next()?.parse::<u64>().ok()?;
        user.checked_add(system)
    }

    fn parse_mem_available(contents: &str) -> io::Result<u64> {
        let mut result = None;
        for line in contents.lines() {
            let mut fields = line.split_ascii_whitespace();
            if fields.next() != Some("MemAvailable:") {
                continue;
            }
            if result.is_some() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "duplicate MemAvailable entry",
                ));
            }
            let kibibytes = fields
                .next()
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "missing MemAvailable value")
                })?
                .parse::<u64>()
                .map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, "invalid MemAvailable value")
                })?;
            if fields.next() != Some("kB") || fields.next().is_some() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid MemAvailable unit or trailing fields",
                ));
            }
            result = Some(kibibytes.checked_mul(1024).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "MemAvailable value overflow")
            })?);
        }
        result.ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "MemAvailable entry is missing")
        })
    }

    fn cgroup_v2_available() -> Option<u64> {
        let limit = fs::read_to_string("/sys/fs/cgroup/memory.max")
            .ok()?
            .trim()
            .parse::<u64>()
            .ok()?;
        let current = fs::read_to_string("/sys/fs/cgroup/memory.current")
            .ok()?
            .trim()
            .parse::<u64>()
            .ok()?;
        Some(limit.saturating_sub(current))
    }

    #[cfg(test)]
    mod tests {
        use super::{parse_cpu_time, parse_mem_available, parse_process_cpu_time};

        #[test]
        fn parses_only_one_checked_memavailable_value() {
            assert_eq!(
                parse_mem_available("MemTotal: 8 kB\nMemAvailable: 12 kB\n").unwrap(),
                12 * 1024
            );
            for invalid in [
                "MemAvailable: 18446744073709551615 kB\n",
                "MemAvailable: nope kB\n",
                "MemAvailable: 12 MB\n",
                "MemAvailable: 12 kB\nMemAvailable: 13 kB\n",
                "MemTotal: 12 kB\n",
            ] {
                assert!(parse_mem_available(invalid).is_err());
            }
        }

        #[test]
        fn parses_aggregate_cpu_counters_and_ignores_per_cpu_rows() {
            let first = parse_cpu_time("cpu 100 20 50 200 10 5 5 0 3 1\ncpu0 40 5 20 90 5 2 2 0\n")
                .unwrap();
            let next = parse_cpu_time("cpu 150 20 90 230 15 8 7 0 4 1\n").unwrap();
            assert_eq!(next.busy_permille_since(first), Some(730));
            assert!(parse_cpu_time("cpu 1 2 3\n").is_none());
            assert!(parse_cpu_time("cpu 1 2 3 4\ncpu 2 3 4 5\n").is_none());
            assert!(parse_cpu_time("cpu 1 2 no 4\n").is_none());
        }

        #[test]
        fn parses_process_cpu_after_a_command_name_with_spaces_and_parentheses() {
            let stat = "123 (sage worker ) v2) S 1 2 3 4 5 6 7 8 9 10 120 30 16 17 18 19 20\n";
            assert_eq!(parse_process_cpu_time(stat), Some(150));
            assert!(parse_process_cpu_time("123 (unterminated S 1 2").is_none());
            assert!(
                parse_process_cpu_time("123 (sage) S 1 2 3 4 5 6 7 8 9 10 11 nope 2").is_none()
            );
        }
    }
}

#[cfg(not(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "windows",
    target_os = "linux"
)))]
mod platform {
    use super::io;

    pub(super) fn available_bytes() -> io::Result<u64> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "native memory admission is unavailable on this platform",
        ))
    }

    pub(super) fn cpu_time_snapshot() -> io::Result<super::CpuTimeSnapshot> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "native CPU load counters are unavailable on this platform",
        ))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn native_probe_returns_a_nonzero_snapshot() {
        assert!(super::available_bytes().expect("native memory probe") > 0);
    }

    #[test]
    fn native_cpu_probe_returns_monotonic_process_samples() {
        let before = super::cpu_time_snapshot().expect("initial native CPU snapshot");
        let start = std::time::Instant::now();
        let mut work = 0_u64;
        while start.elapsed() < std::time::Duration::from_millis(150) {
            work = std::hint::black_box(work.wrapping_add(1));
        }
        std::hint::black_box(work);
        let after = super::cpu_time_snapshot().expect("subsequent native CPU snapshot");
        assert!(after.process_cpu > before.process_cpu);
    }

    #[test]
    fn cpu_counter_deltas_are_checked_and_platform_specific() {
        let before = super::CpuTimeSnapshot::mach([u32::MAX - 4, 20, 100, 7], 10);
        let after = super::CpuTimeSnapshot::mach([5, 30, 120, 9], 13);
        assert_eq!(after.busy_permille_since(before), Some(523));
        assert_eq!(
            after.busy_permille_excluding_process_since(before),
            Some(452)
        );

        let before = super::CpuTimeSnapshot::windows(100, 400, 200, 50);
        let after = super::CpuTimeSnapshot::windows(120, 480, 220, 60);
        assert_eq!(after.busy_permille_since(before), Some(800));
        assert_eq!(
            after.busy_permille_excluding_process_since(before),
            Some(700)
        );

        let same = super::CpuTimeSnapshot::windows(120, 500, 230, 60);
        assert_eq!(same.busy_permille_since(same), None);
        assert_eq!(same.busy_permille_excluding_process_since(same), None);
        assert_eq!(
            super::CpuTimeSnapshot::linux([1, 2, 3, 4, 5, 6, 7, 8], 1).busy_permille_since(before),
            None
        );
    }

    #[test]
    fn process_cpu_subtraction_fails_closed_on_resets_or_inconsistent_intervals() {
        let before = super::CpuTimeSnapshot::linux([100, 0, 50, 200, 0, 0, 0, 0], 10);
        let after = super::CpuTimeSnapshot::linux([120, 0, 70, 220, 0, 0, 0, 0], 50);
        assert_eq!(after.busy_permille_since(before), Some(666));
        assert_eq!(after.busy_permille_excluding_process_since(before), Some(0));

        let process_reset = super::CpuTimeSnapshot::linux([120, 0, 70, 220, 0, 0, 0, 0], 9);
        assert_eq!(
            process_reset.busy_permille_excluding_process_since(before),
            None
        );

        let process_exceeds_host = super::CpuTimeSnapshot::linux([101, 0, 50, 201, 0, 0, 0, 0], 20);
        assert_eq!(
            process_exceeds_host.busy_permille_excluding_process_since(before),
            None
        );
    }
}
