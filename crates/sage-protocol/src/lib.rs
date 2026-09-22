#![forbid(unsafe_code)]

pub mod sage {
    pub mod ipc {
        pub mod v2 {
            include!(concat!(env!("OUT_DIR"), "/sage.ipc.v2.rs"));
        }
    }
}

pub const PROTOCOL_VERSION: u32 = 2;
pub const MAX_FRAME_BYTES: usize = 4 * 1024 * 1024;
