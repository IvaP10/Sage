fn main() {
    let proto = "../../proto/sage/ipc/v1/sage.proto";
    println!("cargo:rerun-if-changed={proto}");

    prost_build::Config::new()
        .compile_protos(&[proto], &["../../proto"])
        .expect("failed to compile SAGE IPC protocol");
}
