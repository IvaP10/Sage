fn main() {
    let proto = "../../proto/sage/ipc/v2/sage.proto";
    println!("cargo:rerun-if-changed={proto}");

    prost_build::Config::new()
        .boxed(".sage.ipc.v2.Frame.payload.adapter_request")
        // Progress carries a full bounded action list. Keep other event/frame
        // variants compact without changing the protobuf wire format.
        .boxed(".sage.ipc.v2.CoreEvent.event.task_update")
        .compile_protos(&[proto], &["../../proto"])
        .expect("failed to compile SAGE IPC protocol");
}
