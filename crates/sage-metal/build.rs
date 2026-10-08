use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/metal_bridge.m");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo sets OUT_DIR"));
    let object = out_dir.join("metal_bridge.o");
    let archive = out_dir.join("libsage_metal_bridge.a");
    let target = env::var("TARGET").expect("Cargo sets TARGET");
    let sdk = env::var("SDKROOT")
        .ok()
        .filter(|path| !path.is_empty())
        .unwrap_or_else(|| {
            let output = Command::new("xcrun")
                .args(["--sdk", "macosx", "--show-sdk-path"])
                .output()
                .expect("xcrun is required to locate the macOS SDK");
            if !output.status.success() {
                panic!("xcrun could not locate the macOS SDK");
            }
            String::from_utf8(output.stdout)
                .expect("SDK path is UTF-8")
                .trim()
                .to_owned()
        });

    let status = Command::new("clang")
        .args([
            "-target",
            &target,
            "-isysroot",
            &sdk,
            "-fobjc-arc",
            "-fmodules",
            "-O2",
            "-c",
            "src/metal_bridge.m",
            "-o",
        ])
        .arg(&object)
        .status()
        .expect("clang is required to build Sage's Metal bridge");
    if !status.success() {
        panic!("failed to compile Sage's first-party Metal bridge");
    }

    let status = Command::new("ar")
        .arg("crus")
        .arg(&archive)
        .arg(&object)
        .status()
        .expect("ar is required to archive Sage's Metal bridge");
    if !status.success() {
        panic!("failed to archive Sage's first-party Metal bridge");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=sage_metal_bridge");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=CoreFoundation");
}
