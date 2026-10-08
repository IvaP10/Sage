//! Run with `cargo run -p sage-core --release --example intent_latency --locked`.
//! This measures grammar/reflex CPU time only, not recognition, IPC or OS work.
use std::hint::black_box;
use std::time::Instant;

fn measure(name: &str, mut operation: impl FnMut()) {
    for _ in 0..1_000 {
        operation();
    }
    let mut samples = Vec::with_capacity(10_000);
    for _ in 0..10_000 {
        let started = Instant::now();
        operation();
        samples.push(started.elapsed().as_nanos());
    }
    samples.sort_unstable();
    println!(
        "{name}: n={} p50={} ns p95={} ns p99={} ns",
        samples.len(),
        samples[5_000],
        samples[9_500],
        samples[9_900]
    );
}

fn main() {
    println!(
        "Sage local intent benchmark; {}; {}; optimized={}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        !cfg!(debug_assertions)
    );
    measure("interrupt_reflex", || {
        black_box(sage_core::intent::reflex(black_box("No, open Firefox")));
    });
    measure("compile_three_actions", || {
        black_box(sage_core::intent::compile(
            black_box("open Chrome and open Firefox then open VS Code"),
            &[],
        ));
    });
    measure("reject_incomplete_intent", || {
        black_box(sage_core::intent::compile(
            black_box("open Chrome and then"),
            &[],
        ));
    });
    println!(
        "Microbenchmark only: excludes speech recognition, storage, IPC, approval and native execution."
    );
}
