//! Ad-hoc latency probe (not a test): rough end-to-end cost of a small native graph and GOOF
//! encode throughput. Run: cargo run -p goofi-engine --example latency --release
use std::time::{Duration, Instant};

use goofi_core::Param;
use goofi_engine::testing::OutputProbe;
use goofi_engine::Graph;

fn main() {
    let mut g = Graph::new();
    let src = g.add_node("_TestConst", None).unwrap();
    g.update_param(src, "constant", "length", Param::int(256, 1, 1_000_000)).unwrap();
    let buf = g.add_node("Buffer", None).unwrap();
    g.update_param(buf, "buffer", "size", Param::int(1024, 1, 10_000_000)).unwrap();
    let probe = OutputProbe::open(&g, buf, "out");
    g.add_link(src, "out", buf, "data").unwrap();

    // The link attaches over a three-phase sequence that advances on acks, so wait for it.
    let ready = Instant::now();
    while ready.elapsed() < Duration::from_secs(2) {
        g.drain_status();
        std::thread::sleep(Duration::from_millis(1));
    }

    // Counted from `meta["index"]`, not caught frames: the services are latest-wins one deep.
    let index = |g: &mut Graph, p: &OutputProbe| {
        p.expect_frame(g, "the buffer to emit").meta().index().unwrap_or(0)
    };
    let first = index(&mut g, &probe);
    let t = Instant::now();
    std::thread::sleep(Duration::from_secs(2));
    let emitted = index(&mut g, &probe).saturating_sub(first);
    println!(
        "_TestConst(256)->Buffer(1024) end-to-end: {:.0} frames/s at the consumer",
        emitted as f64 / t.elapsed().as_secs_f64(),
    );

    let frame = probe.expect_frame(&mut g, "the buffer to emit");
    let n = 100_000u32;
    let t = Instant::now();
    let mut total = 0usize;
    for _ in 0..n {
        total += goofi_codec::encode(&frame).len();
    }
    let per = t.elapsed().as_secs_f64() / n as f64;
    println!(
        "GOOF encode of buffered frame ({} bytes): {:.3} us/encode ({:.1} M encodes/s)",
        total / n as usize,
        per * 1e6,
        1e-6 / per,
    );
}
