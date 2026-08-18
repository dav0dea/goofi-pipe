//! Ad-hoc latency probe (not a test): rough end-to-end cost of a small native graph and GOOF
//! encode throughput. Run: cargo run -p goofi-engine --example latency --release
//!
//! There is no tick to time any more, so what this measures changed with the runtime: the number is
//! the rate a two-node chain actually SUSTAINS, producer to consumer, over shared memory — which is
//! the thing a patch's throughput now depends on. A per-tick cost had no meaning once every node
//! paced itself.
use std::time::{Duration, Instant};

use goofi_core::Param;
use goofi_engine::testing::OutputProbe;
use goofi_engine::Graph;

fn main() {
    // _TestConst(256) -> Buffer(1024), both uncapped, so the chain runs as fast as it can.
    let mut g = Graph::new();
    let src = g.add_node("_TestConst", None).unwrap();
    g.update_param(src, "constant", "length", Param::int(256, 1, 1_000_000)).unwrap();
    let buf = g.add_node("Buffer", None).unwrap();
    g.update_param(buf, "buffer", "size", Param::int(1024, 1, 10_000_000)).unwrap();
    let probe = OutputProbe::open(&g, buf, "out");
    g.add_link(src, "out", buf, "data").unwrap();

    // The link is attached by the three-phase sequence, which advances on acks — so the chain is
    // not carrying data until the status drain has run a few times.
    let ready = Instant::now();
    while ready.elapsed() < Duration::from_secs(2) {
        g.drain_status();
        std::thread::sleep(Duration::from_millis(1));
    }

    // Counted from `meta["index"]`, which advances once per emit, rather than from how many frames
    // the probe happens to catch: the data services are latest-wins one deep, so a subscriber that
    // looks less often than the producer emits legitimately sees fewer.
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

    // encode throughput of the buffered frame
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
