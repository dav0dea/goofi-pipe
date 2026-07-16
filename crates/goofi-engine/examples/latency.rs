//! Ad-hoc latency probe (not a test): rough per-tick cost of a small native
//! DSP graph and GOOF encode throughput. Run: cargo run -p goofi-engine --example latency --release
use std::time::Instant;
use goofi_engine::Graph;
use goofi_core::Param;

fn main() {
    // Oscillator -> Buffer(1024) -> PSD
    let mut g = Graph::new();
    let osc = g.add_node("Oscillator", None).unwrap();
    g.update_param(osc, "oscillator", "n_samples", Param::int(256, 1, 1_000_000)).unwrap();
    let buf = g.add_node("Buffer", None).unwrap();
    g.update_param(buf, "buffer", "size", Param::int(1024, 1, 10_000_000)).unwrap();
    let psd = g.add_node("PSD", None).unwrap();
    g.add_link(osc, "out", buf, "data").unwrap();
    g.add_link(buf, "out", psd, "data").unwrap();

    // warm up
    for _ in 0..200 { g.tick(); }

    let iters = 20_000u32;
    let t = Instant::now();
    for _ in 0..iters { g.tick(); }
    let per = t.elapsed().as_secs_f64() / iters as f64;
    println!("Oscillator->Buffer(1024)->PSD full-graph tick: {:.2} us/tick ({:.0} ticks/s)",
        per * 1e6, 1.0 / per);

    // encode throughput of the PSD frame
    let frame = g.latest_frame(psd, "psd").expect("psd frame");
    let n = 100_000u32;
    let t = Instant::now();
    let mut total = 0usize;
    for _ in 0..n { total += goofi_codec::encode(&frame).len(); }
    let per = t.elapsed().as_secs_f64() / n as f64;
    println!("GOOF encode of PSD frame ({} bytes): {:.3} us/encode ({:.1} M encodes/s)",
        total / n as usize, per * 1e6, 1e-6 / per);
}
