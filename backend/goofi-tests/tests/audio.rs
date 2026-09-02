//! The audio engine under the external clock: one session, every action through the op
//! vocabulary, every probe the interleaved output a device would receive.

use goofi_tests::{hex, j, Goofi};

/// Render `frames` and hand back what the device would get, interleaved, with its channel count.
fn drive(g: &Goofi, frames: usize) -> (Vec<f32>, u16) {
    let mut graph = g.state.graph.lock().unwrap();
    goofi_bridge::audio_engine(&mut graph).drive(frames)
}

fn peak(v: &[f32]) -> f32 {
    v.iter().fold(0f32, |m, x| m.max(x.abs()))
}

fn mean(v: &[f32]) -> f32 {
    v.iter().sum::<f32>() / v.len() as f32
}

fn crossings(v: &[f32]) -> usize {
    v.windows(2).filter(|w| (w[0] < 0.0) != (w[1] < 0.0)).count()
}

/// A tenth of a second at the engine's rate: 44 cycles of 440 Hz, 88 zero crossings.
const TENTH: usize = 4800;

fn near(a: usize, b: usize) -> bool {
    a.abs_diff(b) <= 2
}

#[test]
fn a_patch_sounds_under_the_external_clock() {
    let g = Goofi::new();

    // Step: the palette lists the audio category, and a chain of three sounds at once.
    let types = g.call("library list", j!({}));
    let audio: Vec<&str> = types["types"].as_array().unwrap().iter()
        .filter(|r| r["category"] == "audio").filter_map(|r| r["type"].as_str()).collect();
    for want in ["Osc", "Gain", "AudioOut"] {
        assert!(audio.contains(&want), "`{want}` is missing from the audio palette: {audio:?}");
    }
    let osc = g.add("Osc");
    let gain = g.add("Gain");
    let out = g.add("AudioOut");
    g.link(osc, "out", gain, "input");
    g.link(gain, "out", out, "input");
    let (a, channels) = drive(&g, TENTH);
    assert_eq!(channels, 1, "a mono chain is heard in mono");
    assert!((peak(&a) - 1.0).abs() < 0.01, "a full-scale sine: peak {}", peak(&a));
    assert!(near(crossings(&a), 88), "440 Hz: {} crossings", crossings(&a));

    // Step: a constant param lands at the next block — the gain, then the frequency.
    g.set_param(gain, "gain", "gain", 0.5);
    let (b, _) = drive(&g, TENTH);
    assert!((peak(&b) - 0.5).abs() < 0.01, "half gain: peak {}", peak(&b));
    g.set_param(osc, "osc", "freq", 880.0);
    let (c, _) = drive(&g, TENTH);
    assert!(near(crossings(&c), 176), "880 Hz: {} crossings", crossings(&c));

    // Step: a param referencing an audio output is a plan edge at audio rate — the gain reads
    // the oscillator itself, so the output is its square: never negative, full scale.
    let osc_name = g.doc()["nodes"][hex(osc)]["name"].as_str().unwrap().to_string();
    let bound = g.call(
        "node param edit",
        j!({ "node": hex(gain), "param": "gain/gain", "reference": format!("{osc_name}.out"), "mode": "reference" }),
    );
    assert!(bound["error"].is_null(), "{bound}");
    let (d, _) = drive(&g, TENTH);
    assert!(d.iter().all(|x| *x >= -1e-6), "sin² is never negative: min {}", d.iter().cloned().fold(f32::MAX, f32::min));
    assert!((peak(&d) - 1.0).abs() < 0.01, "sin² peaks at one: {}", peak(&d));

    // Step: a multi input sums its wires. A frozen square (0 Hz from phase zero) is a constant
    // one; with the gain a constant half again, the sum is a half-sine on a half offset.
    let osc2 = g.add("Osc");
    g.set_param(osc2, "osc", "shape", "square");
    g.set_param(osc2, "osc", "freq", 0.0);
    g.link(osc2, "out", gain, "input");
    g.call("node param edit", j!({ "node": hex(gain), "param": "gain/gain", "value": 0.5, "mode": "constant" }));
    let (e, _) = drive(&g, TENTH);
    assert!((peak(&e) - 1.0).abs() < 0.01, "sine plus one, halved: peak {}", peak(&e));
    assert!((mean(&e) - 0.5).abs() < 0.02, "…on a half offset: mean {}", mean(&e));

    // Step: a remove takes its wire out of the sum; a second remove silences the chain; undo
    // brings the oscillator back with its frequency.
    g.call("node remove", j!({ "node": hex(osc2) }));
    let (f, _) = drive(&g, TENTH);
    assert!((peak(&f) - 0.5).abs() < 0.01 && near(crossings(&f), 176), "the sine alone: peak {} crossings {}", peak(&f), crossings(&f));
    g.call("node remove", j!({ "node": hex(osc) }));
    let (silent, _) = drive(&g, TENTH);
    assert_eq!(peak(&silent), 0.0, "nothing feeds the gain");
    assert_eq!(g.call("undo", j!({}))["changed"], true);
    let (back, _) = drive(&g, TENTH);
    assert!((peak(&back) - 0.5).abs() < 0.01 && near(crossings(&back), 176), "undo restored the 880 Hz oscillator: peak {} crossings {}", peak(&back), crossings(&back));
}
