//! The audio engine under the external clock: one session, every action through the op
//! vocabulary, every probe the interleaved output a device would receive.

use goofi_tests::{ep, hex, j, Goofi, Uid};

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

fn state(g: &Goofi, uid: Uid) -> String {
    g.call("node state", j!({ "node": hex(uid) }))["text"].as_str().unwrap().to_string()
}

/// A square held HIGH: a quarter of a hertz from phase zero stays at one for the two seconds
/// this walk drives after it is born.
fn frozen_square(g: &Goofi) -> Uid {
    let osc = g.add("Osc");
    g.set_param(osc, "osc", "shape", "square");
    g.set_param(osc, "osc", "pitch", -10.0);
    osc
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
    assert!(near(crossings(&a), 88), "A4 by default: {} crossings", crossings(&a));

    // Step: a constant param lands at the next block — the gain, then the pitch, in volts per octave.
    g.set_param(gain, "gain", "gain", 0.5);
    let (b, _) = drive(&g, TENTH);
    assert!((peak(&b) - 0.5).abs() < 0.01, "half gain: peak {}", peak(&b));
    g.set_param(osc, "osc", "pitch", 1.75);
    let (c, _) = drive(&g, TENTH);
    assert!(near(crossings(&c), 176), "an octave up, 880 Hz: {} crossings", crossings(&c));

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

    // Step: a multi input sums its wires. A held square is a constant one; with the gain a
    // constant half again, the sum is a half-sine on a half offset.
    let osc2 = frozen_square(&g);
    g.link(osc2, "out", gain, "input");
    g.call("node param edit", j!({ "node": hex(gain), "param": "gain/gain", "value": 0.5, "mode": "constant" }));
    let (e, _) = drive(&g, TENTH);
    assert!((peak(&e) - 1.0).abs() < 0.01, "sine plus one, halved: peak {}", peak(&e));
    assert!((mean(&e) - 0.5).abs() < 0.02, "…on a half offset: mean {}", mean(&e));

    // Step: a reference the graph refuses is never a plan edge: a Str param cannot read an audio
    // output, the record keeps the literal, and the sound is unchanged.
    let refused = g.call(
        "node param edit",
        j!({ "node": hex(osc2), "param": "osc/shape", "reference": format!("{osc_name}.out"), "mode": "reference" }),
    );
    assert!(refused["error"].as_str().is_some_and(|e| e.contains("AUDIO") && e.contains("STRING")), "{refused}");
    let (e2, _) = drive(&g, TENTH);
    assert!((peak(&e2) - 1.0).abs() < 0.01 && (mean(&e2) - 0.5).abs() < 0.02, "the literal stands: peak {} mean {}", peak(&e2), mean(&e2));

    // Step: a loop with no feedback node is excluded and named; what it feeds runs on silence,
    // and breaking the loop clears the fault and the sound comes back.
    let gain2 = g.add("Gain");
    g.link(gain, "out", gain2, "input");
    g.link(gain2, "out", gain, "input");
    for uid in [gain, gain2] {
        g.until("the loop fault to reach the node's state", |g| state(g, uid).contains("loop").then_some(()));
    }
    let (looped, _) = drive(&g, TENTH);
    assert_eq!(peak(&looped), 0.0, "the device hears silence behind the loop");
    g.call("link remove", j!({ "from": ep(hex(gain2), "out"), "to": ep(hex(gain), "input") }));
    g.until("the fault to clear", |g| (!state(g, gain).contains("loop")).then_some(()));
    let (unlooped, _) = drive(&g, TENTH);
    assert!((peak(&unlooped) - 1.0).abs() < 0.01 && (mean(&unlooped) - 0.5).abs() < 0.02, "peak {} mean {}", peak(&unlooped), mean(&unlooped));
    g.call("node remove", j!({ "node": hex(gain2) }));

    // Step: the slab grows past its first 64 slots without losing an instance — the 65th node's
    // wire lands in the sum and every earlier one keeps running.
    let many: Vec<Uid> = (0..64).map(|_| frozen_square(&g)).collect();
    g.link(*many.last().unwrap(), "out", gain, "input");
    let (grown, _) = drive(&g, TENTH);
    assert!((peak(&grown) - 1.5).abs() < 0.01 && (mean(&grown) - 1.0).abs() < 0.02, "sine plus two, halved: peak {} mean {}", peak(&grown), mean(&grown));
    for uid in many {
        g.call("node remove", j!({ "node": hex(uid) }));
    }
    let (shrunk, _) = drive(&g, TENTH);
    assert!((peak(&shrunk) - 1.0).abs() < 0.01 && (mean(&shrunk) - 0.5).abs() < 0.02, "peak {} mean {}", peak(&shrunk), mean(&shrunk));

    // Step: a remove takes its wire out of the sum; a second remove silences the chain; undo
    // brings the oscillator back with its pitch.
    g.call("node remove", j!({ "node": hex(osc2) }));
    let (f, _) = drive(&g, TENTH);
    assert!((peak(&f) - 0.5).abs() < 0.01 && near(crossings(&f), 176), "the sine alone: peak {} crossings {}", peak(&f), crossings(&f));
    g.call("node remove", j!({ "node": hex(osc) }));
    let (silent, _) = drive(&g, TENTH);
    assert_eq!(peak(&silent), 0.0, "nothing feeds the gain");
    assert_eq!(g.call("undo", j!({}))["changed"], true);
    let (back, _) = drive(&g, TENTH);
    assert!((peak(&back) - 0.5).abs() < 0.01 && near(crossings(&back), 176), "undo restored the 880 Hz oscillator: peak {} crossings {}", peak(&back), crossings(&back));

    // Step: with two AudioOut nodes the lowest uid is heard, whatever order the plan runs them
    // in — a fresh chain into the first one sounds although a newer, silent one comes earlier.
    g.call("node remove", j!({ "node": hex(gain) }));
    let out2 = g.add("AudioOut");
    let osc3 = g.add("Osc");
    let gain3 = g.add("Gain");
    g.link(osc3, "out", gain3, "input");
    g.link(gain3, "out", out, "input");
    let (heard, _) = drive(&g, TENTH);
    assert!((peak(&heard) - 1.0).abs() < 0.01 && near(crossings(&heard), 88), "the first AudioOut is heard: peak {} crossings {}", peak(&heard), crossings(&heard));
    assert!(g.doc()["nodes"][hex(out2)].is_object(), "the second one stands, unheard");
}
