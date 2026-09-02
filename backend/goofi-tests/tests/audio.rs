//! The audio engine under the external clock: one session, every action through the op
//! vocabulary, every probe the interleaved output a device would receive.

use std::sync::Arc;

use goofi_tests::{ep, f32s, hex, j, shape, FirstVar, Goofi, Uid};

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
    g.state.graph.lock().unwrap().set_evaluator(Arc::new(FirstVar));

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

    // Step: a binding the control half evaluates lands at control rate — the gain follows a
    // signal-plane constant through `nd()`, and the value it evaluated to rides the wire.
    let mut events = g.events();
    let source = g.add("_TestConst");
    let source_name = g.doc()["nodes"][hex(source)]["name"].as_str().unwrap().to_string();
    g.set_param(source, "constant", "value", 0.25);
    let bound = g.call(
        "node param edit",
        j!({ "node": hex(gain3), "param": "gain/gain", "expression": format!("nd('{source_name}')"), "mode": "expression" }),
    );
    assert!(bound["error"].is_null(), "{bound}");
    g.until("the constant to modulate the gain", |g| {
        let (x, _) = drive(g, TENTH);
        ((peak(&x) - 0.25).abs() < 0.01).then_some(())
    });
    let reported = events.next("param_values");
    assert_eq!(reported["node"], hex(gain3), "{reported}");
    assert_eq!(reported["values"]["gain"]["gain"], 0.25, "{reported}");
    g.set_param(source, "constant", "value", 0.75);
    g.until("the gain to follow its source", |g| {
        let (x, _) = drive(g, TENTH);
        ((peak(&x) - 0.75).abs() < 0.01).then_some(())
    });

    // Step: a plan swap keeps every instance — the oscillator's phase runs on across it.
    let (a, _) = drive(&g, TENTH);
    let spare = g.add("Gain");
    let (b, _) = drive(&g, TENTH);
    assert!((b[0] - a[a.len() - 1]).abs() < 0.1, "no jump at the swap: {} then {}", a[a.len() - 1], b[0]);
    g.call("node remove", j!({ "node": hex(spare) }));

    // Step: the tap — every reader of an audio output subscribes on the derived name: a probe
    // sees whole blocks at the rate, a signal Buffer fills from them, and a snapshot answers.
    let tapped = g.probe(osc3, "out");
    drive(&g, TENTH);
    let block = tapped.expect_frame(&mut g.state.graph.lock().unwrap(), "the oscillator's tapped blocks");
    assert_eq!(shape(&block)[0], 1, "{:?}", shape(&block));
    assert!(shape(&block)[1] >= 64 && shape(&block)[1].is_multiple_of(64), "whole blocks: {:?}", shape(&block));
    assert_eq!(block.meta().sfreq(), Some(48_000.0));
    assert!(crossings(&f32s(&block)) > 0 && peak(&f32s(&block)) <= 1.0, "the sine, as it sounds");
    let buffer = g.add("Buffer");
    g.link(osc3, "out", buffer, "data");
    let buffered = g.probe(buffer, "out");
    drive(&g, TENTH);
    let filled = buffered.expect_frame(&mut g.state.graph.lock().unwrap(), "the buffer to fill from the tap");
    assert!(shape(&filled)[0] == 1 && crossings(&f32s(&filled)) > 0, "the buffer holds the sine: {:?}", shape(&filled));
    let snapshot = g.until("a snapshot of the gain's output", |g| {
        drive(g, TENTH);
        let answer = g.call("node snapshot", j!({ "output": ep(hex(gain3), "out") }));
        answer["npy_b64"].is_string().then_some(answer)
    });
    assert_eq!(snapshot["meta"]["sfreq"], 48000.0, "{snapshot}");

    // Step: the in-order crossing — a signal ramp at 256 Hz enters through `SignalIn`, resampled
    // to the rate: it rises from zero, a tenth of a second is a twentieth of it, and the next
    // tenth continues where this one stopped.
    g.set_param(source, "constant", "value", 0.0);
    g.until("the gain to close", |g| {
        let (x, _) = drive(g, TENTH);
        (peak(&x) == 0.0).then_some(())
    });
    g.call("node remove", j!({ "node": hex(buffer) }));
    let ramp = g.add("_TestRamp");
    let signal_in = g.add("SignalIn");
    g.link(ramp, "out", signal_in, "data");
    g.link(signal_in, "out", out, "input");
    let first = g.until("the ramp to enter", |g| {
        let (x, _) = drive(g, TENTH);
        (peak(&x) > 0.0).then_some(x)
    });
    let rising: Vec<f32> = first.iter().copied().skip_while(|x| *x == 0.0).collect();
    assert!(rising.windows(2).all(|w| w[1] >= w[0]), "the ramp rises in order");
    let top = *rising.last().unwrap();
    assert!(rising[0] < 0.01 && top <= 0.051, "from zero, a twentieth per tenth: {} .. {top}", rising[0]);
    let (second, _) = drive(&g, TENTH);
    assert!(second[0] >= top - 0.001 && second.windows(2).all(|w| w[1] >= w[0]), "…and continues: {} after {top}", second[0]);

    // Step: a gate from the signal plane — `Env.gate` referencing a constant — opens the envelope
    // at control rate, and dropping it releases.
    g.call("node remove", j!({ "node": hex(signal_in) }));
    g.call("node remove", j!({ "node": hex(ramp) }));
    let env = g.add("Env");
    let gate = g.add("_TestConst");
    g.link(env, "out", out, "input");
    let gate_name = g.doc()["nodes"][hex(gate)]["name"].as_str().unwrap().to_string();
    let bound = g.call(
        "node param edit",
        j!({ "node": hex(env), "param": "env/gate", "reference": format!("{gate_name}.out"), "mode": "reference" }),
    );
    assert!(bound["error"].is_null(), "{bound}");
    let (shut, _) = drive(&g, TENTH);
    assert_eq!(peak(&shut), 0.0, "a low gate is silence");
    g.set_param(gate, "constant", "value", 1.0);
    g.until("the gate to open", |g| {
        let (x, _) = drive(g, TENTH);
        (x[x.len() - 1] > 0.99).then_some(())
    });
    g.set_param(gate, "constant", "value", 0.0);
    g.until("the release to finish", |g| {
        let (x, _) = drive(g, TENTH);
        (x[x.len() - 1] == 0.0).then_some(())
    });

    // Step: an audio-rate gate is one voice per channel — a four-channel ramp through `SignalIn`
    // as the gate: the channel still below the threshold is shut, the three above it sound.
    let ramp4 = g.add("_TestRamp");
    g.set_param(ramp4, "ramp", "channels", 4);
    let in4 = g.add("SignalIn");
    g.link(ramp4, "out", in4, "data");
    let in4_name = g.doc()["nodes"][hex(in4)]["name"].as_str().unwrap().to_string();
    let bound = g.call(
        "node param edit",
        j!({ "node": hex(env), "param": "env/gate", "reference": format!("{in4_name}.out"), "mode": "reference" }),
    );
    assert!(bound["error"].is_null(), "{bound}");
    g.until("four voices, one shut", |g| {
        let (x, channels) = drive(g, TENTH);
        let last = &x[x.len() - 4..];
        (channels == 4 && last[0] == 0.0 && last[1..].iter().all(|v| *v > 0.99)).then_some(())
    });
    for uid in [env, in4, ramp4] {
        g.call("node remove", j!({ "node": hex(uid) }));
    }

    // Step: a stereo source into the mono chain — the channel count follows the widest part,
    // and the second channel is its own.
    let ramp2 = g.add("_TestRamp");
    g.set_param(ramp2, "ramp", "channels", 2);
    let in2 = g.add("SignalIn");
    g.link(ramp2, "out", in2, "data");
    g.link(in2, "out", out, "input");
    g.until("stereo", |g| {
        let (x, channels) = drive(g, TENTH);
        (channels == 2 && (x[x.len() - 1] - x[x.len() - 2] - 1.0).abs() < 0.01).then_some(())
    });
    for uid in [in2, ramp2] {
        g.call("node remove", j!({ "node": hex(uid) }));
    }

    // Step: a loop closed by `Feedback` runs, and no fault names it: a held one through half
    // gain and back converges on one, and opening the loop halves it again.
    let held = frozen_square(&g);
    g.call("link remove", j!({ "from": ep(hex(osc3), "out"), "to": ep(hex(gain3), "input") }));
    g.link(held, "out", gain3, "input");
    g.set_param(source, "constant", "value", 0.5);
    let fb = g.add("Feedback");
    g.link(gain3, "out", fb, "input");
    g.link(fb, "out", gain3, "input");
    assert!(g.stays(|g| !state(g, gain3).contains("loop")), "a feedback node closes the loop without a fault");
    g.until("the loop to converge", |g| {
        let (x, _) = drive(g, TENTH);
        let tail = &x[x.len() - 480..];
        ((mean(tail) - 1.0).abs() < 0.01 && peak(&x) <= 1.001).then_some(())
    });
    g.call("node remove", j!({ "node": hex(fb) }));
    let (opened, _) = drive(&g, TENTH);
    assert!((mean(&opened[opened.len() - 480..]) - 0.5).abs() < 0.01, "the loop opened: {}", mean(&opened[opened.len() - 480..]));
}
