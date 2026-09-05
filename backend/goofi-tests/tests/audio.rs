//! The audio engine under the external clock: one session, every action through the op
//! vocabulary, every probe the interleaved output a device would receive.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use goofi_tests::{drive, ep, f32s, hex, j, shape, FirstVar, Goofi, Uid};

/// Drive tenths until the output satisfies `want` — a constant lands within one control hop, and
/// the tenth after shows it — and hand that tenth back.
fn sounds(g: &Goofi, what: &str, want: impl Fn(&[f32]) -> bool) -> Vec<f32> {
    g.until(what, |g| {
        let (x, _) = drive(g, TENTH);
        want(&x).then_some(x)
    })
}

/// Drive tenths until CHANNEL 0 of `uid`'s own output satisfies `want`, and hand that back — the
/// node itself rather than the device's sum, for a node no `AudioOut` is behind.
fn heard(g: &Goofi, uid: Uid, what: &str, want: impl Fn(&[f32]) -> bool) -> Vec<f32> {
    let probe = g.probe(uid, "out");
    g.until(what, |g| {
        drive(g, TENTH);
        let d = probe.latest()?;
        let lane = f32s(&d)[..shape(&d)[1]].to_vec();
        want(&lane).then_some(lane)
    })
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

/// Crossings scaled to a tenth: a tap's frame holds whatever the last tick drained.
fn per_tenth(v: &[f32]) -> usize {
    crossings(v) * TENTH / v.len().max(1)
}

fn state(g: &Goofi, uid: Uid) -> String {
    g.call("node state", j!({ "node": hex(uid) }))["text"].as_str().unwrap().to_string()
}

/// Ask for a param's list and read it off the `state_update` every client gets — the echo that
/// clears the spinner — rather than through any door of the test's own.
fn refreshed(g: &Goofi, ev: &mut goofi_tests::Events, uid: Uid, group: &str, name: &str) -> Vec<String> {
    g.call("node param refresh", j!({ "node": hex(uid), "param": format!("{group}/{name}") }));
    let p = g.until("the refresh echo", |_| {
        let p = ev.next("state_update");
        (p["node"] == hex(uid) && p["refreshed_params"] == j!([[group, name]])).then_some(p)
    });
    p["params"][group][name]["options"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect()
}

/// A constant one: an envelope gated open with no attack, which sustains at one for as long as
/// the walk runs. NOT an oscillator held below its first edge — that is a wall-clock budget, and
/// a step that drove a few tenths more read every later sum at minus one.
fn held_one(g: &Goofi) -> Uid {
    let env = g.add("Env");
    g.set_param(env, "env", "attack", 0.0);
    g.set_param(env, "env", "gate", true);
    env
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

    // Step: the palette lists the audio engine's types, and a chain of three sounds at once.
    let types = g.call("library list", j!({}));
    let mut audio: Vec<&str> = types["types"].as_array().unwrap().iter()
        .filter_map(|r| r["type"].as_str()).filter(|t| t.starts_with("audio:")).collect();
    audio.sort_unstable();
    // The shipped set, whole: three built in because their control halves own OS handles, and
    // seven files built by the same pipeline an authored node takes.
    assert_eq!(audio, ["audio:AudioIn", "audio:AudioOut", "audio:Env", "audio:Feedback", "audio:Gain",
                       "audio:MidiIn", "audio:Osc", "audio:SignalIn", "audio:Slew", "audio:Svf"]);
    let osc = g.add("Osc");
    let gain = g.add("Gain");
    let out = g.add("AudioOut");
    g.link(osc, "out", gain, "input");
    g.link(gain, "out", out, "input");
    let (a, channels) = drive(&g, TENTH);
    assert_eq!(channels, 1, "a mono chain is heard in mono");
    assert!((peak(&a) - 1.0).abs() < 0.01, "a full-scale sine: peak {}", peak(&a));
    assert!(near(crossings(&a), 88), "A4 by default: {} crossings", crossings(&a));

    // Step: a constant param lands within one control hop — the gain, then the pitch, in volts
    // per octave.
    g.set_param(gain, "gain", "gain", 0.5);
    sounds(&g, "half gain", |x| (peak(x) - 0.5).abs() < 0.01);
    g.set_param(osc, "osc", "pitch", 1.75);
    sounds(&g, "an octave up, 880 Hz", |x| near(crossings(x), 176));

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
    let osc2 = held_one(&g);
    g.link(osc2, "out", gain, "input");
    g.call("node param edit", j!({ "node": hex(gain), "param": "gain/gain", "value": 0.5, "mode": "constant" }));
    sounds(&g, "sine plus one, halved, on a half offset", |x| (peak(x) - 1.0).abs() < 0.01 && (mean(x) - 0.5).abs() < 0.02);

    // Step: a reference the graph refuses is never a plan edge: a Str param cannot read an audio
    // output, the record keeps the literal, and the sound is unchanged.
    let refused = g.call(
        "node param edit",
        j!({ "node": hex(osc), "param": "osc/shape", "reference": format!("{osc_name}.out"), "mode": "reference" }),
    );
    assert!(refused["error"].as_str().is_some_and(|e| e.contains("AUDIO") && e.contains("STRING")), "{refused}");
    let (e2, _) = drive(&g, TENTH);
    assert!((peak(&e2) - 1.0).abs() < 0.01 && (mean(&e2) - 0.5).abs() < 0.02, "the literal stands: peak {} mean {}", peak(&e2), mean(&e2));

    // Step: two nodes that carry state across blocks, and neither can fake it. The filter is
    // taken to a cutoff far below its source — which its default would not answer — then across
    // its modes, then made resonant, where the peak it builds needs many blocks of memory to
    // reach. The slew is given a full-scale square and a one-second slope: what comes out is a
    // hundredth of it.
    let square = g.add("Osc");
    g.set_param(square, "osc", "shape", "square");
    let filter = g.add("Svf");
    g.link(square, "out", filter, "input");
    g.set_param(filter, "filter", "cutoff", -4.0);
    heard(&g, filter, "an A4 square five octaves under the cutoff", |x| peak(x) < 0.2);
    g.set_param(filter, "filter", "mode", "high");
    heard(&g, filter, "the same corner, passing everything above it", |x| peak(x) > 0.9 && near(per_tenth(x), 88));
    g.set_param(filter, "filter", "mode", "low");
    g.set_param(filter, "filter", "cutoff", 0.75);
    g.set_param(filter, "filter", "q", 10.0);
    heard(&g, filter, "resonance built across blocks", |x| peak(x) > 3.0);

    let slew = g.add("Slew");
    g.set_param(slew, "slew", "rise", 1.0);
    g.set_param(slew, "slew", "fall", 1.0);
    g.link(square, "out", slew, "input");
    heard(&g, slew, "a full-scale square rate-limited to a hundredth", |x| peak(x) > 0.0 && peak(x) < 0.01);
    for uid in [square, filter, slew] {
        g.call("node remove", j!({ "node": hex(uid) }));
    }

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
    let many: Vec<Uid> = (0..64).map(|_| held_one(&g)).collect();
    let stillborn: Vec<String> = many.iter().filter_map(|u| g.error(*u)).collect();
    assert!(stillborn.is_empty(), "every one of them starts: {stillborn:?}");
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

    // Step: every AudioOut naming the device sums into it, each through its own gain; one naming
    // another device is told where the clock is, and leaves the sum until it agrees.
    g.call("node remove", j!({ "node": hex(gain) }));
    let out2 = g.add("AudioOut");
    let osc3 = g.add("Osc");
    let gain3 = g.add("Gain");
    g.link(osc3, "out", gain3, "input");
    g.link(gain3, "out", out, "input");
    g.link(gain3, "out", out2, "input");
    let (both, _) = drive(&g, TENTH);
    assert!((peak(&both) - 2.0).abs() < 0.02 && near(crossings(&both), 88), "two outputs sum: peak {} crossings {}", peak(&both), crossings(&both));
    g.set_param(out2, "audio", "gain", 0.5);
    sounds(&g, "the second output's own gain", |x| (peak(x) - 1.5).abs() < 0.02);
    g.set_param(out2, "audio", "device", "elsewhere");
    let why = g.until("the clock's holder to be named", |g| g.error(out2));
    assert!(why.contains("the clock is on `default`"), "{why}");
    sounds(&g, "the disagreeing output to leave the sum", |x| (peak(x) - 1.0).abs() < 0.01);
    g.set_param(out2, "audio", "device", "default");
    g.until("the fault to clear", |g| g.error(out2).is_none().then_some(()));
    sounds(&g, "…and to rejoin", |x| (peak(x) - 1.5).abs() < 0.02);
    g.call("node remove", j!({ "node": hex(out2) }));

    // Step: the device list is a refresh answered by the node's own thread and echoed to every
    // client, the host default first; the clock itself reports through `session status`.
    let mut ev = g.events();
    assert_eq!(refreshed(&g, &mut ev, out, "audio", "device")[0], "default");
    let status = g.call("session status", j!({}));
    assert_eq!(status["audio"]["clock"], "external", "{status}");
    assert_eq!(status["audio"]["rate"], 48000.0, "{status}");
    assert!(status["audio"]["device"].is_null(), "no device under the external clock: {status}");
    assert!(status["audio"]["channels"].as_u64().is_some_and(|c| c >= 1), "{status}");
    assert_eq!((status["audio"]["callbacks"].as_u64(), status["audio"]["xruns"].as_u64()), (Some(0), Some(0)), "{status}");
    // A pulse is refused on a param that is not one; no shipped audio node declares a pulse yet.
    let why = g.refuse("node param pulse", j!({ "node": hex(gain3), "param": "gain/gain" }));
    assert!(why.contains("not a pulse"), "{why}");

    // Step: a device or a port that is not there is an error on the param that named it, and
    // what can be named is a refresh; a MIDI port's voices are the channels a gate sees.
    g.set_param(gain3, "gain", "gain", 0.0);
    sounds(&g, "the chain to fall silent", |x| peak(x) == 0.0);
    let mic = g.add("AudioIn");
    // The external clock renders at `drive`'s speed, which no live capture stream can meet: the
    // default device is resolved and never opened, and a machine with no card says so instead.
    let why = g.until("the default device to be refused", |g| g.error(mic));
    assert!(
        why.contains("the external clock owns no device") || why.contains("no default input device"),
        "an AudioIn under the external clock names a reason rather than opening the microphone: {why}"
    );
    g.set_param(mic, "audio", "device", "nowhere");
    let why = g.until("the absent device to be named", |g| g.error(mic).filter(|e| e.contains("nowhere")));
    assert!(why.contains("no input device `nowhere`"), "{why}");
    assert_eq!(refreshed(&g, &mut ev, mic, "audio", "device")[0], "default");
    g.call("node remove", j!({ "node": hex(mic) }));
    let midi = g.add("MidiIn");
    g.set_param(midi, "midi", "port", "nowhere");
    let why = g.until("the absent port to be named", |g| g.error(midi));
    // A kernel built without sound — GitHub's Linux runner — has no sequencer to look a port up in.
    let sequencer = midir::MidiOutput::new("goofi-test").is_ok();
    assert_eq!(why.contains("no MIDI port `nowhere`"), sequencer, "{why}");
    g.set_param(midi, "midi", "port", "none");
    g.until("the port error to clear", |g| g.error(midi).is_none().then_some(()));
    assert_eq!(refreshed(&g, &mut ev, midi, "midi", "port")[0], "none");
    g.set_param(midi, "midi", "voices", 3);
    let voices = g.add("Env");
    let midi_name = g.doc()["nodes"][hex(midi)]["name"].as_str().unwrap().to_string();
    let bound = g.call(
        "node param edit",
        j!({ "node": hex(voices), "param": "env/gate", "reference": format!("{midi_name}.gate"), "mode": "reference" }),
    );
    assert!(bound["error"].is_null(), "{bound}");
    g.link(voices, "out", out, "input");
    g.until("three silent voices", |g| {
        let (x, channels) = drive(g, TENTH);
        (channels == 3 && peak(&x[x.len() - 3..]) == 0.0).then_some(())
    });

    // Step: a note through a real port takes the first voice — its gate, its pitch in volts per
    // octave and its velocity — the next notes take the next voices in turn, and a release frees
    // them. A virtual port, which WinMM has none of, and a machine with no sequencer cannot open.
    #[cfg(unix)]
    {
        use midir::os::unix::VirtualOutput;
        if sequencer {
            let mut keys = midir::MidiOutput::new("goofi-test")
                .expect("a MIDI client")
                .create_virtual("goofi-test-out")
                .expect("a virtual MIDI port");
            let port = g.until("the virtual port to be listed", |g| {
                refreshed(g, &mut ev, midi, "midi", "port").into_iter().find(|n| n.contains("goofi-test-out"))
            });
            g.set_param(midi, "midi", "port", port.as_str());
            let pitch = g.probe(midi, "pitch");
            let velocity = g.probe(midi, "velocity");
            g.until("A4 to take the first voice", |g| {
                keys.send(&[0x90, 69, 100]).expect("a note on"); // sent again per poll: a held note keeps its voice
                let (x, channels) = drive(g, TENTH);
                let heard = channels == 3 && x[x.len() - 3] > 0.99;
                let played = pitch.latest().is_some_and(|d| (f32s(&d)[0] - 0.75).abs() < 1e-3)
                    && velocity.latest().is_some_and(|d| (f32s(&d)[0] - 100.0 / 127.0).abs() < 1e-3);
                (heard && played).then_some(())
            });
            keys.send(&[0x90, 72, 64]).expect("a note on");
            keys.send(&[0x90, 76, 127]).expect("a note on");
            g.until("C5 and E5 to take the second and third voices", |g| {
                drive(g, TENTH);
                let voice = |d: &goofi_core::Data, c: usize| f32s(d)[c * shape(d)[1]];
                let placed = |d: &goofi_core::Data| shape(d)[0] == 3 && (voice(d, 1) - 1.0).abs() < 1e-3 && (voice(d, 2) - 4.0 / 3.0).abs() < 1e-3;
                pitch.latest().is_some_and(|d| placed(&d)).then_some(())
            });
            g.until("the voices to release", |g| {
                for note in [69, 72, 76] {
                    keys.send(&[0x80, note, 0]).expect("a note off");
                }
                let (x, _) = drive(g, TENTH);
                (peak(&x[x.len() - 3..]) == 0.0).then_some(())
            });
        }
    }
    g.call("node remove", j!({ "node": hex(voices) }));
    g.call("node remove", j!({ "node": hex(midi) }));

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
    let reported = loop {
        let ev = events.next("param_values");
        if ev["node"] == hex(gain3) && !ev["values"]["gain"].is_null() {
            break ev;
        }
    };
    assert_eq!(reported["values"]["gain"]["gain"], 0.25, "{reported}");
    g.set_param(source, "constant", "value", 0.75);
    sounds(&g, "the gain to follow its source", |x| (peak(x) - 0.75).abs() < 0.01);

    // Step: a binding that evaluates to NaN is a binding error like any other — the param names
    // it and the node reads silence — because a NaN is not a value a plan can carry.
    let poison = g.add("_TestConst");
    let poison_name = g.doc()["nodes"][hex(poison)]["name"].as_str().unwrap().to_string();
    g.set_param(poison, "constant", "nan", true);
    g.call("node param edit", j!({ "node": hex(gain3), "param": "gain/gain", "reference": format!("{poison_name}.out"), "mode": "reference" }));
    let why = g.until("the NaN to be refused", |g| g.error(gain3));
    assert!(why.contains("evaluated to NaN"), "{why}");
    sounds(&g, "silence where the NaN would have played", |x| x.iter().all(|v| v.is_finite()) && peak(x) == 0.0);
    g.call("node param edit", j!({ "node": hex(gain3), "param": "gain/gain", "value": 1.0, "mode": "constant" }));
    g.until("the error to clear", |g| g.error(gain3).is_none().then_some(()));

    // Step: a reference the control half cannot copy is a binding error on the node; back on a
    // constant, the literal lands and the error clears.
    g.set_param(source, "constant", "length", 4);
    let bound = g.call(
        "node param edit",
        j!({ "node": hex(gain3), "param": "gain/gain", "reference": format!("{source_name}.out"), "mode": "reference" }),
    );
    assert!(bound["error"].is_null(), "{bound}");
    let why = g.until("the binding error to reach the node", |g| g.error(gain3));
    assert!(why.contains("one element"), "{why}");
    g.call("node param edit", j!({ "node": hex(gain3), "param": "gain/gain", "value": 0.5, "mode": "constant" }));
    sounds(&g, "the literal to land", |x| (peak(x) - 0.5).abs() < 0.01);
    g.until("the error to clear", |g| g.error(gain3).is_none().then_some(()));

    // Step: a plan swap keeps every instance — the oscillator's phase runs on across it.
    let (a, _) = drive(&g, TENTH);
    let spare = g.add("Gain");
    let (b, _) = drive(&g, TENTH);
    assert!((b[0] - a[a.len() - 1]).abs() < 0.1, "no jump at the swap: {} then {}", a[a.len() - 1], b[0]);
    g.call("node remove", j!({ "node": hex(spare) }));

    // Step: the tap — every reader of an audio output subscribes on the derived name: a probe
    // sees whole blocks at the rate, a signal Buffer fills from them, and a snapshot answers.
    let tapped = g.probe(osc3, "out");
    let block = g.until("the oscillator's tapped blocks", |g| {
        drive(g, TENTH);
        tapped.latest()
    });
    assert_eq!(shape(&block)[0], 1, "{:?}", shape(&block));
    assert!(shape(&block)[1] >= 64 && shape(&block)[1].is_multiple_of(64), "whole blocks: {:?}", shape(&block));
    assert_eq!(block.meta().sfreq(), Some(48_000.0));
    assert!(crossings(&f32s(&block)) > 0 && peak(&f32s(&block)) <= 1.0, "the sine, as it sounds");
    let buffer = g.add("Buffer");
    g.link(osc3, "out", buffer, "data");
    let buffered = g.probe(buffer, "out");
    let filled = g.until("the buffer to fill from the tap", |g| {
        drive(g, TENTH);
        buffered.latest()
    });
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
    g.set_param(gain3, "gain", "gain", 0.0);
    sounds(&g, "the gain to close", |x| peak(x) == 0.0);
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
    let span = second[second.len() - 1] - second[0];
    assert!((span - 0.05).abs() < 0.002, "a whole tenth is a twentieth of the ramp: {span}");

    // Step: a frame that is not a number crosses as silence — a NaN stays on the plane that made
    // it and never enters the plan.
    g.call("node remove", j!({ "node": hex(ramp) }));
    g.link(poison, "out", signal_in, "data");
    let poisoned = g.probe(poison, "out");
    g.until("the NaN to be emitted", |g| {
        drive(g, TENTH);
        poisoned.latest().filter(|d| f32s(d)[0].is_nan())
    });
    sounds(&g, "silence where the NaN entered", |x| x.iter().all(|v| v.is_finite()) && peak(x) == 0.0);
    g.call("node remove", j!({ "node": hex(poison) }));

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
    sounds(&g, "the release to finish", |x| x[x.len() - 1] == 0.0);

    // Step: the gate's other two sources open it the same way — a constant, then an expression
    // over the signal-plane constant.
    g.call("node param edit", j!({ "node": hex(env), "param": "env/gate", "value": true, "mode": "constant" }));
    sounds(&g, "a constant gate to open", |x| x[x.len() - 1] > 0.99);
    let bound = g.call(
        "node param edit",
        j!({ "node": hex(env), "param": "env/gate", "expression": format!("nd('{gate_name}')"), "mode": "expression" }),
    );
    assert!(bound["error"].is_null(), "{bound}");
    sounds(&g, "an expression gate to close", |x| x[x.len() - 1] == 0.0);
    g.set_param(gate, "constant", "value", 1.0);
    sounds(&g, "…and to open", |x| x[x.len() - 1] > 0.99);

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
    let held = held_one(&g);
    g.call("link remove", j!({ "from": ep(hex(osc3), "out"), "to": ep(hex(gain3), "input") }));
    g.link(held, "out", gain3, "input");
    g.set_param(gain3, "gain", "gain", 0.5);
    let fb = g.add("Feedback");
    g.link(gain3, "out", fb, "input");
    g.link(fb, "out", gain3, "input");
    assert!(g.stays(|g| !state(g, gain3).contains("loop")), "a feedback node closes the loop without a fault");
    g.until("the loop to converge", |g| {
        let (x, _) = drive(g, TENTH);
        let tail = &x[x.len() - 480..];
        ((mean(tail) - 1.0).abs() < 0.01 && peak(&x) <= 1.001).then_some(())
    });

    // Step: a feedback node wired to itself reads its own last block through a copy, never the
    // region it writes — a wire the graph accepts must not tear the audio thread.
    let fb2 = g.add("Feedback");
    g.link(fb2, "out", fb2, "input");
    g.link(fb2, "out", out, "input");
    let (still, _) = drive(&g, TENTH);
    assert!((mean(&still[still.len() - 480..]) - 1.0).abs() < 0.01, "it holds its zero: {}", mean(&still[still.len() - 480..]));
    g.call("node remove", j!({ "node": hex(fb2) }));
    g.call("node remove", j!({ "node": hex(fb) }));
    let (opened, _) = drive(&g, TENTH);
    assert!((mean(&opened[opened.len() - 480..]) - 0.5).abs() < 0.01, "the loop opened: {}", mean(&opened[opened.len() - 480..]));

    // Step: an authored audio node builds, loads behind the boundary and sounds beside the chain.
    let dir = g.state.mount().join("nodes_audio");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("Trap.rs"),
        "use goofi_audio_sdk::goofi_core::SlotType;\n\
         use goofi_audio_sdk::{AudioNode, Block, Manifest, OutputDecl, ParamDecl, ParamSpec};\n\
         goofi_audio_sdk::params! {\n    \
         ARM = ParamDecl { group: \"trap\", name: \"arm\", spec: ParamSpec::Bool { default: false }, expression: None, doc: None },\n    \
         STALL = ParamDecl { group: \"trap\", name: \"stall\", spec: ParamSpec::Bool { default: false }, expression: None, doc: None },\n    \
         POISON = ParamDecl { group: \"trap\", name: \"poison\", spec: ParamSpec::Bool { default: false }, expression: None, doc: None },\n\
         }\n\
         static OUTS: &[OutputDecl] = &[OutputDecl { name: \"out\", kind: SlotType::Audio }];\n\
         static MANIFEST: Manifest = Manifest { tags: &[], doc: \"a quarter, a panic, or a stall\", inputs: &[], outputs: OUTS, params: PARAMS };\n\
         #[derive(Default)]\nstruct Trap;\n\
         impl AudioNode for Trap {\n    \
         fn prepare(&mut self, _rate: f64) {}\n    \
         fn process(&mut self, b: &mut Block<'_>) {\n        \
         if b.params[P::ARM].chan(0)[0] > 0.5 { panic!(\"armed\"); }\n        \
         if b.params[P::STALL].chan(0)[0] > 0.5 {\n            \
         let t = std::time::Instant::now();\n            \
         while t.elapsed() < std::time::Duration::from_millis(3) {}\n        \
         }\n        \
         if b.params[P::POISON].chan(0)[0] > 0.5 { b.outs[0].chan_mut(0).fill(f32::NAN); return; }\n        \
         b.outs[0].chan_mut(0).fill(0.25);\n    \
         }\n}\n\
         goofi_audio_sdk::export!(Trap, MANIFEST);\n",
    )
    .unwrap();
    assert_eq!(g.call("library refresh", j!({}))["added"], j!(["audio:Trap"]));
    let trap = g.add("Trap");
    g.link(trap, "out", out, "input");
    let level = |x: &[f32]| mean(&x[x.len() - 480..]);
    sounds(&g, "the quarter to join the half", |x| (level(x) - 0.75).abs() < 0.01);

    // Step: a panic in `process` is caught at the boundary, never the process's end: the node
    // faults with its own words, leaves the plan so the chain reads silence where it was, and a
    // restart brings it back.
    g.set_param(trap, "trap", "arm", true);
    g.until("the panic to be named", |g| {
        drive(g, TENTH);
        state(g, trap).contains("panic: armed").then_some(())
    });
    sounds(&g, "the quarter to leave the sum", |x| (level(x) - 0.5).abs() < 0.01);
    g.set_param(trap, "trap", "arm", false);
    g.call("node restart", j!({ "node": hex(trap) }));
    g.until("the restarted node to be clean", |g| (!state(g, trap).contains("panic")).then_some(()));
    sounds(&g, "the quarter to rejoin", |x| (level(x) - 0.75).abs() < 0.01);

    // Step: a node that takes longer than a block, eight blocks in a row, is taken out of the
    // plan by the watchdog and says so; its neighbours never miss a block for it.
    g.set_param(trap, "trap", "stall", true);
    g.until("the watchdog to take it", |g| {
        drive(g, TENTH);
        state(g, trap).contains("overran the block").then_some(())
    });
    sounds(&g, "the stalled quarter to leave the sum", |x| (level(x) - 0.5).abs() < 0.01);
    g.set_param(trap, "trap", "stall", false);
    g.call("node restart", j!({ "node": hex(trap) }));
    sounds(&g, "the quarter to rejoin once more", |x| (level(x) - 0.75).abs() < 0.01);

    // Step: a block that is not a number is the same fault — the node is named, the chain reads
    // silence where it was, and no neighbour ever sees the NaN.
    g.set_param(trap, "trap", "poison", true);
    g.until("the NaN to be named", |g| {
        drive(g, TENTH);
        state(g, trap).contains("not a number").then_some(())
    });
    sounds(&g, "the poisoned quarter to leave the sum", |x| x.iter().all(|v| v.is_finite()) && (level(x) - 0.5).abs() < 0.01);
    g.set_param(trap, "trap", "poison", false);
    g.call("node restart", j!({ "node": hex(trap) }));
    sounds(&g, "the quarter to rejoin a third time", |x| (level(x) - 0.75).abs() < 0.01);

    // Step: a panic anywhere else in the contract is the same fault at the first block — a
    // constructor that panics, and a `prepare` that does, each named for where it happened.
    let stillborn = "use goofi_audio_sdk::{AudioNode, Block, Manifest};\n\
         static MANIFEST: Manifest = Manifest { tags: &[], doc: \"never born\", inputs: &[], outputs: &[], params: &[] };\n\
         struct Stillborn;\n\
         impl Default for Stillborn { fn default() -> Stillborn { panic!(\"no birth\") } }\n\
         impl AudioNode for Stillborn {\n    \
         fn prepare(&mut self, _rate: f64) {}\n    \
         fn process(&mut self, _b: &mut Block<'_>) {}\n}\n\
         goofi_audio_sdk::export!(Stillborn, MANIFEST);\n";
    let unready = "use goofi_audio_sdk::{AudioNode, Block, Manifest};\n\
         static MANIFEST: Manifest = Manifest { tags: &[], doc: \"never ready\", inputs: &[], outputs: &[], params: &[] };\n\
         #[derive(Default)]\nstruct Unready;\n\
         impl AudioNode for Unready {\n    \
         fn prepare(&mut self, _rate: f64) { panic!(\"not at this rate\") }\n    \
         fn process(&mut self, _b: &mut Block<'_>) {}\n}\n\
         goofi_audio_sdk::export!(Unready, MANIFEST);\n";
    std::fs::write(dir.join("Stillborn.rs"), stillborn).unwrap();
    std::fs::write(dir.join("Unready.rs"), unready).unwrap();
    assert_eq!(g.call("library refresh", j!({}))["added"], j!(["audio:Stillborn", "audio:Unready"]));
    for (type_name, said) in [("Stillborn", "panic: the constructor panicked"), ("Unready", "panic: prepare: not at this rate")] {
        let uid = g.add(type_name);
        g.until(&format!("{type_name} to fault"), |g| {
            drive(g, TENTH);
            state(g, uid).contains(said).then_some(())
        });
    }

    // Step: what a node keeps beyond its params rides its blob in the workspace — past a
    // restart, into an archive and back over the live patch, and through a delete and its undo.
    // Exact, not near: the node puts out the count it was born with.
    std::fs::write(
        dir.join("Ticks.rs"),
        "use goofi_audio_sdk::goofi_core::SlotType;\n\
         use goofi_audio_sdk::{AudioNode, Block, Manifest, OutputDecl};\n\
         static OUTS: &[OutputDecl] = &[OutputDecl { name: \"out\", kind: SlotType::Audio }];\n\
         static MANIFEST: Manifest = Manifest { tags: &[], doc: \"counts its blocks, and says the count it was born with\", inputs: &[], outputs: OUTS, params: &[] };\n\
         #[derive(Default)]\nstruct Ticks { born: u32, count: u32 }\n\
         impl AudioNode for Ticks {\n    \
         fn prepare(&mut self, _rate: f64) {}\n    \
         fn process(&mut self, b: &mut Block<'_>) {\n        \
         b.outs[0].chan_mut(0).fill(self.born as f32);\n        \
         self.count += 1;\n    \
         }\n    \
         fn save(&self) -> Vec<u8> { self.count.to_le_bytes().to_vec() }\n    \
         fn load(&mut self, bytes: &[u8]) {\n        \
         if let Ok(b) = bytes.try_into() { self.count = u32::from_le_bytes(b); self.born = self.count; }\n    \
         }\n}\n\
         goofi_audio_sdk::export!(Ticks, MANIFEST);\n",
    )
    .unwrap();
    assert_eq!(g.call("library refresh", j!({}))["added"], j!(["audio:Ticks"]));
    let ticks = g.add("Ticks");
    // Any frame after a tenth says it. NOT the first sample of a running count: a control tick
    // landing mid-drive splits the tenth into two frames, and the one-deep probe keeps the LAST.
    let born_at = |g: &Goofi, uid: Uid, what: &str| -> f32 {
        let probe = g.probe(uid, "out");
        drive(g, TENTH);
        g.until(what, |_| probe.latest().and_then(|d| f32s(&d).first().copied()))
    };
    let blocks = (TENTH / goofi_audio_sdk::BLOCK) as f32;
    drive(&g, 2 * TENTH);
    g.call("node restart", j!({ "node": hex(ticks) }));
    assert_eq!(born_at(&g, ticks, "the count after a restart"), 2.0 * blocks);

    // Saved at three tenths, driven on to five, then loaded OVER itself: the outgoing node's
    // count must retire into the mount being replaced, not over the archive's.
    let keep = tempfile::tempdir().unwrap();
    let target = keep.path().join("state.gfi");
    g.call("session save", j!({ "path": target.to_string_lossy() }));
    drive(&g, 2 * TENTH);
    g.call("session load", j!({ "path": target.to_string_lossy() }));
    assert_eq!(born_at(&g, ticks, "the count after a load"), 3.0 * blocks);

    g.call("node remove", j!({ "node": hex(ticks) }));
    g.call("undo", j!({}));
    assert_eq!(born_at(&g, ticks, "the count after an undo"), 4.0 * blocks);

    // A delete leaves its blob for its undo, and a save packs it; the load that ends that undo
    // sweeps it, so the number minted again — or chosen — is born on nothing.
    g.call("node remove", j!({ "node": hex(ticks) }));
    g.call("session save", j!({ "path": target.to_string_lossy() }));
    let opened = Goofi::new();
    opened.call("session load", j!({ "path": target.to_string_lossy() }));
    opened.call("node add", j!({ "type": "Ticks", "member_uid": hex(ticks) }));
    assert_eq!(born_at(&opened, ticks, "the count of a new node at a dead node's uid"), 0.0);

    // Step: a `.vst3` bundle in the workspace is a node — scanned in a child process, its buses
    // ports, its parameters params, and its event input three voice params — and a bundle
    // whose scanner dies is greyed with the reason while the server answers on.
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures").join("vst3");
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let nested = PathBuf::from(std::env::var_os("CARGO_TARGET_DIR").expect("the harness names the nested target"));
    let built = |crash: bool| -> PathBuf {
        let mut cmd = std::process::Command::new(&cargo);
        cmd.arg("build").arg("--manifest-path").arg(fixture.join("Cargo.toml"));
        if crash {
            cmd.arg("--features").arg("crash");
        }
        let out = cmd.output().expect("cargo runs");
        assert!(out.status.success(), "the fixture plugin builds: {}", String::from_utf8_lossy(&out.stderr));
        let (prefix, suffix) = (std::env::consts::DLL_PREFIX, std::env::consts::DLL_SUFFIX);
        nested.join("debug").join(format!("{prefix}goofi_vst3_fixture{suffix}"))
    };
    // The load above swapped the mount, so the live workspace is asked for again.
    let dir = g.state.mount().join("nodes_audio");
    let bundled = |name: &str, artifact: PathBuf| {
        let arch = std::env::consts::ARCH;
        let (folder, file) = if cfg!(target_os = "macos") {
            ("MacOS".to_string(), name.to_string())
        } else if cfg!(windows) {
            (format!("{arch}-win"), format!("{name}.vst3"))
        } else {
            (format!("{arch}-linux"), format!("{name}.so"))
        };
        let into = dir.join(format!("{name}.vst3")).join("Contents").join(folder);
        std::fs::create_dir_all(&into).unwrap();
        std::fs::copy(&artifact, into.join(file)).unwrap();
    };
    bundled("GoofiFixture", built(false));
    // Two audio classes in the one bundle: an effect, and a synth whose subcategories say so.
    assert_eq!(g.call("library refresh", j!({}))["added"], j!(["audio:GoofiFixture", "audio:GoofiSynth"]));
    let plug = g.add("GoofiFixture");
    let params = g.doc()["nodes"][hex(plug)]["params"].clone();
    for (group, name) in [("voice", "gate"), ("voice", "pitch"), ("voice", "velocity"), ("plugin", "gain")] {
        assert!(!params[group][name].is_null(), "the derived manifest carries {group}/{name}: {params}");
    }
    // Each parameter shape by its own rule: stepped within the ceiling is a list of the plugin's
    // own words, stepped past it a number, and read-only is not a param at all.
    let row = |ty: &str| g.call("library list", j!({}))["types"].as_array().unwrap().iter()
        .find(|v| v["type"] == ty).cloned().unwrap_or_else(|| panic!("{ty} is in the palette"));
    let plugin = row("audio:GoofiFixture");
    // A plugin declares no tag: its VST3 subcategories place it, and the vendor rides the doc line.
    assert_eq!(plugin["tags"], j!(["transform"]), "an effect, not an instrument: {plugin}");
    assert_eq!(plugin["doc"], j!("goofi: GoofiFixture"), "{plugin}");
    let synth = row("audio:GoofiSynth");
    assert_eq!(synth["tags"], j!(["generator"]), "`Instrument` in its subcategories: {synth}");
    let schema = plugin["params"].clone();
    assert_eq!(schema["plugin"]["shape"]["options"], j!(["soft", "mid", "hard"]), "{schema}");
    assert_eq!(schema["plugin"]["steps"]["vmax"], j!(200), "{schema}");
    assert!(schema["plugin"]["meter"].is_null(), "a read-only parameter is omitted: {schema}");
    let src = g.add("Osc");
    g.link(src, "out", plug, "input");
    heard(&g, plug, "the oscillator through the plugin at unit gain", |x| (peak(x) - 1.0).abs() < 0.02 && near(per_tenth(x), 88));
    g.set_param(plug, "plugin", "gain", 0.5);
    heard(&g, plug, "half gain", |x| (peak(x) - 0.5).abs() < 0.02);
    g.call("link remove", j!({ "from": ep(hex(src), "out"), "to": ep(hex(plug), "input") }));
    g.set_param(plug, "voice", "gate", true);
    heard(&g, plug, "a C4 from a rising gate at pitch zero", |x| (peak(x) - 0.5).abs() < 0.02 && near(per_tenth(x), 52));
    g.set_param(plug, "voice", "gate", false);
    heard(&g, plug, "silence after the fall", |x| peak(x) < 1e-3);
    g.call("node restart", j!({ "node": hex(plug) }));
    g.link(src, "out", plug, "input");
    heard(&g, plug, "the gain the record keeps, past a restart", |x| (peak(x) - 0.5).abs() < 0.02);

    bundled("Crasher", built(true));
    assert_eq!(g.call("library refresh", j!({}))["added"], j!(["audio:Crasher"]));
    let row = g.call("library list", j!({}))["types"].as_array().unwrap().iter()
        .find(|v| v["type"] == "audio:Crasher").cloned().expect("greyed, not absent");
    assert_eq!(row["available"], false, "{row}");
    assert!(row["missing_deps"].to_string().contains("scanner"), "the scanner's death is the reason: {row}");
    assert!(g.refuse("node add", j!({ "type": "Crasher" })).contains("unavailable"));
    heard(&g, plug, "the server answers on", |x| (peak(x) - 0.5).abs() < 0.02);

    // The plugin's state is its own blob, and the record is its params: the fixture latches the
    // first time `shape` reaches its last step and halves its tone for ever after, which no param
    // can say. Latched, then set back, saved and reopened: `shape` is the record's `soft` and the
    // tone is still the latched one, so the blob came back and the record went on top of it.
    g.call("link remove", j!({ "from": ep(hex(src), "out"), "to": ep(hex(plug), "input") }));
    g.set_param(plug, "voice", "gate", true);
    heard(&g, plug, "the tone before the latch", |x| (peak(x) - 0.5).abs() < 0.02);
    g.set_param(plug, "plugin", "shape", "hard");
    heard(&g, plug, "the latched tone", |x| (peak(x) - 0.25).abs() < 0.02);
    g.set_param(plug, "plugin", "shape", "soft");
    g.call("session save", j!({ "path": target.to_string_lossy() }));
    let carried = Goofi::new();
    carried.call("session load", j!({ "path": target.to_string_lossy() }));
    assert_eq!(carried.doc()["nodes"][hex(plug)]["params"]["plugin"]["shape"]["value"], j!("soft"));
    heard(&carried, plug, "the archive carried the blob", |x| (peak(x) - 0.25).abs() < 0.02 && near(per_tenth(x), 52));
    drop(carried);

    // Step: a plugin's own editor is a window on the machine goofi runs on, off the LIVE
    // instance's controller, and a knob turned in it reaches the record and the audio through
    // the param door. The palette's word decides once: where no display answers, no editor.
    g.set_param(plug, "voice", "gate", false);
    g.link(src, "out", plug, "input");
    heard(&g, plug, "half gain, before the window", |x| (peak(x) - 0.5).abs() < 0.02);
    let editor = g.call("library get", j!({ "type": "audio:GoofiFixture" }))["editor"] == j!(true);
    let gain = |g: &Goofi| g.doc()["nodes"][hex(plug)]["params"]["plugin"]["gain"]["value"].clone();
    if editor {
        assert_eq!(g.call("node editor", j!({ "node": hex(plug) }))["changed"], true);
        g.until("the knob the editor turned to reach the record", |g| (gain(g) == j!(0.25)).then_some(()));
        heard(&g, plug, "the quarter gain the editor asked for", |x| (peak(x) - 0.25).abs() < 0.02);
        assert_eq!(g.call("node editor", j!({ "node": hex(plug) }))["changed"], false, "already open");
        assert_eq!(g.call("node editor", j!({ "node": hex(plug), "show": false }))["changed"], true);
        assert_eq!(g.call("node editor", j!({ "node": hex(plug), "show": false }))["changed"], false, "already closed");
        assert_eq!(g.call("node editor", j!({ "node": hex(plug) }))["changed"], true, "opens again");
        g.call("node remove", j!({ "node": hex(plug) }));
        assert!(g.refuse("node editor", j!({ "node": hex(plug) })).contains("no such node"), "the window went with the node");
    } else {
        assert_eq!(gain(&g), j!(0.5));
        assert!(g.refuse("node editor", j!({ "node": hex(plug) })).contains("has no editor"));
    }
}
