//! The audio engine under the external clock: one session, every action through the op
//! vocabulary, every probe the interleaved output a device would receive.

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
    let osc2 = frozen_square(&g);
    g.link(osc2, "out", gain, "input");
    g.call("node param edit", j!({ "node": hex(gain), "param": "gain/gain", "value": 0.5, "mode": "constant" }));
    sounds(&g, "sine plus one, halved, on a half offset", |x| (peak(x) - 1.0).abs() < 0.01 && (mean(x) - 0.5).abs() < 0.02);

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
    let stillborn: Vec<String> = many.iter().filter_map(|u| g.error(*u)).collect();
    assert!(stillborn.is_empty(), "every square starts: {stillborn:?}");
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

    // Step: a device or a port that is not there is an error on the param that named it, and
    // what can be named is a refresh; a MIDI port's voices are the channels a gate sees.
    g.set_param(gain3, "gain", "gain", 0.0);
    sounds(&g, "the chain to fall silent", |x| peak(x) == 0.0);
    let mic = g.add("AudioIn");
    g.set_param(mic, "audio", "device", "nowhere");
    let why = g.until("the absent device to be named", |g| g.error(mic).filter(|e| e.contains("nowhere")));
    assert!(why.contains("no input device `nowhere`"), "{why}");
    assert_eq!(refreshed(&g, &mut ev, mic, "audio", "device")[0], "default");
    g.call("node remove", j!({ "node": hex(mic) }));
    let midi = g.add("MidiIn");
    g.set_param(midi, "midi", "port", "nowhere");
    let why = g.until("the absent port to be named", |g| g.error(midi));
    assert!(why.contains("no MIDI port `nowhere`"), "{why}");
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
    // them. A virtual port, which WinMM has none of.
    #[cfg(unix)]
    {
        use midir::os::unix::VirtualOutput;
        let mut keys = midir::MidiOutput::new("goofi-test")
            .expect("a MIDI client: on Linux this needs the ALSA sequencer, `modprobe snd-seq`")
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
    let held = frozen_square(&g);
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
         STALL = ParamDecl { group: \"trap\", name: \"stall\", spec: ParamSpec::Bool { default: false }, expression: None, doc: None },\n\
         }\n\
         static OUTS: &[OutputDecl] = &[OutputDecl { name: \"out\", kind: SlotType::Audio }];\n\
         static MANIFEST: Manifest = Manifest { category: \"test\", doc: \"a quarter, a panic, or a stall\", inputs: &[], outputs: OUTS, params: PARAMS };\n\
         #[derive(Default)]\nstruct Trap;\n\
         impl AudioNode for Trap {\n    \
         fn prepare(&mut self, _rate: f64) {}\n    \
         fn process(&mut self, b: &mut Block<'_>) {\n        \
         if b.params[P::ARM].chan(0)[0] > 0.5 { panic!(\"armed\"); }\n        \
         if b.params[P::STALL].chan(0)[0] > 0.5 {\n            \
         let t = std::time::Instant::now();\n            \
         while t.elapsed() < std::time::Duration::from_millis(3) {}\n        \
         }\n        \
         b.outs[0].chan_mut(0).fill(0.25);\n    \
         }\n}\n\
         goofi_audio_sdk::export!(Trap, MANIFEST);\n",
    )
    .unwrap();
    assert_eq!(g.call("library refresh", j!({}))["added"], j!(["Trap"]));
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
}
