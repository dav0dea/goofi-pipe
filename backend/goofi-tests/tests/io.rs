//! The nodes that leave the process: a patch publishes what it made, another node in the same
//! patch receives it back over the machine's own network, and what arrives is what was sent.
//!
//! Every stream and port is named for THIS process, because a suite run must not find — or feed —
//! another goofi on the same machine.
#![cfg(not(feature = "embed"))]

use goofi_tests::{f32s, hex, j, labels, require_python, shape, text, Goofi};

#[test]
fn a_patch_publishes_what_it_made_and_reads_it_back_off_the_machine() {
    let _py = require_python();
    let g = Goofi::new();
    let mine = std::process::id();

    // LSL: a named grid goes out as a stream, and comes back with the names and the rate on it.
    let src = g.add("_TestGrid");
    let meta = g.add("Meta");
    g.set_param(meta, "meta", "labels", "Fz,Cz,Pz");
    g.set_param(meta, "meta", "axis", 0);
    let out = g.add("LslOut");
    let stream = format!("goofi-loop-{mine}");
    g.set_param(out, "lsl", "name", stream.clone());
    g.link(src, "out", meta, "input");
    g.link(meta, "out", out, "input");

    let inn = g.add("LslIn");
    g.set_param(inn, "lsl", "name", stream.clone());
    let received = g.probe(inn, "out");
    g.ready(inn);
    assert!(g.error(inn).is_none(), "an absent stream is not an error: {:?}", g.error(inn));
    let d = g.until("the stream this patch is publishing", |_| {
        received.latest().filter(|d| shape(d)[0] == 3 && shape(d)[1] > 0)
    });
    assert_eq!(d.meta().sfreq(), Some(256.0), "the rate rides the stream's own description");
    assert_eq!(labels(&d, "dim0"), ["Fz", "Cz", "Pz"], "and so do the channel names");

    // The ⟳ on the stream picker lists what is on the network, this patch's own stream included.
    let mut ev = g.events();
    g.call("node param refresh", j!({ "node": hex(inn), "param": "lsl/name" }));
    let echo = g.until("the picker's echo", |_| {
        let p = ev.next("state_update");
        (p["node"] == hex(inn) && p["refreshed_params"] == j!([["lsl", "name"]])).then_some(p)
    });
    let options = echo["params"]["lsl"]["name"]["options"].as_array().cloned().unwrap_or_default();
    assert!(options.contains(&j!(stream)), "the live stream is offered: {options:?}");
    assert!(options.contains(&j!("")), "…and `any` stays an option: {options:?}");

    // OSC: a table goes out as one message per leaf, and the address is the path back into it.
    let port = 40_000 + (mine % 20_000) as i64;
    let level = g.add("signal:Constant");
    g.set_param(level, "constant", "value", 0.25);
    let name = g.add("Text");
    g.set_param(name, "text", "value", "resting");
    let table = g.add("Table");
    g.set_param(table, "table", "keys", "alpha,state");
    let sender = g.add("OscOut");
    g.set_param(sender, "osc", "port", port);
    g.link(level, "out", table, "arrays");
    g.link(name, "out", table, "strings");
    g.link(table, "out", sender, "input");

    let listener = g.add("OscIn");
    g.set_param(listener, "osc", "port", port);
    let pick = g.add("TableSelect");
    g.set_param(pick, "table", "key", "goofi.alpha");
    let number = g.probe(pick, "array");
    g.link(listener, "out", pick, "input");
    g.until("the number back off the network, under the address it was sent to", |_| {
        number.latest().filter(|d| f32s(d) == vec![0.25])
    });

    // The string took the same journey, on the output whose kind it has.
    g.set_param(pick, "table", "key", "goofi.state");
    let word = g.probe(pick, "string");
    g.until("the string on the string output", |_| {
        word.latest().filter(|d| text(d) == Some("resting"))
    });

    // MIDI: with no port chosen there is nothing to open, which is silence rather than a fault.
    // The engine has to be named here — `audio:MidiIn` is a different node with the same name.
    for ty in ["signal:MidiIn", "signal:MidiOut"] {
        let n = g.add(ty);
        g.ready(n);
        assert!(g.error(n).is_none(), "{ty} with no port is silent, not broken: {:?}", g.error(n));
        g.call("node param refresh", j!({ "node": hex(n), "param": "midi/port" }));
        g.until("the port picker to answer", |_| {
            let p = ev.next("state_update");
            (p["node"] == hex(n) && p["refreshed_params"] == j!([["midi", "port"]])).then_some(())
        });
    }

    for n in [out, inn, sender, listener] {
        assert!(g.error(n).is_none(), "an io node carries no error: {:?}", g.error(n));
    }
}
