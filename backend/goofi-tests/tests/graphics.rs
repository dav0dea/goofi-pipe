//! The graphics engine under the external clock: one session, every action through the op
//! vocabulary, every probe a subscriber on the derived name of a texture slot — the door `/data`
//! opens — and every frame a readback the GPU actually made.

use goofi_tests::{ep, f32s, hex, j, render, shape, Goofi, Uid};

const NO_GPU: &str = "no graphics engine here. The suite needs a GPU adapter: install a Vulkan \
                      driver, or Mesa's lavapipe (`mesa-vulkan-drivers`)";

/// Tick until the probe on `uid`'s output holds a frame `want` accepts, and hand it back.
fn drawn(g: &Goofi, uid: Uid, what: &str, want: impl Fn(&goofi_core::Data) -> bool) -> goofi_core::Data {
    let probe = g.probe(uid, "out");
    g.until(what, |g| {
        render(g, 1);
        probe.latest().filter(&want)
    })
}

/// The texel at `(row, col)`: four floats, row 0 the top.
fn px(d: &goofi_core::Data, row: usize, col: usize) -> [f32; 4] {
    let s = shape(d);
    let at = (row * s[1] + col) * 4;
    f32s(d)[at..at + 4].try_into().expect("four channels")
}

fn close(a: [f32; 4], b: [f32; 4]) -> bool {
    a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-3)
}

#[test]
fn shaders_render_on_the_gpu() {
    let g = Goofi::new();

    // Step: the engine registered, and its library is in the ONE palette beside the others.
    let types = g.call("library list", j!({ "full": true }));
    let names: Vec<&str> = types["types"].as_array().unwrap().iter().filter_map(|r| r["type"].as_str()).collect();
    assert!(names.contains(&"graphics:Constant"), "{NO_GPU}\n{names:?}");
    let row = types["types"].as_array().unwrap().iter().find(|r| r["type"] == "graphics:Constant").unwrap();
    assert_eq!(row["output_slots"]["out"], "TEXTURE", "{row}");
    assert_eq!(g.call("library get", j!({ "type": "graphics:Constant" }))["tier"], "shader");
    let status = g.call("session status", j!({}))["graphics"].clone();
    assert_eq!(status["clock"], "external", "{status}");
    assert!(status["adapter"].as_str().is_some_and(|a| !a.is_empty()), "{status}");

    // Step: a Constant reads back the colour it was given, at the generator's own size.
    let c = g.add("graphics:Constant");
    g.ready(c);
    g.set_param(c, "colour", "r", 0.25);
    g.set_param(c, "colour", "g", 0.5);
    g.set_param(c, "colour", "b", 1.0);
    let frame = drawn(&g, c, "the constant's colour", |d| close(px(d, 0, 0), [0.25, 0.5, 1.0, 1.0]));
    assert_eq!(shape(&frame), vec![512, 512, 4], "a node with nothing behind it is 512 square");
    assert!(close(px(&frame, 511, 511), [0.25, 0.5, 1.0, 1.0]), "the same colour to the far corner");

    // Step: the universal `output` group resizes it, and what is wired behind FOLLOWS the size.
    g.set_param(c, "output", "width", 64);
    g.set_param(c, "output", "height", 32);
    let frame = drawn(&g, c, "the resized frame", |d| shape(d) == vec![32, 64, 4]);
    assert!(close(px(&frame, 31, 63), [0.25, 0.5, 1.0, 1.0]));
    let level = g.add("graphics:Level");
    g.ready(level);
    g.link(c, "out", level, "input");
    let frame = drawn(&g, level, "the level follows its input's size", |d| shape(d) == vec![32, 64, 4]);
    assert!(close(px(&frame, 0, 0), [0.25, 0.5, 1.0, 1.0]), "gain 1 is a copy");

    // Step: a chain — the format is HDR, so doubling a value past 1 keeps what it made.
    g.set_param(level, "level", "gain", 2.0);
    drawn(&g, level, "the doubled frame", |d| close(px(d, 5, 5), [0.5, 1.0, 2.0, 1.0]));
    g.set_param(level, "level", "gain", 1.0);

    // Step: an unwired texture input is transparent black — present, never an error.
    g.call("link remove", j!({ "from": ep(hex(c), "out"), "to": ep(hex(level), "input") }));
    let frame = drawn(&g, level, "the unwired level", |d| close(px(d, 0, 0), [0.0, 0.0, 0.0, 0.0]));
    assert_eq!(shape(&frame), vec![512, 512, 4], "with nothing to follow, it is a generator's size");
    assert!(g.error(level).is_none(), "an unwired input is not a fault");

    // Step: a signal frame uploads — the fixture's gradient, sampled back texel for texel and the
    // right way up. The one place the two row orders could disagree.
    let img = g.add("_TestImage");
    g.ready(img);
    let up = g.add("graphics:ArrayIn");
    g.ready(up);
    g.set_param(up, "output", "width", 4);
    g.set_param(up, "output", "height", 4);
    g.link(img, "out", up, "input");
    let frame = drawn(&g, up, "the uploaded image", |d| shape(d) == vec![4, 4, 4]);
    assert!(close(px(&frame, 0, 0), [0.0, 1.0, 0.5, 1.0]), "row 0 is the top: {:?}", px(&frame, 0, 0));
    assert!(close(px(&frame, 3, 3), [1.0, 0.0, 0.5, 1.0]), "and row 3 the bottom: {:?}", px(&frame, 3, 3));

    g.call("link remove", j!({ "from": ep(hex(img), "out"), "to": ep(hex(up), "input") }));
    drawn(&g, up, "the unlinked upload", |d| close(px(d, 0, 0), [0.0, 0.0, 0.0, 0.0]));

    // Step: a texture chain does not flip either. A gradient down the frame, copied by a Level,
    // still runs the same way — the half of the orientation rule an upload cannot see.
    let vert = g.add("graphics:Ramp");
    g.ready(vert);
    g.set_param(vert, "ramp", "angle", 90.0);
    g.set_param(vert, "output", "width", 8);
    g.set_param(vert, "output", "height", 8);
    let copy = g.add("graphics:Level");
    g.ready(copy);
    g.link(vert, "out", copy, "input");
    let direct = drawn(&g, vert, "the vertical ramp", |d| shape(d) == vec![8, 8, 4]);
    let copied = drawn(&g, copy, "the copy of it", |d| shape(d) == vec![8, 8, 4]);
    assert!(px(&direct, 0, 0)[0] < px(&direct, 7, 0)[0], "the ramp runs down: {:?}", px(&direct, 0, 0));
    assert!(close(px(&copied, 0, 0), px(&direct, 0, 0)), "and the copy is not mirrored");
    assert!(close(px(&copied, 7, 0), px(&direct, 7, 0)));

    // Step: a reference moves a param at control rate, the one door every modulation uses.
    let knob = g.add("_TestScalar");
    g.ready(knob);
    g.set_param(knob, "control", "value", 0.5);
    let knob_name = g.name(&hex(knob));
    g.link(c, "out", level, "input");
    g.call("node param edit", j!({ "node": hex(level), "param": "level/gain",
                                  "reference": format!("{knob_name}.out"), "mode": "reference" }));
    drawn(&g, level, "half gain by reference", |d| close(px(d, 0, 0), [0.125, 0.25, 0.5, 1.0]));
    g.set_param(knob, "control", "value", 2.0);
    drawn(&g, level, "double gain by reference", |d| close(px(d, 0, 0), [0.5, 1.0, 2.0, 1.0]));

    // Step: a loop closes through Feedback and accumulates a tenth a tick; one without it faults.
    let fb = g.add("graphics:Feedback");
    g.ready(fb);
    let acc = g.add("graphics:Level");
    g.ready(acc);
    g.set_param(acc, "level", "offset", 0.1);
    g.link(fb, "out", acc, "input");
    g.link(acc, "out", fb, "input");
    let first = drawn(&g, acc, "the first tick", |d| px(d, 0, 0)[0] > 0.05);
    let after = drawn(&g, acc, "five ticks on", |d| px(d, 0, 0)[0] > px(&first, 0, 0)[0] + 0.4);
    assert!(px(&after, 0, 0)[0] < 10.0, "a tenth a tick, not a runaway: {:?}", px(&after, 0, 0));
    assert!(g.error(fb).is_none() && g.error(acc).is_none(), "a loop through Feedback is not a fault");
    let lone = g.add("graphics:Level");
    g.ready(lone);
    g.link(lone, "out", lone, "input");
    g.until("a loop with no feedback node faults", |g| {
        render(g, 1);
        g.error(lone).filter(|e| e.contains("feedback"))
    });
    g.call("node remove", j!({ "node": hex(lone) }));

    // Step: the shipped set composes, and every one of it compiles on this machine.
    let ramp = g.add("graphics:Ramp");
    g.ready(ramp);
    let comp = g.add("graphics:Composite");
    g.ready(comp);
    g.set_param(comp, "composite", "mode", "add");
    g.link(ramp, "out", comp, "a");
    g.link(c, "out", comp, "b");
    drawn(&g, comp, "the sum of a ramp and the constant", |d| px(d, 0, 0)[2] > 1.0 - 1e-3);
    // A row says naga read the file; a FRAME says this device built the pipeline behind it, which
    // is the half a validation pass cannot answer for.
    let shipped: Vec<String> = g.call("library list", j!({}))["types"]
        .as_array()
        .expect("a palette")
        .iter()
        .filter_map(|r| r["type"].as_str())
        .filter(|t| t.starts_with("graphics:"))
        .map(String::from)
        .collect();
    assert_eq!(shipped.len(), 13, "the shipped set: {shipped:?}");
    for ty in &shipped {
        let node = g.add(ty);
        g.ready(node);
        drawn(&g, node, ty, |d| shape(d).len() == 3);
        assert!(g.error(node).is_none(), "{ty} stands with an error");
        g.call("node remove", j!({ "node": hex(node) }));
    }

    // Step: a `.wgsl` that does not compile is a greyed type carrying naga's own line number.
    let dir = g.state.mount().join("nodes_graphics");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("Broken.wgsl"), BROKEN).unwrap();
    g.call("library refresh", j!({}));
    let listed = g.call("library list", j!({ "full": true }));
    let greyed = listed["types"].as_array().unwrap().iter().find(|r| r["type"] == "graphics:Broken").cloned();
    let greyed = greyed.expect("a file that does not compile is still a row");
    assert_eq!(greyed["available"], false, "{greyed}");
    let why = greyed["doc"].as_str().unwrap_or_default();
    assert!(why.contains(":4:"), "the file's OWN line 4, not the prelude's: {why}");
    assert!(g.refuse("node add", j!({ "type": "graphics:Broken" })).contains("unavailable"));

    // Step: a workspace node is authored, loaded, and reloaded through the one refresh door.
    std::fs::write(dir.join("Half.wgsl"), HALF).unwrap();
    assert_eq!(g.call("library refresh", j!({}))["added"], j!(["graphics:Half"]));
    let half = g.add("graphics:Half");
    g.ready(half);
    g.link(c, "out", half, "input");
    drawn(&g, half, "half of the constant", |d| close(px(d, 0, 0), [0.125, 0.25, 0.5, 1.0]));
    std::fs::write(dir.join("Half.wgsl"), QUARTER).unwrap();
    assert_eq!(g.call("library refresh", j!({}))["changed"], j!(["graphics:Half"]));
    drawn(&g, half, "a quarter, after the reload", |d| close(px(d, 0, 0), [0.0625, 0.125, 0.25, 1.0]));

    // Step: a node nobody reads renders nothing — which is what makes an idle patch free.
    let stages = |g: &Goofi| g.call("session status", j!({}))["graphics"]["stages"].as_u64().unwrap();
    let lonely = g.add("graphics:Constant");
    g.ready(lonely);
    render(&g, 5);
    let before = stages(&g);
    render(&g, 10);
    assert_eq!(stages(&g), before, "ten ticks, no reader, no work");
    let probe = g.probe(lonely, "out");
    g.until("a reader wakes it", |g| {
        render(g, 1);
        probe.latest()
    });
    // EXACTLY one stage a tick, with several nodes live: "it moved" would pass a demand walk that
    // wakes the whole patch whenever anybody reads anything.
    let watched = stages(&g);
    render(&g, 3);
    assert_eq!(stages(&g), watched + 3, "one reader, one stage a tick, whatever else is live");

    // Step: a restart is a rebirth through the same trait doors — new generation, new services.
    let generation = g.state.graph.lock().unwrap().node_generation(c);
    let stale = g.probe(c, "out");
    g.call("node restart", j!({ "node": hex(c) }));
    g.ready(c);
    assert_eq!(g.state.graph.lock().unwrap().node_generation(c), generation + 1);
    drawn(&g, c, "the reborn constant", |d| close(px(d, 0, 0), [0.25, 0.5, 1.0, 1.0]));
    let seen = stale.count();
    render(&g, 5);
    assert_eq!(stale.count(), seen, "the corpse's service name went silent");

    // Step: a remove through the one op surface tears the node down and the rest stand.
    g.call("node remove", j!({ "node": hex(lonely) }));
    assert!(!g.nodes().contains(&hex(lonely)));
    drawn(&g, c, "the constant still renders", |d| close(px(d, 0, 0), [0.25, 0.5, 1.0, 1.0]));
}

/// The clock the binary actually runs on: nobody calls `render()`, and the engine draws anyway.
#[test]
fn the_engine_draws_on_its_own_clock() {
    let g = Goofi::timed();
    let c = g.add("graphics:Constant");
    g.ready(c);
    g.set_param(c, "colour", "r", 0.75);

    // Step: with no reader the clock turns and nothing is drawn — the demand rule holds here too.
    let idle = |g: &Goofi| g.call("session status", j!({}))["graphics"].clone();
    let start = g.until("the clock turns", |g| {
        idle(g)["frames"].as_u64().filter(|f| *f > 3)
    });
    assert_eq!(idle(&g)["clock"], "timer");
    assert_eq!(idle(&g)["stages"], j!(0), "no reader, no stage, whatever the clock does");

    // Step: a viewer arrives and the frames it gets were drawn by nobody's hand.
    let probe = g.probe(c, "out");
    let frame = g.until("a frame off the timer", |_| probe.latest());
    assert!(close(px(&frame, 0, 0), [0.75, 1.0, 1.0, 1.0]), "{:?}", px(&frame, 0, 0));
    let ran = idle(&g);
    assert!(ran["frames"].as_u64().is_some_and(|f| f > start), "{ran}");
    assert!(ran["stages"].as_u64().is_some_and(|s| s > 0), "{ran}");
}

const BROKEN: &str = "/* goofi\n{ \"doc\": \"does not compile\" }\n*/\nfn shade(uv: vec2f) -> vec4f { return nothing(uv); }\n";
const HALF: &str = "/* goofi\n{ \"doc\": \"half of the input\", \"inputs\": [{\"name\": \"input\", \"kind\": \"TEXTURE\"}] }\n*/\nfn shade(uv: vec2f) -> vec4f { let c = textureSample(input, samp, uv); return vec4f(c.rgb * 0.5, c.a); }\n";
const QUARTER: &str = "/* goofi\n{ \"doc\": \"a quarter of the input\", \"inputs\": [{\"name\": \"input\", \"kind\": \"TEXTURE\"}] }\n*/\nfn shade(uv: vec2f) -> vec4f { let c = textureSample(input, samp, uv); return vec4f(c.rgb * 0.25, c.a); }\n";
