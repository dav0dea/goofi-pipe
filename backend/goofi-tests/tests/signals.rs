//! The signal nodes doing signal work: a stream is shaped, filtered, windowed and transformed,
//! and what comes out the far end is measured rather than assumed.
//!
//! DIMENSIONALITY is the property that spans the set, so the source emits a grid, not a vector.

use goofi_core::Data;
use goofi_tests::{f32s, hex, shape, text, j, Goofi};

/// The bin carrying the most power, and its value.
fn peak(d: &Data) -> (usize, f32) {
    let v = f32s(d);
    let k = (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b])).expect("a non-empty spectrum");
    (k, v[k])
}

#[test]
fn a_chain_filters_a_live_stream_and_reads_the_band_that_survives() {
    // 256 Hz of a 10 Hz sine through a band that admits it, windowed to two seconds — so a bin is
    // half a hertz and a spectrum labelled in bins instead of hertz reads WRONG.
    let g = Goofi::new();
    let osc = g.add("LFO");
    let flt = g.add("signal:Filter");
    let buf = g.add("Buffer");
    let psd = g.add("Psd");
    let set = |n, group: &str, name: &str, v: serde_json::Value| {
        g.set_param(n, group, name, v);
    };
    set(osc, "output", "sfreq", j!(256.0));
    set(osc, "output", "mode", j!("block"));
    set(osc, "lfo", "frequency", j!(10.0));
    set(buf, "buffer", "size", j!(512));
    set(flt, "filter", "low", j!(5.0));
    set(flt, "filter", "high", j!(20.0));
    set(psd, "psd", "mode", j!("fft"));

    let probe = g.probe(psd, "out"); // opened before the wires: the data services keep no history
    g.link(osc, "out", flt, "input");
    g.link(flt, "out", buf, "input");
    g.link(buf, "out", psd, "input");

    // 512 real samples become 257 one-sided bins, and the rank is untouched.
    let full = g.until("a spectrum of the full window", |_| {
        probe.latest().filter(|d| shape(d) == vec![257])
    });
    let (bin, power) = peak(&full);
    assert_eq!(bin, 20, "a 10 Hz sine peaks in the 10 Hz bin — bin 20 at half a hertz each, not {bin}");

    // The frequency axis rides the frame as coordinates, so a viewer never re-derives it.
    let freqs = full.meta().channels().get(0).and_then(|a| a.coords.clone()).expect("bin coords");
    assert_eq!(freqs.len(), 257);
    assert_eq!(freqs[20], goofi_core::Coord::Num(10.0), "bin 20 is 10 Hz at sfreq 256 over 512");
    assert_eq!(full.meta().sfreq(), None, "a spectrum is not a time series any more");

    // Welch cuts the same window into one-second segments, so the answer is steadier and the bins
    // are twice as wide — and the sine still lands on the bin that carries 10 Hz.
    set(psd, "psd", "mode", j!("welch"));
    set(psd, "welch", "segment", j!(1.0));
    let welch = g.until("a spectrum of averaged segments", |_| {
        probe.latest().filter(|d| shape(d) == vec![129])
    });
    assert_eq!(peak(&welch).0, 10, "a 10 Hz sine peaks in the 10 Hz bin — bin 10 at one hertz each");

    // The range page cuts the spectrum to a band, and the coordinates come with it.
    set(psd, "range", "low", j!(8.0));
    set(psd, "range", "high", j!(12.0));
    let band = g.until("the spectrum cut to a band", |_| {
        probe.latest().filter(|d| shape(d) == vec![5])
    });
    let coords = band.meta().channels().get(0).and_then(|a| a.coords.clone()).expect("bin coords");
    assert_eq!(coords[0], goofi_core::Coord::Num(8.0), "the band starts where it was told to");
    assert_eq!(coords[4], goofi_core::Coord::Num(12.0), "and ends where it was told to");
    assert_eq!(peak(&band).0, 2, "10 Hz is the middle of an 8 to 12 Hz band");

    // The window is a second long, so this waits for the property rather than for the next frame.
    set(psd, "range", "low", j!(0.0));
    set(psd, "range", "high", j!(0.0));
    set(psd, "psd", "mode", j!("fft"));
    set(flt, "filter", "mode", j!("lowpass"));
    set(flt, "filter", "high", j!(2.0));
    g.until("the peak to collapse once the band excludes it", |_| {
        probe.latest().filter(|d| shape(d) == vec![257] && peak(d).1 < power * 0.01)
    });
}

#[test]
fn a_buffer_keeps_the_rank_it_was_given_and_rolls_the_axis_it_was_told_to() {
    let g = Goofi::new();
    let src = g.add("_TestGrid");
    let time = g.add("Buffer");
    let chans = g.add("Buffer");
    let one = g.add("Buffer");
    let set = |n, name: &str, v: serde_json::Value| {
        g.set_param(n, "buffer", name, v);
    };
    set(time, "size", j!(8));
    set(chans, "size", j!(2));
    set(chans, "axis", j!(-2));
    set(one, "size", j!(1));

    let (pt, pc, po) = (g.probe(time, "out"), g.probe(chans, "out"), g.probe(one, "out"));
    for b in [time, chans, one] {
        g.link(src, "out", b, "input");
    }

    // [3, 4] became [3, 8], not the 24-long vector a flattening buffer would produce.
    let rolled = g.until("a full window on the time axis", |_| {
        pt.latest().filter(|d| shape(d) == vec![3, 8])
    });
    let v = f32s(&rolled);
    for r in 0..3 {
        let row = &v[r * 8..(r + 1) * 8];
        assert!(row.windows(2).all(|w| w[1] > w[0]), "row {r} runs forwards in time: {row:?}");
        // Every row holds the SAME frames, so the gap to row 0 is the row's own offset. The gap
        // BETWEEN entries is not checked: latest-wins one deep legitimately drops frames.
        assert!(
            row.iter().zip(&v[..8]).all(|(x, first)| (x - first - r as f32 * 100.0).abs() < 0.01),
            "row {r} is one channel's own history, not a slice of the flattened lot: {row:?}",
        );
    }

    // The same node told to roll the OTHER axis keeps the last two rows of one frame instead.
    let across = g.until("a window on the channel axis", |_| {
        pc.latest().filter(|d| shape(d) == vec![2, 4])
    });
    let v = f32s(&across);
    assert!((v[4] - v[0] - 100.0).abs() < 0.01, "the two kept rows are adjacent channels: {v:?}");

    // A window of one is the identity on rank: the rolled axis is simply length 1.
    let single = g.until("a window of one", |_| po.latest());
    assert_eq!(shape(&single), vec![3, 1], "size 1 shortens the axis, it does not remove it");
}

#[test]
fn the_generators_answer_on_their_own_and_a_settled_one_answers_when_asked() {
    // The family in one session: two producers that pace themselves, and two sources with no
    // input at all, which nothing can ring — so their own birth and their own edits run them.
    let g = Goofi::new();
    let lfo = g.add("LFO");
    let noise = g.add("signal:Noise");
    let konst = g.add("Constant");
    let words = g.add("Text");
    let set = |n, group: &str, name: &str, v: serde_json::Value| {
        g.set_param(n, group, name, v);
    };

    // A value-mode LFO is one sample per update, which is what a param reference reads.
    let plfo = g.probe(lfo, "out");
    set(lfo, "lfo", "waveform", j!("square"));
    set(lfo, "lfo", "amplitude", j!(2.0));
    let one = g.until("a square at amplitude two", |_| {
        plfo.latest().filter(|d| shape(d) == vec![1] && (f32s(d)[0].abs() - 2.0).abs() < 1e-6)
    });
    assert_eq!(shape(&one), vec![1], "value mode is one sample per update");

    // The same node in block mode is a signal: the samples real time advanced by, at its own rate.
    // Ten updates a second against 256 samples a second: a block holds about twenty-five.
    set(lfo, "common", "max_frequency", j!(10.0));
    set(lfo, "output", "mode", j!("block"));
    set(lfo, "output", "sfreq", j!(256.0));
    let block = g.until("a block of samples at the new rate", |_| {
        plfo.latest().filter(|d| shape(d).len() == 1 && shape(d)[0] > 1 && d.meta().sfreq() == Some(256.0))
    });
    assert!(shape(&block)[0] > 1, "a block holds the samples the clock advanced by");

    // Noise counts its channels on the first axis, and the block adds time after them.
    let pn = g.probe(noise, "out");
    set(noise, "output", "channels", j!(4));
    let vals = g.until("four channels of noise", |_| pn.latest().filter(|d| shape(d) == vec![4]));
    assert!(f32s(&vals).iter().all(|v| v.abs() <= 1.0), "uniform noise stays in range: {:?}", f32s(&vals));
    set(noise, "output", "mode", j!("block"));
    let grid = g.until("a block of noise", |_| {
        pn.latest().filter(|d| shape(d).len() == 2 && d.meta().sfreq() == Some(250.0))
    });
    assert_eq!(shape(&grid)[0], 4, "channels stay on the first axis: {:?}", shape(&grid));

    // A Constant is a source like any other: it emits on its own schedule, so a viewer opened at
    // any moment sees it and a wire made at any moment receives it. The shape it is asked for is
    // the shape that comes out.
    let pk = g.probe(konst, "out");
    set(konst, "constant", "value", j!(3.0));
    set(konst, "constant", "shape", j!("2,3"));
    let filled = g.until("the constant to carry its edit", |_| {
        pk.latest().filter(|d| shape(d) == vec![2, 3] && f32s(d).iter().all(|v| *v == 3.0))
    });
    assert_eq!(f32s(&filled).len(), 6, "the shape it was asked for is the shape it filled");

    // And a wire made long afterwards receives it, with nothing to re-plan or replay.
    let buf = g.add("Buffer");
    let pb = g.probe(buf, "out");
    g.link(konst, "out", buf, "input");
    let rolled = g.until("the wire to receive the constant", |_| {
        pb.latest().filter(|d| !f32s(d).is_empty() && f32s(d).iter().all(|v| *v == 3.0))
    });
    assert!(!f32s(&rolled).is_empty(), "a wire made after the one emit still gets a frame");

    // Text is the same rule on the other slot kind.
    let pt = g.probe(words, "out");
    set(words, "text", "value", j!("hello"));
    let said = g.until("the text to answer its edit", |_| pt.latest().filter(|d| text(d) == Some("hello")));
    assert_eq!(text(&said), Some("hello"));
    for n in [lfo, noise, konst, words] {
        assert!(g.error(n).is_none(), "a generator carries no error");
    }
}

#[test]
fn the_array_nodes_reshape_a_grid_and_the_rate_follows_the_time_axis() {
    // One session over the array family. The source is a [3, 4] grid at 256 Hz, so a node that
    // flattens a rank, or keeps a rate whose axis it just removed, is caught here.
    let g = Goofi::new();
    let src = g.add("_TestGrid");
    let set = |n, group: &str, name: &str, v: serde_json::Value| {
        g.set_param(n, group, name, v);
    };

    // Math and Function are elementwise, so the shape and the whole meta ride through untouched.
    let math = g.add("Math");
    set(math, "math", "multiply", j!(2.0));
    set(math, "math", "post_add", j!(1.0));
    let pm = g.probe(math, "out");
    g.link(src, "out", math, "input");
    let scaled = g.until("the scaled grid", |_| pm.latest().filter(|d| shape(d) == vec![3, 4]));
    assert_eq!(scaled.meta().sfreq(), Some(256.0), "elementwise work leaves the rate alone");

    let func = g.add("Function");
    set(func, "function", "function", j!("negate"));
    let pf = g.probe(func, "out");
    g.link(math, "out", func, "input");
    let flipped = g.until("the negated grid", |_| {
        pf.latest().filter(|d| shape(d) == vec![3, 4] && f32s(d).iter().all(|v| *v < 0.0))
    });
    assert_eq!(shape(&flipped), vec![3, 4], "a function does not change the shape");

    // Reducing the LAST axis takes the rate with it; reducing another leaves time in place.
    let over_time = g.add("Reduce");
    let over_channels = g.add("Reduce");
    set(over_channels, "reduce", "axis", j!(0));
    let (pt, pc) = (g.probe(over_time, "out"), g.probe(over_channels, "out"));
    g.link(src, "out", over_time, "input");
    g.link(src, "out", over_channels, "input");
    let collapsed = g.until("one value per channel", |_| pt.latest().filter(|d| shape(d) == vec![3]));
    assert_eq!(collapsed.meta().sfreq(), None, "the rate belonged to the axis that is gone");
    let across = g.until("one value per sample", |_| pc.latest().filter(|d| shape(d) == vec![4]));
    assert_eq!(across.meta().sfreq(), Some(256.0), "the last axis is still time, so the rate stays");

    // A reorder carries the labels; a re-cut cannot, so it carries nothing.
    let swap = g.add("Reshape");
    set(swap, "reshape", "axes", j!("1,0"));
    let ps = g.probe(swap, "out");
    g.link(src, "out", swap, "input");
    let turned = g.until("the transposed grid", |_| ps.latest().filter(|d| shape(d) == vec![4, 3]));
    let v = f32s(&turned);
    assert!((v[1] - v[0] - 100.0).abs() < 0.01, "column-major order puts the channels together: {v:?}");
    set(swap, "reshape", "shape", j!("12"));
    let flat = g.until("the re-cut frame", |_| ps.latest().filter(|d| shape(d) == vec![12]));
    assert_eq!(flat.meta().sfreq(), None, "a re-cut makes entries the input never had");

    // Select keeps part of an axis and can drop it when one entry is left.
    let pick = g.add("Select");
    set(pick, "select", "mode", j!("index"));
    set(pick, "select", "include", j!("0,2"));
    let pp = g.probe(pick, "out");
    g.link(src, "out", pick, "input");
    g.until("two of the three channels", |_| pp.latest().filter(|d| shape(d) == vec![2, 4]));
    set(pick, "select", "include", j!("1"));
    set(pick, "select", "squeeze", j!(true));
    g.until("the axis to go with the last entry", |_| pp.latest().filter(|d| shape(d) == vec![4]));

    // Join stacks onto a new axis, and names it after the nodes the frames came from.
    let a = g.add("Constant");
    let b = g.add("Constant");
    g.call("node edit", j!({ "node": hex(a), "name": "alpha" }));
    g.call("node edit", j!({ "node": hex(b), "name": "beta" }));
    set(a, "constant", "value", j!(2.0));
    set(a, "constant", "shape", j!("3,4"));
    set(b, "constant", "value", j!(3.0));
    set(b, "constant", "shape", j!("3,4"));
    let end_to_end = g.add("Join");
    let pe = g.probe(end_to_end, "out");
    g.link(a, "out", end_to_end, "input");
    g.link(b, "out", end_to_end, "input");
    g.until("the two frames end to end", |_| pe.latest().filter(|d| shape(d) == vec![6, 4]));

    let join = g.add("Join");
    set(join, "join", "mode", j!("stack"));
    let pj = g.probe(join, "out");
    g.link(a, "out", join, "input");
    g.link(b, "out", join, "input");
    let stacked = g.until("both frames on a new axis", |_| pj.latest().filter(|d| shape(d) == vec![2, 3, 4]));
    let names: Vec<String> = stacked.meta().channels().get(0).and_then(|x| x.coords.clone())
        .expect("the stacked axis is named").iter()
        .map(|c| match c { goofi_core::Coord::Str(s) => s.to_string(), goofi_core::Coord::Num(n) => n.to_string() })
        .collect();
    assert_eq!(names, ["alpha.out", "beta.out"], "the new axis carries its senders");

    // Operation folds the wires left, stretching a length of one the way numpy does.
    let one = g.add("Constant");
    set(one, "constant", "value", j!(5.0));
    let op = g.add("Operation");
    set(op, "operation", "mode", j!("multiply"));
    let po = g.probe(op, "out");
    g.link(a, "out", op, "input");
    g.link(one, "out", op, "input");
    let product = g.until("the broadcast product", |_| {
        po.latest().filter(|d| shape(d) == vec![3, 4] && f32s(d).iter().all(|v| (v - 10.0).abs() < 1e-4))
    });
    assert_eq!(shape(&product), vec![3, 4], "a single value stretches over the frame it meets");

    // And a signal correlated with itself is perfect at no lag, whatever the frame held.
    let auto = g.add("Operation");
    set(auto, "operation", "mode", j!("autocorrelation"));
    let pa = g.probe(auto, "out");
    g.link(src, "out", auto, "input");
    let lags = g.until("the lag axis", |_| pa.latest().filter(|d| shape(d) == vec![3, 7]));
    let v = f32s(&lags);
    assert!((v[3] - 1.0).abs() < 1e-4, "lag zero is a signal against itself: {v:?}");
    let offsets = lags.meta().channels().get(1).and_then(|x| x.coords.clone()).expect("lag coords");
    assert_eq!(offsets[3], goofi_core::Coord::Num(0.0), "the middle of the lag axis is no lag at all");

    for n in [math, func, over_time, over_channels, swap, pick, end_to_end, join, op, auto] {
        assert!(g.error(n).is_none(), "an array node carries no error: {:?}", g.error(n));
    }
}

#[test]
fn the_control_nodes_turn_a_signal_into_a_decision_a_route_and_a_label() {
    // The three nodes that carry no signal of their own, each read against a source that repeats,
    // so a param edit shows on the next frame rather than needing one to be provoked.
    let g = Goofi::new();
    let set = |n, group: &str, name: &str, v: serde_json::Value| {
        g.set_param(n, group, name, v);
    };
    let level = g.add("Constant");
    set(level, "constant", "value", j!(2.0));
    set(level, "constant", "shape", j!("3,4"));

    // Above and below are the same comparison read from either side.
    let over = g.add("Threshold");
    set(over, "threshold", "level", j!(1.0));
    let under = g.add("Threshold");
    set(under, "threshold", "level", j!(1.0));
    set(under, "threshold", "mode", j!("below"));
    let (po, pu) = (g.probe(over, "out"), g.probe(under, "out"));
    g.link(level, "out", over, "input");
    g.link(level, "out", under, "input");
    let high = g.until("the decision above the level", |_| {
        po.latest().filter(|d| shape(d) == vec![3, 4] && f32s(d).iter().all(|v| *v == 1.0))
    });
    assert_eq!(shape(&high), vec![3, 4], "a decision keeps the shape it was given");
    g.until("the same comparison read the other way", |_| {
        pu.latest().filter(|d| f32s(d).iter().all(|v| *v == 0.0))
    });

    // Switch reads its wires in the order they were connected; the next frame carries a new route.
    let grid = g.add("_TestGrid");
    let count = g.add("_TestCounter");
    let route = g.add("Switch");
    let pr = g.probe(route, "out");
    g.link(grid, "out", route, "input");
    g.link(count, "out", route, "input");
    g.until("the first wire", |_| pr.latest().filter(|d| shape(d) == vec![3, 4]));
    set(route, "switch", "index", j!(1));
    g.until("the second wire", |_| pr.latest().filter(|d| shape(d) == vec![1]));

    // Meta writes what a source could not say about itself.
    let named = g.add("Meta");
    set(named, "meta", "sfreq", j!(128.0));
    set(named, "meta", "labels", j!("alpha"));
    let pn = g.probe(named, "out");
    g.link(count, "out", named, "input");
    let stamped = g.until("the written rate", |_| pn.latest().filter(|d| d.meta().sfreq() == Some(128.0)));
    let labels = stamped.meta().channels().get(0).and_then(|x| x.coords.clone()).expect("the written label");
    assert_eq!(labels[0], goofi_core::Coord::Str("alpha".into()), "the name the node was told to write");

    for n in [over, under, route, named] {
        assert!(g.error(n).is_none(), "a control node carries no error: {:?}", g.error(n));
    }
}

#[test]
fn a_stitching_node_answers_from_the_past_and_a_transform_round_trips() {
    // A stitching node keeps no state of its own, only the recent past of its input. The counter
    // sends ONE sample per frame, so a node that keeps no past cannot look back at all. The
    // z-score step below is what pins that: it has a closed form, on one wire, and it fails when
    // the past is taken away. The others share the one `Stream` it proves.
    let g = Goofi::new();
    let set = |n, group: &str, name: &str, v: serde_json::Value| {
        g.set_param(n, group, name, v);
    };
    let counter = g.add("_TestCounter");

    // A level held steady comes out at the same level: the mean is exact, and the clamped head of
    // the window does not pull it. A node that averaged in a zero it had never been sent would.
    let level = g.add("Constant");
    set(level, "constant", "value", j!(6.0));
    set(level, "constant", "shape", j!("16"));
    let smooth = g.add("Smooth");
    set(smooth, "smooth", "size", j!(4));
    let ps = g.probe(smooth, "out");
    g.link(level, "out", smooth, "input");
    let held = g.until("the smoothed level", |_| ps.latest().filter(|d| shape(d) == vec![16]));
    assert!(f32s(&held).iter().all(|v| (v - 6.0).abs() < 1e-4),
            "a steady level smooths to itself: {:?}", f32s(&held));

    // A z-score over a window of eight consecutive whole numbers is the same number every time:
    // the newest sample is 3.5 above the mean, and the spread of that window is the square root
    // of 5.25.
    let norm = g.add("Normalize");
    set(norm, "window", "size", j!(8));
    let pn = g.probe(norm, "out");
    g.link(counter, "out", norm, "input");
    let want = 3.5 / 5.25f32.sqrt();
    g.until("the z-score of a ramp to settle", |_| {
        pn.latest().filter(|d| shape(d) == vec![1] && (f32s(d)[0] - want).abs() < 0.02)
    });

    // A delay keeps the shape and the level it was given, at any reach: it moves the stream along
    // its own axis and invents nothing. What the reach itself buys is the past proven above.
    let delay = g.add("signal:Delay");
    set(delay, "delay", "size", j!(4));
    let pd = g.probe(delay, "out");
    g.link(level, "out", delay, "input");
    let later = g.until("the delayed level", |_| pd.latest().filter(|d| shape(d) == vec![16]));
    assert!(f32s(&later).iter().all(|v| (v - 6.0).abs() < 1e-4),
            "a delay moves the stream, it does not change it: {:?}", f32s(&later));

    // The spectrum both ways over a constant: what comes back is what went in, and the rate the
    // forward pass folded into the bin spacing is read out of it again.
    let flat = g.add("Constant");
    set(flat, "constant", "value", j!(5.0));
    set(flat, "constant", "shape", j!("64"));
    let stamp = g.add("Meta");
    set(stamp, "meta", "sfreq", j!(256.0));
    let fwd = g.add("Fft");
    let back = g.add("Fft");
    set(back, "fft", "mode", j!("inverse"));
    let (pf, pb) = (g.probe(fwd, "out"), g.probe(back, "out"));
    g.link(flat, "out", stamp, "input");
    g.link(stamp, "out", fwd, "input");
    g.link(fwd, "out", back, "input");
    let spectrum = g.until("the one-sided bins", |_| pf.latest().filter(|d| shape(d) == vec![33, 2]));
    assert_eq!(spectrum.meta().sfreq(), None, "a spectrum is not a time series");
    let freqs = spectrum.meta().channels().get(0).and_then(|x| x.coords.clone()).expect("bin coords");
    assert_eq!(freqs[1], goofi_core::Coord::Num(4.0), "64 samples at 256 Hz make four-hertz bins");
    let again = g.until("the samples back", |_| {
        pb.latest().filter(|d| shape(d) == vec![64] && f32s(d).iter().all(|v| (v - 5.0).abs() < 1e-3))
    });
    assert_eq!(again.meta().sfreq(), Some(256.0), "the rate came back out of the bin spacing");

    for n in [smooth, norm, delay, fwd, back] {
        assert!(g.error(n).is_none(), "a stitching node carries no error: {:?}", g.error(n));
    }
}

#[test]
fn the_analysis_nodes_read_a_known_sine_and_say_what_it_is() {
    // One 10 Hz sine at 256 Hz, two seconds of it in a window, read four ways. Every assertion
    // below is a closed form of that sine rather than a property of whatever arrived.
    let g = Goofi::new();
    let set = |n, group: &str, name: &str, v: serde_json::Value| {
        g.set_param(n, group, name, v);
    };
    let lfo = g.add("LFO");
    set(lfo, "lfo", "frequency", j!(10.0));
    set(lfo, "output", "mode", j!("block"));
    set(lfo, "output", "sfreq", j!(256.0));
    set(lfo, "common", "max_frequency", j!(20.0));
    let window = g.add("Buffer");
    set(window, "buffer", "size", j!(512));
    g.link(lfo, "out", window, "input");

    // A sine's envelope is its amplitude, and its instantaneous frequency is its frequency. Both
    // ring at the edges of a frame, so the middle half is what is read.
    let hil = g.add("Hilbert");
    let (pe, ph) = (g.probe(hil, "envelope"), g.probe(hil, "frequency"));
    g.link(window, "out", hil, "input");
    let env = g.until("the envelope of a full window", |_| pe.latest().filter(|d| shape(d) == vec![512]));
    let mid = |v: &[f32]| v[128..384].to_vec();
    let e = mid(&f32s(&env));
    assert!(e.iter().all(|v| (v - 1.0).abs() < 0.05), "a sine of amplitude one has an envelope of one: {:?}", &e[..4]);
    let hz = g.until("the instantaneous frequency", |_| ph.latest().filter(|d| shape(d) == vec![512]));
    let f = mid(&f32s(&hz));
    assert!(f.iter().all(|v| (v - 10.0).abs() < 0.5), "a 10 Hz sine turns at 10 Hz: {:?}", &f[..4]);

    // The wavelet's strongest row is the one whose own frequency is the sine's.
    let wav = g.add("Wavelet");
    let pw = g.probe(wav, "out");
    g.link(window, "out", wav, "input");
    let scal = g.until("a scalogram of the window", |_| pw.latest().filter(|d| shape(d) == vec![40, 512]));
    let v = f32s(&scal);
    let strongest = (0..40)
        .max_by(|a, b| {
            let mean = |r: &usize| v[r * 512 + 128..r * 512 + 384].iter().sum::<f32>();
            mean(a).total_cmp(&mean(b))
        })
        .expect("forty rows");
    let coords = scal.meta().channels().get(0).and_then(|x| x.coords.clone()).expect("the frequency axis");
    let goofi_core::Coord::Num(row_hz) = coords[strongest] else { panic!("numeric frequencies") };
    assert!((row_hz - 10.0).abs() < 1.5, "the strongest row is the sine's own frequency, got {row_hz}");

    // Shifting the sine up by forty hertz moves its peak from ten to fifty.
    let shift = g.add("signal:FreqShift");
    set(shift, "freq_shift", "frequency", j!(40.0));
    let psd = g.add("Psd");
    set(psd, "psd", "mode", j!("fft"));
    let pp = g.probe(psd, "out");
    g.link(window, "out", shift, "input");
    g.link(shift, "out", psd, "input");
    let spectrum = g.until("the shifted spectrum", |_| {
        pp.latest().filter(|d| shape(d) == vec![257] && peak(d).0 > 80 && peak(d).0 < 120)
    });
    assert_eq!(peak(&spectrum).0, 100, "ten hertz shifted up by forty is fifty, at half a hertz a bin");

    // The decomposition names its modes and keeps time last, so the rate rides through.
    let emd = g.add("Emd");
    let pm = g.probe(emd, "out");
    g.link(window, "out", emd, "input");
    let modes = g.until("the modes of the window", |_| pm.latest().filter(|d| shape(d) == vec![5, 512]));
    let names = modes.meta().channels().get(0).and_then(|x| x.coords.clone()).expect("the mode axis");
    assert_eq!(names[0], goofi_core::Coord::Str("IMF1".into()), "the modes are named in order");
    assert_eq!(modes.meta().sfreq(), Some(256.0), "the last axis is still time");

    // Half the rate is half the samples, and the frame says so rather than leaving a reader to
    // divide. The 10 Hz line survives, because 10 Hz is well under the new Nyquist.
    let down = g.add("Resample");
    set(down, "resample", "sfreq", j!(128.0));
    let pd = g.probe(down, "out");
    g.link(window, "out", down, "input");
    let halved = g.until("the window at half the rate", |_| pd.latest().filter(|d| shape(d) == vec![256]));
    assert_eq!(halved.meta().sfreq(), Some(128.0), "the new rate rides the frame");
    let peak_of = |v: &[f32]| v[64..192].iter().fold(0f32, |m, x| m.max(x.abs()));
    assert!((peak_of(&f32s(&halved)) - 1.0).abs() < 0.05, "the sine is still a sine at half the rate");

    // A square matrix of one value has one axis that carries everything and two that carry
    // nothing, which is the answer to check an eigendecomposition against.
    let flat = g.add("Constant");
    set(flat, "constant", "value", j!(2.0));
    set(flat, "constant", "shape", j!("3,3"));
    let eig = g.add("Eigen");
    let (pv, pw) = (g.probe(eig, "values"), g.probe(eig, "vectors"));
    g.link(flat, "out", eig, "input");
    let values = g.until("the eigenvalues of a flat matrix", |_| pv.latest().filter(|d| shape(d) == vec![3]));
    let v = f32s(&values);
    assert!((v[0] - 6.0).abs() < 1e-3, "three rows of two carry six on one axis, got {v:?}");
    assert!(v[1].abs() < 1e-3 && v[2].abs() < 1e-3, "and nothing on the other two, got {v:?}");
    let vectors = g.until("the eigenvectors beside them", |_| pw.latest().filter(|d| shape(d) == vec![3, 3]));
    assert_eq!(shape(&vectors), vec![3, 3], "one vector per column");

    for n in [hil, wav, shift, psd, emd, down, eig] {
        assert!(g.error(n).is_none(), "an analysis node carries no error: {:?}", g.error(n));
    }
}

#[test]
fn the_text_and_table_nodes_carry_a_value_out_to_json_and_back() {
    let g = Goofi::new();
    let set = |n, group: &str, name: &str, v: serde_json::Value| {
        g.set_param(n, group, name, v);
    };

    // Two strings, joined in wire order, then placed by a template that pads one of them.
    let label = g.add("Text");
    set(label, "text", "value", j!("alpha"));
    let unit = g.add("Text");
    set(unit, "text", "value", j!("beta"));
    let fmt = g.add("Format");
    let pf = g.probe(fmt, "out");
    g.link(label, "out", fmt, "input");
    g.link(unit, "out", fmt, "input");
    g.until("the two strings joined in wire order", |_| {
        pf.latest().filter(|d| text(d) == Some("alpha beta"))
    });
    set(fmt, "format", "mode", j!("template"));
    set(fmt, "format", "template", j!("{1}:{0:>7}"));
    g.until("the template to place each wire and pad the first", |_| {
        pf.latest().filter(|d| text(d) == Some("beta:  alpha"))
    });

    // An array and a string become one table, under the keys asked for, and that is what json says.
    let level = g.add("Constant");
    set(level, "constant", "value", j!(2.5));
    set(level, "constant", "shape", j!("2"));
    let table = g.add("Table");
    set(table, "table", "keys", j!("level,name"));
    let json = g.add("ToJson");
    let pj = g.probe(json, "out");
    g.link(level, "out", table, "arrays");
    g.link(fmt, "out", table, "strings");
    g.link(table, "out", json, "input");
    let written = r#"{"level": [2.5, 2.5], "name": "beta:  alpha"}"#;
    g.until("the table written as json", |_| pj.latest().filter(|d| text(d) == Some(written)));

    // And back: the same text parses to the same table, and one field leaves on the output whose
    // kind it has — the other two stay silent.
    let back = g.add("FromJson");
    let pick = g.add("TableSelect");
    set(pick, "table", "key", j!("level"));
    let (pa, ps) = (g.probe(pick, "array"), g.probe(pick, "string"));
    g.link(json, "out", back, "input");
    g.link(back, "out", pick, "input");
    let round = g.until("the array field back out of the parsed table", |_| {
        pa.latest().filter(|d| shape(d) == vec![2])
    });
    assert_eq!(f32s(&round), vec![2.5, 2.5], "the numbers survive the text and come back");
    assert!(ps.latest().is_none(), "an array field leaves the string output silent");

    set(pick, "table", "key", j!("name"));
    g.until("the string field on the string output", |_| {
        ps.latest().filter(|d| text(d) == Some("beta:  alpha"))
    });

    for n in [fmt, table, json, back, pick] {
        assert!(g.error(n).is_none(), "a text node carries no error: {:?}", g.error(n));
    }
}
