//! The signal nodes doing signal work: a stream is shaped, filtered, windowed and transformed,
//! and what comes out the far end is measured rather than assumed.
//!
//! DIMENSIONALITY is the property that spans the set, so the source emits a grid, not a vector.

use goofi_core::Data;
use goofi_tests::{f32s, shape, text, j, Goofi};

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
    let osc = g.add("Oscillator");
    let flt = g.add("Filter");
    let buf = g.add("Buffer");
    let psd = g.add("Psd");
    let set = |n, group: &str, name: &str, v: serde_json::Value| {
        g.set_param(n, group, name, v);
    };
    set(osc, "oscillator", "sfreq", j!(256.0));
    set(osc, "oscillator", "frequency", j!(10.0));
    set(buf, "buffer", "size", j!(512));
    set(flt, "filter", "low", j!(5.0));
    set(flt, "filter", "high", j!(20.0));

    let probe = g.probe(psd, "psd"); // opened before the wires: the data services keep no history
    g.link(osc, "out", flt, "data");
    g.link(flt, "out", buf, "data");
    g.link(buf, "out", psd, "data");

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

    // The window is a second long, so this waits for the property rather than for the next frame.
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
        g.link(src, "out", b, "data");
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
    let noise = g.add("Noise");
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

    // A Constant has no input and does not autotrigger, so nothing in the graph can ring it. An
    // edit is what runs it, and the shape it is asked for is the shape that comes out.
    let pk = g.probe(konst, "out");
    set(konst, "constant", "value", j!(3.0));
    set(konst, "constant", "shape", j!("2,3"));
    let filled = g.until("the constant to answer its edit", |_| {
        pk.latest().filter(|d| shape(d) == vec![2, 3] && f32s(d).iter().all(|v| *v == 3.0))
    });
    assert_eq!(f32s(&filled).len(), 6, "the shape it was asked for is the shape it filled");

    // And a wire made long after that one emit still gets a frame, because pub/sub keeps none.
    let buf = g.add("Buffer");
    let pb = g.probe(buf, "out");
    g.link(konst, "out", buf, "data");
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
