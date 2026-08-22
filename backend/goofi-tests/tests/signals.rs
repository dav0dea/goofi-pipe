//! The signal nodes doing signal work: a stream is shaped, filtered, windowed and transformed,
//! and what comes out the far end is measured rather than assumed.
//!
//! DIMENSIONALITY is the property that spans the set, so the source emits a grid, not a vector.

use goofi_core::{Data, Value};
use goofi_tests::{hex, j, Goofi};

fn shape(d: &Data) -> Vec<usize> {
    let Value::Array(a) = d.value() else { panic!("not an array: {d:?}") };
    a.shape().to_vec()
}

fn f32s(d: &Data) -> Vec<f32> {
    let Value::Array(a) = d.value() else { panic!("not an array: {d:?}") };
    a.as_bytes().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

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
        g.call("set_param", j!({ "node": hex(n), "group": group, "name": name, "value": v }));
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
        g.call("set_param", j!({ "node": hex(n), "group": "buffer", "name": name, "value": v }));
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
