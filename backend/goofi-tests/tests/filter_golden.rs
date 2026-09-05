//! The zero-phase filter against scipy. `sosfiltfilt` is the authority for what a Butterworth band
//! run both ways does to a signal, and the node is measured against it sample by sample.
//!
//! Lowpass and highpass only. A Butterworth cascade of second-order sections is the bilinear
//! transform of one analog prototype, so the two designs agree exactly there. A bandpass does not:
//! scipy transforms the prototype to a band, and the node cascades a highpass with a lowpass.
//!
//! `tests/gen_filter_golden.py` writes the fixture; the case list here mirrors it.

use goofi_tests::{f32s, install, require_python, shape, Goofi};

const GOLDEN: &str = include_str!("fixtures/filter_golden.json");

/// A producer that emits the golden's input, so the node under test sees the same samples scipy did.
fn source_of(input: &[f32]) -> String {
    format!(
        "import goofi\nimport numpy as np\n\n\nclass Golden(goofi.Node):\n    \
         \"\"\"Emits the filter golden's input, at the rate it was generated for.\"\"\"\n\n    \
         TAGS = [\"generator\"]\n    OUTPUTS = {{\"out\": goofi.DataType.ARRAY}}\n    \
         PRODUCER = True\n    SAMPLES = {input:?}\n\n    \
         def process(self):\n        \
         return np.array(self.SAMPLES, dtype=np.float32), {{\"sfreq\": 256.0}}\n",
    )
}

#[test]
fn the_zero_phase_filter_answers_what_scipy_answers() {
    let _py = require_python();
    let golden: serde_json::Value = serde_json::from_str(GOLDEN).expect("the golden parses");
    let numbers = |v: &serde_json::Value| -> Vec<f32> {
        v.as_array().expect("an array").iter().map(|x| x.as_f64().expect("a number") as f32).collect()
    };
    let input = numbers(&golden["input"]);

    let g = Goofi::new();
    install(&g, "golden.py", &source_of(&input));
    let src = g.add("Golden");
    let flt = g.add("signal:Filter");
    let probe = g.probe(flt, "out");
    g.link(src, "out", flt, "input");

    for case in golden["cases"].as_array().expect("cases") {
        let (name, mode) = (case["name"].as_str().unwrap(), case["mode"].as_str().unwrap());
        let (cutoff, order) = (case["cutoff"].as_f64().unwrap(), case["order"].as_i64().unwrap());
        let want = numbers(&case["expected"]);
        g.set_param(flt, "filter", "mode", mode);
        g.set_param(flt, "filter", "order", order);
        // The edge the mode does not use is pushed out of the way, so it cannot narrow the band.
        let (low, high) = if mode == "highpass" { (cutoff, 128.0) } else { (0.0, cutoff) };
        g.set_param(flt, "filter", "low", low);
        g.set_param(flt, "filter", "high", high);

        let worst = |got: &[f32]| {
            got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max)
        };
        // The node computes in f32 where scipy computes in f64, so the gate is a thousandth of the
        // signal's own swing rather than an equality.
        let out = g.until(name, |_| {
            probe.latest().filter(|d| shape(d) == vec![input.len()] && worst(&f32s(d)) < 2.0e-3)
        });
        assert_eq!(f32s(&out).len(), want.len(), "{name}: the whole span comes back");
        assert!(g.error(flt).is_none(), "{name}: {:?}", g.error(flt));
    }
}
