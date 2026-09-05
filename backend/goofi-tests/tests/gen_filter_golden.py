#!/usr/bin/env python3
"""Generate the zero-phase filter golden from scipy, the authority the node is measured against.

Run with an interpreter that has numpy and scipy:

    .gfivenv/bin/python backend/goofi-tests/tests/gen_filter_golden.py

Writes backend/goofi-tests/tests/fixtures/filter_golden.json — the input, the design, and what
`sosfiltfilt` makes of it. The Rust scenario feeds the SAME input to the node and compares.

Lowpass and highpass only. A Butterworth cascade of second-order sections is the bilinear
transform of one analog prototype, so scipy's design and the node's agree exactly there. A
bandpass does NOT agree: scipy transforms the prototype to a band, and the node cascades a
highpass with a lowpass, which is a different filter with the same name.
"""
import json
import os

import numpy as np
from scipy.signal import butter, sosfiltfilt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fixtures", "filter_golden.json")

SFREQ = 256.0
N = 512


def signal():
    """A sine, a slower sine and a fixed wobble: enough that every band changes the answer."""
    t = np.arange(N) / SFREQ
    rng = np.random.default_rng(7)
    return (
        np.sin(2 * np.pi * 10.0 * t)
        + 0.5 * np.sin(2 * np.pi * 2.0 * t)
        + 0.2 * rng.standard_normal(N)
    ).astype(np.float32)


CASES = [
    {"name": "lowpass_4_at_5", "mode": "lowpass", "cutoff": 5.0, "order": 4},
    {"name": "highpass_4_at_5", "mode": "highpass", "cutoff": 5.0, "order": 4},
    {"name": "lowpass_2_at_20", "mode": "lowpass", "cutoff": 20.0, "order": 2},
]


def main():
    x = signal()
    cases = []
    for case in CASES:
        sos = butter(case["order"], case["cutoff"], btype=case["mode"], fs=SFREQ, output="sos")
        y = sosfiltfilt(sos, x.astype(np.float64))
        cases.append({**case, "expected": [float(v) for v in y]})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"sfreq": SFREQ, "input": [float(v) for v in x], "cases": cases}, f)
    print(f"wrote {OUT}: {len(cases)} cases over {N} samples")


if __name__ == "__main__":
    main()
