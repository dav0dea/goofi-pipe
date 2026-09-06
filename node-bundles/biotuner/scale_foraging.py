"""ScaleForaging — a tuning is held until one that is related enough arrives to replace it.

Takes a TUNING (ratios inside an octave), as `Tuning` emits them, and passes one through. Every
frame `Tuning` offers a new scale; most of them are discarded. A candidate replaces the held scale
only when its similarity to that scale clears `threshold` and at least `rate` seconds have passed
since the last change, so the tuning walks between related scales instead of flickering with the
signal. That walk is what makes a biotuning playable: a scale that changes every frame is noise,
and one that never changes is not listening.

Similarity is asymmetric on purpose. Every degree of the CANDIDATE is scored against its best
match in the held scale and the scores are averaged, which asks "does this scale sit inside the one
I am playing?" — so a candidate that adds a degree is close and one that moves every degree is not.
Two identical scales score 100.

  dyad   how harmonically related the two degrees are, from biotuner's `dyad_similarity`: the
         reading the scale itself is built on, where a fifth apart is closer than a tritone apart
  cents  how far apart they are in pitch, 100 at unison falling to 0 at `tolerance` cents away,
         which hears no harmony at all and only distance

`direction` flips the test: `similar` keeps the walk close, `different` forages for contrast and
takes only the candidates that are far from what is playing.

Nothing here decides on its own — `threshold`, `rate` and `hold` are params, so a feature from
anywhere in the patch drives them by reference. That is the device's "decision feature", with any
feature and any target rather than one of each.

The first candidate is taken outright, with nothing to compare it against: `changed` is one and
`similarity` is NaN on that frame. Every axis before the last survives, so a scale per channel
forages per channel, and rows are padded with NaN to the widest held scale as the rest of the
bundle pads.
"""

import time

import numpy as np
from biotuner.metrics import dyad_similarity
import goofi


def similarity(candidate, held, metric, tolerance):
    """How close `candidate` sits to `held`, 0..100, matching each degree to its best partner."""
    partners = [h for h in held if h > 0]
    if not partners or not candidate:
        return 0.0
    scores = []
    for c in candidate:
        if metric == "cents":
            apart = min(abs(1200.0 * np.log2(c / h)) for h in partners)
            scores.append(100.0 * max(0.0, 1.0 - apart / tolerance))
        else:
            scores.append(max(dyad_similarity(c / h) for h in partners))
    return float(np.mean(scores))


class ScaleForaging(goofi.Node):
    """Hold a tuning, and take a new one only when it is related enough to the last."""

    TAGS = ["music", "control"]
    INPUTS = {"input": goofi.InputSlot(goofi.DataType.ARRAY, required=True)}
    OUTPUTS = {
        "tuning": goofi.DataType.ARRAY,
        "similarity": goofi.DataType.ARRAY,
        "changed": goofi.DataType.ARRAY,
    }
    PARAMS = {
        "foraging": {
            "threshold": goofi.FloatParam(50.0, 0.0, 100.0, doc="How close a candidate must be to be taken, 0..100."),
            "direction": goofi.StringParam(
                "similar",
                ["similar", "different"],
                doc="Whether a candidate is taken for clearing the threshold or for falling below it.",
            ),
            "rate": goofi.FloatParam(2.0, 0.0, 600.0, doc="The least time in seconds between two changes."),
            "metric": goofi.StringParam(
                "dyad", ["dyad", "cents"], doc="How two degrees are compared: harmonically, or by pitch distance."
            ),
            "tolerance": goofi.FloatParam(
                100.0, 1.0, 1200.0, doc="cents: how far apart two degrees may sit before they score nothing. 100 is a semitone."
            ),
            "hold": goofi.BoolParam(False, doc="Freeze the scale being played, whatever arrives."),
            "reset": goofi.PulseParam(doc="Forget the held scale, so the next candidate is taken outright."),
        }
    }

    def setup(self):
        # One held scale per row of the incoming batch, and when that row was last replaced.
        self.held = []
        self.last = []

    def pulse_foraging_reset(self):
        self.held = []
        self.last = []

    def process(self, input):
        p = self.params.foraging
        x = np.asarray(input.data, dtype=np.float64)
        if x.ndim == 0:
            raise ValueError("ScaleForaging reads a scale, not a single number")

        lead, rows = x.shape[:-1], x.reshape(-1, x.shape[-1])
        # A different batch width is a different set of scales; nothing carries across it.
        if len(self.held) != rows.shape[0]:
            self.held = [None] * rows.shape[0]
            self.last = [0.0] * rows.shape[0]

        now = time.monotonic()
        scores = np.full(rows.shape[0], np.nan)
        changed = np.zeros(rows.shape[0])
        for i, row in enumerate(rows):
            candidate = [float(v) for v in row if np.isfinite(v) and v > 0]
            if not candidate:
                continue
            if self.held[i] is None:
                self.held[i], self.last[i], changed[i] = candidate, now, 1.0
                continue
            score = similarity(candidate, self.held[i], p.metric, p.tolerance)
            scores[i] = score
            close = score >= p.threshold
            wanted = close if p.direction == "similar" else not close
            if wanted and not p.hold and now - self.last[i] >= p.rate:
                self.held[i], self.last[i], changed[i] = candidate, now, 1.0

        width = max((len(h) for h in self.held if h), default=0) or 1
        out = np.full((rows.shape[0], width), np.nan)
        for i, h in enumerate(self.held):
            if h:
                out[i, : len(h)] = h
        return {
            "tuning": out.reshape(lead + (width,)).astype(np.float32),
            "similarity": scores.reshape(lead if lead else (1,)).astype(np.float32),
            "changed": changed.reshape(lead if lead else (1,)).astype(np.float32),
        }
