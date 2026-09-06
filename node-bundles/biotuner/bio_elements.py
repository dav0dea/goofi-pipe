"""BioElements — the peaks of a signal matched against the emission spectra of the elements.

Takes PEAKS in Hz, as `Peaks` emits them, and asks which elements have spectral lines standing in
the same relations. An element's lines sit in the optical range and a biosignal's peaks sit under
50 Hz, so nothing is compared in absolute terms: the peaks are folded into `band` and matched by
INTERVAL, in cents. What comes back is a ranking, not an identification — a signal does not contain
hydrogen, its intervals resemble hydrogen's.

Inputs:
  input  peaks in Hz, as `Peaks` emits them
  amps   optional, and read by `method` `tuning` alone

Outputs:
  elements    one entry per element kept, its score per row of input
  scores      the same scores as numbers, ranked, for binding or plotting
  ranked      the element names in that order, so a reader can label the scores
  categories  what each ranked element is — a noble gas, a lanthanide — in the same order

`method` decides what "resemble" measures. `lines` counts how many peaks land within `tolerance`
of a line, which rewards an element with lines where the signal has peaks. `tuning` compares the
whole set as a scale, weighting by harmonic simplicity, so it answers which element's spectrum is
built like this signal rather than which one overlaps it — and it is the one that reads `amps`.

`table` picks the line list: `air` are the wavelengths as measured through air, `vacuum` as
measured without it. They differ slightly and consistently; `air` is the usual one.

`depth` and `keep` are different questions and neither substitutes for the other. `depth` is how
many of each element's lines are looked at, and it MOVES the ranking — five makes hydrogen first at
0.900, forty makes potassium first at 0.695, because a sparse element scores well while little of
it is in view. `keep` only trims the answer.

`scope` is what keeps this affordable. `pooled` matches every channel's peaks together and answers
once, which is what a whole signal usually asks; `each` keeps the bundle's convention that every
axis before the last survives, and pays for it — see the cost below. The table carries one score
per row under each element either way, so its shape never depends on the data.

This node is SLOW where the rest of the bundle is not: a match scores all 99 elements and costs
about 300ms here, so it emits at three a second rather than thirty. That is a property of the
comparison rather than a fault, and it costs its neighbours nothing — every node owns its thread —
but do not put it where a frame-rate answer is wanted.

Every ROW pays that separately, which is what makes a many-channel frame a problem rather than a
wait: thirty-two channels want nine seconds and the subprocess tier gives a tick ten. So the work
is bounded by `budget` and refuses with the measured cost when a frame will not fit, rather than
dying late with nothing to say.
"""

import time

import numpy as np
from biotuner.bioelements import match_elements, match_elements_by_tuning
import goofi


class BioElements(goofi.Node):
    """Which elements' spectra stand in the same relations as a signal's peaks."""

    TAGS = ["analysis"]
    INPUTS = {
        "input": goofi.InputSlot(goofi.DataType.ARRAY, required=True),
        "amps": goofi.InputSlot(goofi.DataType.ARRAY, required=False),
    }
    OUTPUTS = {
        "elements": goofi.DataType.TABLE,
        "scores": goofi.DataType.ARRAY,
        "ranked": goofi.DataType.STRING,
        "categories": goofi.DataType.STRING,
    }
    PARAMS = {
        "elements": {
            "method": goofi.StringParam(
                "lines", ["lines", "tuning"], doc="Match by lines the peaks land on, or by the scale they make."
            ),
            "table": goofi.StringParam("air", ["air", "vacuum"], doc="Which line list — measured through air, or without."),
            "scope": goofi.StringParam(
                "pooled", ["pooled", "each"], doc="One answer for the whole frame, or one per channel."
            ),
            "keep": goofi.IntParam(8, 1, 40, doc="How many elements to answer with, best first."),
            "budget": goofi.FloatParam(5.0, 0.5, 9.0, doc="Seconds a frame may spend matching before it refuses."),
            "depth": goofi.IntParam(40, 1, 200, doc="Spectral lines per element the match considers."),
            "tolerance": goofi.FloatParam(50.0, 1.0, 200.0, doc="lines: how far from a line a peak may sit, in cents."),
            "bandLow": goofi.FloatParam(3000.0, 20.0, 20000.0, doc="lines: bottom of the band the peaks fold into, in Hz."),
            "bandHigh": goofi.FloatParam(7000.0, 40.0, 40000.0, doc="lines: top of that band, in Hz."),
        }
    }

    def process(self, input, amps=None):
        p = self.params.elements
        x = np.asarray(input.data, dtype=np.float64)
        if x.ndim == 0:
            raise ValueError("BioElements reads a list of peaks, not a single number")
        given = None if amps is None else np.asarray(amps.data, dtype=np.float64)

        lead, rows = x.shape[:-1], x.reshape(-1, x.shape[-1])
        arows = None if given is None else given.reshape(-1, given.shape[-1])
        # Pooled is one match over every channel's peaks at once, which is the question a whole
        # signal usually asks — and the only shape that stays affordable on a 64-channel frame.
        if p.scope == "pooled":
            lead, rows = (), rows.reshape(1, -1)
            arows = None if arows is None else arows.reshape(1, -1)

        # A match costs about a third of a second and every row pays it, so a many-channel frame
        # walks past the tier's tick deadline and the node dies saying only that it was late.
        # Spend a budget instead, and name what it bought.
        started = time.monotonic()
        ranked = []
        for i, row in enumerate(rows):
            ranked.append(self._rank(row, None if arows is None else arows[i % arows.shape[0]], p))
            if time.monotonic() - started > p.budget and i + 1 < rows.shape[0]:
                each = (time.monotonic() - started) / (i + 1)
                raise ValueError(
                    f"BioElements matched {i + 1} of {rows.shape[0]} rows in {p.budget}s at {each * 1e3:.0f}ms a row — "
                    "take fewer channels upstream, or raise `budget`"
                )

        # Ranked by the MEAN across rows, so one element means one row of scores whatever the
        # per-row order was, and the table's shape never depends on the data.
        names = sorted({n for r in ranked for n in r}, key=lambda n: -np.mean([r.get(n, (0.0, ""))[0] for r in ranked]))
        names = names[: p.keep]
        scores = np.array([[r.get(n, (0.0, ""))[0] for n in names] for r in ranked]) if names else np.zeros((rows.shape[0], 1))
        kinds = [next((r[n][1] for r in ranked if n in r), "") for n in names]

        return {
            "elements": {n: np.asarray(scores[:, k], dtype=np.float32) for k, n in enumerate(names)},
            "scores": scores.reshape(lead + (scores.shape[-1],)).astype(np.float32),
            "ranked": ", ".join(names),
            "categories": ", ".join(kinds),
        }

    def _rank(self, row, row_amps, p):
        peaks = [float(v) for v in row if np.isfinite(v) and v > 0]
        # A match is over a SET of relations, so one peak names nothing.
        if len(peaks) < 2:
            return {}
        if p.method == "tuning":
            amp = None
            if row_amps is not None:
                a = np.asarray([v for v in row_amps if np.isfinite(v)], dtype=np.float64)[: len(peaks)]
                amp = a.tolist() if a.size == len(peaks) else None
            df = match_elements_by_tuning(peaks, amps=amp, table=p.table, top=p.depth)
            column = "tuning_score"
        else:
            df = match_elements(
                peaks, table=p.table, top=p.depth, tol_cents=p.tolerance, band=(p.bandLow, p.bandHigh)
            )
            column = "score"
        head = df.head(p.keep)
        return {str(r.element): (float(getattr(r, column)), str(r.category)) for r in head.itertuples()}
