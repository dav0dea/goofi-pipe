"""TuningMatrix — how consonant every degree of a scale is against every other.

Takes a TUNING (ratios inside an octave), as `Tuning` emits them. Each pair of degrees is scored
by one harmonicity measure, giving a square grid.

Inputs:
  input  a tuning: ratios inside an octave, as `Tuning` emits them

Outputs:
  matrix         the full `[N, N]` grid, symmetric, for a viewer to draw
  metricPerStep  one number per degree — the grid's column means — which is what a param follows
                 when you want each scale degree to drive something of its own
  metric         one number for the whole scale: how consonant this tuning is overall

The last axis is scale degrees and is consumed. Every axis before it survives, so `[C, N]` in
gives `[C, N, N]`, `[C, N]` and `[C]`. Rows are padded with NaN to the widest scale in the batch,
and NaN padding on the way in is dropped, so a `Tuning` output feeds straight in.

Scores are NOT normalised and their range depends on `function`: `dyad_similarity` runs 0..100,
`consonance` 0..1, and `metric_denom` grows with the denominators, so lower is simpler there.
"""

import numpy as np
from biotuner.metrics import compute_consonance, dyad_similarity, metric_denom, tuning_cons_matrix
import goofi

FUNCTIONS = {"dyad_similarity": dyad_similarity, "consonance": compute_consonance, "metric_denom": metric_denom}


class TuningMatrix(goofi.Node):
    """The pairwise consonance of a scale's degrees."""

    TAGS = ["analysis"]
    INPUTS = {"input": goofi.InputSlot(goofi.DataType.ARRAY, required=True)}
    OUTPUTS = {
        "matrix": goofi.DataType.ARRAY,
        "metricPerStep": goofi.DataType.ARRAY,
        "metric": goofi.DataType.ARRAY,
    }
    PARAMS = {
        "matrix": {
            "function": goofi.StringParam(
                "dyad_similarity",
                list(FUNCTIONS),
                doc="How each pair is scored. `dyad_similarity` 0..100, `consonance` 0..1, "
                "`metric_denom` grows with the denominators.",
            ),
            "ratio_type": goofi.StringParam(
                "all",
                ["all", "pos_harm", "sub_harm"],
                doc="Which intervals are scored: every pair, only those above the root, or only those below.",
            ),
        }
    }

    def process(self, input):
        p = self.params.matrix
        x = np.asarray(input.data, dtype=np.float64)
        if x.ndim == 0:
            raise ValueError("TuningMatrix reads a scale, not a single number")

        lead, rows = x.shape[:-1], x.reshape(-1, x.shape[-1])
        scored = []
        for row in rows:
            scale = [float(v) for v in row if np.isfinite(v)]
            if len(scale) < 2:
                scored.append((np.empty((0, 0)), np.empty(0), np.nan))
                continue
            per_step, metric, matrix = tuning_cons_matrix(scale, FUNCTIONS[p.function], ratio_type=p.ratio_type)
            scored.append(
                (
                    np.asarray(matrix, dtype=np.float64),
                    np.asarray(per_step, dtype=np.float64).ravel(),
                    float(metric),
                )
            )

        width = max((s.size for _, s, _ in scored), default=0) or 1
        mats = np.full((rows.shape[0], width, width), np.nan)
        steps = np.full((rows.shape[0], width), np.nan)
        whole = np.full(rows.shape[0], np.nan)
        for i, (m, s, v) in enumerate(scored):
            if m.size:
                n = min(m.shape[0], width)
                mats[i, :n, :n] = m[:n, :n]
            steps[i, : s.size] = s[:width]
            whole[i] = v
        return {
            "matrix": mats.reshape(lead + (width, width)).astype(np.float32),
            "metricPerStep": steps.reshape(lead + (width,)).astype(np.float32),
            "metric": whole.reshape(lead if lead else (1,)).astype(np.float32),
        }
