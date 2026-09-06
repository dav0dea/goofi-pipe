"""VitalPreset — a tuning becomes a Vital patch, written when you ask for it.

Takes a TUNING (ratios inside an octave), as `Tuning` emits them, and builds a biotuner `Timbre`
whose partials are those ratios over `base_freq`. Writing is a PULSE rather than a stream: a
preset is a file, and a synth wants one when you have found a sound you like, not sixty times a
second. `path` carries where the last one went, so an agent can pick it up.

Wire the optional `signal` input to hand the raw biosignal to the kinds that can use it — only
`ensemble` does today, which writes a whole bundle rather than a single patch.


`kind` is one of:
  spectral        the partials as Vital's own harmonic editor sees them: one patch, one spectrum
  inharmonic      the same partials left unsnapped, so a stretched or gamelan-like set stays so
  wavetableMorph  a 64-frame wavetable that sweeps the spectral tilt, with an LFO on the position
  ensemble        a directory rather than a patch: every projection biotuner can make of it

`matching` decides how the partials are fitted to the ratios — `consonance_weighted` favours the
simple intervals, `sethares` minimises roughness, `direct` does not fit at all and takes the
ratios as they came.
"""

import os
import sys

import numpy as np
from biotuner.harmonic_timbre import timbre_from_ratios
from biotuner.harmonic_timbre.exporters import to_vital
import goofi

# The folder each platform's Vital reads its user patches from, so a written preset is one the
# synth already lists rather than one to go hunting for.
def _vital_dir():
    home = os.path.expanduser("~")
    if sys.platform == "win32":
        return os.path.join(home, "Documents", "Vital")
    if sys.platform == "darwin":
        return os.path.join(home, "Music", "Vital")
    return os.path.join(home, ".local", "share", "vital")


KINDS = {
    "spectral": to_vital.to_vital_spectral,
    "inharmonic": to_vital.to_vital_inharmonic,
    "wavetableMorph": to_vital.to_vital_wavetable_morph,
    "ensemble": to_vital.to_vital_ensemble,
}


class VitalPreset(goofi.Node):
    """Write a Vital preset built from a tuning.

    Inputs:
      input   a tuning: ratios inside an octave, as `Tuning` emits them
      signal  optional raw biosignal, read by the `ensemble` kind alone

    Outputs:
      path  where the last preset was written, so an agent can pick it up
    """

    TAGS = ["output"]
    INPUTS = {
        "input": goofi.InputSlot(goofi.DataType.ARRAY, required=True),
        "signal": goofi.InputSlot(goofi.DataType.ARRAY, required=False),
    }
    OUTPUTS = {"path": goofi.DataType.STRING}
    PARAMS = {
        "preset": {
            "kind": goofi.StringParam("spectral", list(KINDS), doc="Which projection of the timbre to write."),
            "name": goofi.StringParam("biotuner", doc="The patch's name, and its file name."),
            "folder": goofi.StringParam("", doc="Where to write. Empty is this platform's Vital user folder."),
            "base_freq": goofi.FloatParam(220.0, 20.0, 2000.0, doc="The frequency ratio 1 sits at, in Hz."),
            "matching": goofi.StringParam(
                "consonance_weighted",
                ["consonance_weighted", "direct", "sethares", "harmonic_entropy", "hybrid"],
                doc="How the partials are fitted to the ratios.",
            ),
            "write": goofi.PulseParam(doc="Write the preset from the tuning last seen."),
        }
    }

    def setup(self):
        self.ratios = None
        self.sfreq = None
        self.signal = None
        self.path = ""

    def process(self, input, signal=None):
        x = np.squeeze(np.asarray(input.data, dtype=np.float64))
        self.ratios = [float(v) for v in np.atleast_1d(x) if np.isfinite(v)]
        if signal is not None:
            self.signal = np.squeeze(np.asarray(signal.data, dtype=np.float64))
            self.sfreq = signal.meta.get("sfreq")
        return {"path": self.path}

    def pulse_preset_write(self):
        p = self.params.preset
        # A scale is a set of INTERVALS, so one ratio makes no timbre to write.
        if not self.ratios or len(self.ratios) < 2:
            raise ValueError("VitalPreset needs a tuning of at least two ratios before it can write")

        timbre = timbre_from_ratios(self.ratios, matching_method=p.matching, base_freq=p.base_freq)
        folder = p.folder or _vital_dir()
        os.makedirs(folder, exist_ok=True)
        name = p.name or "biotuner"

        if p.kind == "ensemble":
            # The ensemble writes a DIRECTORY, and takes the raw signal where one is wired.
            extra = {}
            if self.signal is not None and self.sfreq:
                extra = {"signal": self.signal, "sf": float(self.sfreq)}
            KINDS[p.kind](timbre, os.path.join(folder, name), bundle_name=name, **extra)
            self.path = os.path.join(folder, name)
        else:
            out = os.path.join(folder, name + ".vital")
            KINDS[p.kind](timbre, out, preset_name=name)
            self.path = out
