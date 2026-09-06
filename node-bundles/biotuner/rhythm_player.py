"""RhythmPlayer — a pattern walked in time, so a rhythm becomes gates a synth can play.

`EuclidRhythm` and `Polyrhythm` answer what a rhythm IS; nothing there advances. This walks a
pattern at a tempo and answers where it is now: one gate per voice, high for the front of a step
that carries an onset. Wire `gate` through `audio:SignalIn` and it crosses as one channel per
voice, which is what a plugin's gate or an envelope reads.

  gate     one row per voice, 1 while its onset sounds and 0 otherwise
  step     which position of the grid is playing, counted from zero
  phase    how far through the current step, 0 to 1, for anything that wants to slide

Takes the pattern matrix either node emits — `[voices, steps]`, or a single `[steps]` row. NaN is
padding and reads as no onset, so a ragged `EuclidRhythm` batch plays without trimming.

Time is read from the CLOCK rather than counted in frames: a node's rate is not a musical unit and
a dropped frame must not slow the music down. `length` therefore matters more than the frame rate —
a step shorter than the gap between two frames is a step that can be stepped over, and at goofi's
usual rate that floor is around 30ms, so a grid of 360 positions wants a slow `bpm` to be heard
whole.
"""

import time

import numpy as np
import goofi


class RhythmPlayer(goofi.Node):
    """Walk a rhythm pattern at a tempo and emit its gates."""

    TAGS = ["transform", "music"]
    INPUTS = {"input": goofi.InputSlot(goofi.DataType.ARRAY, required=True)}
    OUTPUTS = {
        "gate": goofi.DataType.ARRAY,
        "step": goofi.DataType.ARRAY,
        "phase": goofi.DataType.ARRAY,
    }
    PARAMS = {
        "player": {
            "bpm": goofi.FloatParam(120.0, 1.0, 600.0, doc="Beats a minute."),
            "stepsPerBeat": goofi.FloatParam(4.0, 0.25, 16.0, doc="Grid positions to a beat. 4 is sixteenths."),
            "length": goofi.FloatParam(0.5, 0.01, 1.0, doc="How much of a step the gate stays high for."),
            "running": goofi.BoolParam(True, doc="Off holds the position and drops every gate."),
            "restart": goofi.PulseParam(doc="Return to the start of the grid."),
        }
    }

    def setup(self):
        self.started = time.monotonic()
        self.held = 0.0  # grid positions elapsed, kept across a stop so a pause does not rewind

    def pulse_player_restart(self):
        self.started, self.held = time.monotonic(), 0.0

    def process(self, input):
        p = self.params.player
        x = np.asarray(input.data, dtype=np.float64)
        if x.ndim == 0:
            raise ValueError("RhythmPlayer reads a pattern, not a single number")
        grid = x.reshape(1, -1) if x.ndim == 1 else x.reshape(-1, x.shape[-1])
        # The last two axes are the pattern; anything before them is a batch this cannot play.
        voices, steps = grid.shape[-2], grid.shape[-1]
        if steps < 1:
            raise ValueError("RhythmPlayer reads a pattern with at least one step")

        now = time.monotonic()
        rate = p.bpm * p.stepsPerBeat / 60.0
        if p.running:
            self.held += (now - self.started) * rate
        self.started = now

        at = self.held % steps
        index = int(at)
        phase = at - index
        # NaN is padding, and padding is silence.
        onsets = np.nan_to_num(grid[:, index], nan=0.0)
        gate = np.where((onsets > 0.5) & p.running & (phase < p.length), 1.0, 0.0)

        return {
            "gate": gate.reshape(voices, 1).astype(np.float32),
            "step": np.asarray([[float(index)]], dtype=np.float32),
            "phase": np.asarray([[float(phase)]], dtype=np.float32),
        }
