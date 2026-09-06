"""MicrotonalKeys — a keyboard plays a tuning, one voice per key, at the tuning's own step count.

Takes a TUNING (ratios inside an octave) and the `notes` frame from `signal:MidiIn`, and answers
what each held key should SOUND as under that tuning — as pitch, gate and velocity per voice,
ready to reference from a plugin's `voice` params.

A key is a STEP of the scale, not a semitone. With seven ratios the octave lands every seven keys,
with five every five: the keyboard is relabelled rather than retuned, which is what lets a scale
of any size be played without running out of keys. `base_note` is the key that plays the first
ratio, and `base_freq` is what that key sounds at — set it to C0's 16.35 Hz and the scale is built
up from there.

This needs no MIDI channels and no pitch bend. Those exist because MIDI can only send a whole
note number, so a per-note tuning has to be smuggled through one bend per channel — which is what
caps a patch at as many notes as it has channels. goofi's pitch is already continuous volts per
octave, zero at C4, so the microtonality is simply the number, and the only ceiling is `voices`.

  voices     the whole keyboard in one wire: pitches then velocities, the layout a plugin's
             `voice` input reads — cross it with `audio:SignalIn` and that is the only cable
  pitch      volts per octave, zero at C4 — reference this from `voice.pitch`
  gate       1 while the key is held
  velocity   how hard it was struck, 0 to 1
  freq       the same pitch in Hz, for reading rather than playing

Every output is `[voices, 1]` rather than flat, because the audio engine reads a one-dimensional
frame as ONE channel of many samples where these are many channels of one.

A voice keeps its key until the key is released, so a held note never jumps channels when another
is pressed beside it.
"""

import numpy as np
import goofi

C4_HZ = 261.63


class MicrotonalKeys(goofi.Node):
    """Play a tuning from a MIDI keyboard: one voice per key, one step per key."""

    TAGS = ["transform", "midi"]
    INPUTS = {
        "input": goofi.InputSlot(goofi.DataType.ARRAY, required=True),
        "notes": goofi.InputSlot(goofi.DataType.ARRAY, required=True),
    }
    OUTPUTS = {
        "voices": goofi.DataType.ARRAY,
        "pitch": goofi.DataType.ARRAY,
        "gate": goofi.DataType.ARRAY,
        "velocity": goofi.DataType.ARRAY,
        "freq": goofi.DataType.ARRAY,
    }
    PARAMS = {
        "keys": {
            "voices": goofi.IntParam(8, 1, 8, doc="How many keys may sound at once. Eight is what the bundled `voices` wire holds."),
            "base_note": goofi.IntParam(12, 0, 127, doc="The key that plays the first ratio. 12 is C0."),
            "base_freq": goofi.FloatParam(16.35, 1.0, 2000.0, doc="What that key sounds at, in Hz. 16.35 is C0."),
            "octave": goofi.FloatParam(2.0, 1.1, 8.0, doc="The interval the scale repeats at; 2 is the octave."),
        }
    }

    def setup(self):
        self.held = {}  # midi note -> voice index, so a held key keeps its voice

    def process(self, input, notes):
        p = self.params.keys
        ratios = np.asarray([v for v in np.ravel(np.asarray(input.data, dtype=np.float64)) if np.isfinite(v)])
        # The octave that closes a scale is the next scale's first step, so it is not a step here —
        # otherwise every octave would sound twice and the key count would drift.
        ratios = ratios[ratios < p.octave - 1e-9]
        if ratios.size == 0:
            raise ValueError("MicrotonalKeys needs at least one ratio below the octave")

        vel = np.ravel(np.asarray(notes.data, dtype=np.float64))
        down = [n for n in range(min(vel.size, 128)) if vel[n] > 0.0]

        # A key keeps the voice it was given; a released one frees its voice for the next press.
        self.held = {n: v for n, v in self.held.items() if n in down}
        free = [v for v in range(p.voices) if v not in self.held.values()]
        for n in down:
            if n not in self.held and free:
                self.held[n] = free.pop(0)

        pitch = np.zeros(p.voices, dtype=np.float64)
        gate = np.zeros(p.voices, dtype=np.float64)
        veloc = np.zeros(p.voices, dtype=np.float64)
        freq = np.zeros(p.voices, dtype=np.float64)
        n_steps = ratios.size
        for note, v in self.held.items():
            step = note - p.base_note
            # Floor division carries the octave BELOW the base note too, so the scale runs down as
            # far as the keyboard does rather than folding at zero.
            rung, degree = divmod(step, n_steps)
            hz = p.base_freq * (p.octave ** rung) * ratios[degree]
            freq[v] = hz
            pitch[v] = np.log2(hz / C4_HZ)
            gate[v] = 1.0
            veloc[v] = vel[note]

        col = lambda a: a.astype(np.float32).reshape(-1, 1)
        bundle = np.concatenate([pitch, veloc])
        return {"voices": col(bundle), "pitch": col(pitch), "gate": col(gate), "velocity": col(veloc), "freq": col(freq)}
