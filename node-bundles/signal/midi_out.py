"""MidiOut — sends MIDI for the notes being held and for one controller value.

`notes` is read as the set of note numbers held RIGHT NOW: a number that appears sends note-on
and a number that disappears sends note-off, so nothing is sent twice and nothing hangs. What is
still held is released when the node stops.
"""

import mido
import numpy as np
import goofi


class MidiOut(goofi.Node):
    """Send MIDI: note-on for a number that appears, note-off for one that goes."""

    TAGS = ["output", "midi"]
    INPUTS = {
        "notes": goofi.InputSlot(goofi.DataType.ARRAY),
        "value": goofi.InputSlot(goofi.DataType.ARRAY),
    }
    PARAMS = {
        "midi": {
            "port": goofi.StringParam("", options=[""], refresh=True, doc="Which port to send on."),
            "channel": goofi.IntParam(1, 1, 16, doc="Which channel to send on."),
            "velocity": goofi.IntParam(100, 1, 127, doc="How hard every note is struck."),
            "controller": goofi.IntParam(1, 0, 127, doc="Which controller `value` is sent as."),
        }
    }

    def setup(self):
        self.port = None
        self.opened = None
        self.held = set()
        self.last_cc = None

    def refresh_midi_port(self):
        return [""] + sorted(mido.get_output_names())

    def process(self, notes, value):
        p = self.params.midi
        if p.port != self.opened:
            self.stop()
            if p.port:
                self.port = mido.open_output(p.port)
            self.opened = p.port
        if self.port is None:
            return None
        channel = p.channel - 1

        if notes is not None:
            wanted = {int(round(n)) for n in np.asarray(notes.data).ravel() if 0 <= n <= 127}
            for note in sorted(self.held - wanted):
                self.port.send(mido.Message("note_off", note=note, channel=channel))
            for note in sorted(wanted - self.held):
                self.port.send(mido.Message("note_on", note=note, velocity=p.velocity, channel=channel))
            self.held = wanted

        if value is not None:
            level = int(round(float(np.asarray(value.data).ravel()[0]) * 127))
            level = max(0, min(127, level))
            if level != self.last_cc:
                self.port.send(mido.Message("control_change", control=p.controller, value=level, channel=channel))
                self.last_cc = level
        return None

    def stop(self):
        if self.port is not None:
            for note in sorted(self.held):
                self.port.send(mido.Message("note_off", note=note, channel=self.params.midi.channel - 1))
            self.port.close()
            self.port = None
        self.held = set()
