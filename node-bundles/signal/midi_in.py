"""MidiIn — what a MIDI controller is doing right now, as two frames of 128 numbers.

`notes` carries the velocity of every note being held and 0 for every note that is not; `cc`
carries the last value of every controller, in 0 to 1. Both are the STATE rather than the events,
so a node reading them never has to remember what came before.

This is `signal:MidiIn`. The audio engine has one of its own, which drives voices instead.
"""

import threading

import mido
import numpy as np
import goofi


class MidiIn(goofi.Node):
    """Receive MIDI: the velocity of every held note, and the value of every controller."""

    TAGS = ["input", "midi"]
    OUTPUTS = {"cc": goofi.DataType.ARRAY, "notes": goofi.DataType.ARRAY}
    PRODUCER = True
    PARAMS = {
        "midi": {
            "port": goofi.StringParam("", options=[""], refresh=True, doc="Which port to listen on."),
            "channel": goofi.IntParam(0, 0, 16, doc="Which channel to listen to. 0 is every channel."),
        }
    }

    def setup(self):
        self.lock = threading.Lock()
        self.notes = np.zeros(128, dtype=np.float32)
        self.cc = np.zeros(128, dtype=np.float32)
        self.port = None
        self.opened = None

    def refresh_midi_port(self):
        return [""] + sorted(mido.get_input_names())

    def receive(self, message):
        p = self.params.midi
        if p.channel and getattr(message, "channel", p.channel - 1) != p.channel - 1:
            return
        with self.lock:
            if message.type == "note_on" and message.velocity > 0:
                self.notes[message.note] = message.velocity / 127.0
            elif message.type in ("note_off", "note_on"):
                self.notes[message.note] = 0.0
            elif message.type == "control_change":
                self.cc[message.control] = message.value / 127.0

    def process(self):
        wanted = self.params.midi.port
        if wanted != self.opened:
            self.stop()
            if wanted:
                self.port = mido.open_input(wanted, callback=self.receive)
            self.opened = wanted
        if self.port is None:
            return None
        with self.lock:
            notes, cc = self.notes.copy(), self.cc.copy()
        numbers = {"dim0": [str(i) for i in range(128)]}
        return {
            "notes": (notes, {"channels": numbers}),
            "cc": (cc, {"channels": numbers}),
        }

    def stop(self):
        if self.port is not None:
            self.port.close()
            self.port = None
