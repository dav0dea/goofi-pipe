"""MidiIn — what a MIDI controller is doing right now, as frames rather than events.

`notes` carries the velocity of every note being held and 0 for every note that is not, so the
note number is the INDEX and the velocity is the value; `cc` carries the last value of every
controller, in 0 to 1. `bend` is the pitch wheel in -1 to 1, centred at 0, and `pressure` is
channel aftertouch in 0 to 1 — both single numbers, because a keyboard sends one of each. All
four are the STATE rather than the events, so a node reading them never has to remember what
came before, and a wheel left off-centre keeps reading off-centre.

This is `signal:MidiIn`. The audio engine has one of its own, which drives voices instead.
"""

import threading

import mido
import numpy as np
import goofi


class MidiIn(goofi.Node):
    """Receive MIDI: every held note, every controller, the pitch wheel and aftertouch."""

    TAGS = ["input", "midi"]
    OUTPUTS = {
        "cc": goofi.DataType.ARRAY,
        "notes": goofi.DataType.ARRAY,
        "bend": goofi.DataType.ARRAY,
        "pressure": goofi.DataType.ARRAY,
    }
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
        self.bend = 0.0
        self.pressure = 0.0
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
            # The wheel is 14-bit and asymmetric — -8192 down, 8191 up — so each side is scaled by
            # its own end, or a full push down would never quite reach -1.
            elif message.type == "pitchwheel":
                self.bend = message.pitch / (8191.0 if message.pitch >= 0 else 8192.0)
            elif message.type == "aftertouch":
                self.pressure = message.value / 127.0

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
            bend, pressure = self.bend, self.pressure
        numbers = {"dim0": [str(i) for i in range(128)]}
        one = lambda v: np.asarray([v], dtype=np.float32)
        return {
            "notes": (notes, {"channels": numbers}),
            "cc": (cc, {"channels": numbers}),
            "bend": one(bend),
            "pressure": one(pressure),
        }

    def stop(self):
        if self.port is not None:
            self.port.close()
            self.port = None
