"""OscIn — receives Open Sound Control messages, as one table entry per address.

The server runs on its own thread and the node reads what it has collected, so nothing blocks a
tick. Latest wins: an address that arrives twice between two reads keeps only what came last. A
message carrying one string is a String; anything else is an Array of its numbers.

The address is the path into the table, so `/goofi/eeg/alpha` arrives as `goofi.eeg.alpha` —
the same shape `OscOut` sends from.
"""

import threading

import numpy as np
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer
import goofi


class OscIn(goofi.Node):
    """Receive OSC messages: one table entry per address, the latest value each."""

    TAGS = ["input"]
    OUTPUTS = {"out": goofi.DataType.TABLE}
    PRODUCER = True
    PARAMS = {
        "osc": {
            "host": goofi.StringParam("0.0.0.0", doc="The address to listen on."),
            "port": goofi.IntParam(9000, 1, 65535, doc="The port to listen on."),
            "clear": goofi.PulseParam(doc="Forget every address collected so far."),
        }
    }

    def setup(self):
        self.lock = threading.Lock()
        self.latest = {}
        self.server = None
        self.bound = None

    def pulse_osc_clear(self):
        with self.lock:
            self.latest = {}

    def receive(self, address, *args):
        with self.lock:
            self.latest[address] = args

    def process(self):
        p = self.params.osc
        if (p.host, p.port) != self.bound:
            self.stop()
            dispatcher = Dispatcher()
            dispatcher.set_default_handler(self.receive)
            self.server = ThreadingOSCUDPServer((p.host, p.port), dispatcher)
            threading.Thread(target=self.server.serve_forever, daemon=True).start()
            self.bound = (p.host, p.port)

        with self.lock:
            held = dict(self.latest)
        if not held:
            return None
        tree = {}
        for address, args in held.items():
            parts = [part for part in address.strip("/").split("/") if part]
            here = tree
            for part in parts[:-1]:
                if not isinstance(here.get(part), dict):
                    here[part] = {}
                here = here[part]
            here[parts[-1]] = entry(args)
        return {"out": tree}

    def stop(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None


def entry(args):
    """One message's arguments as a frame: a lone string stays text, everything else is numbers."""
    if len(args) == 1 and isinstance(args[0], str):
        return args[0]
    return np.asarray([float(a) for a in args], dtype=np.float32)
