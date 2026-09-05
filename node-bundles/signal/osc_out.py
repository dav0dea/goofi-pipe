"""OscOut — sends a table as Open Sound Control messages, one per entry.

A key becomes an address under `prefix`, and a table inside the table nests the address, so
`{"eeg": {"alpha": 0.4}}` under `/goofi` is sent to `/goofi/eeg/alpha`. A bundle sends the whole
table with one timestamp, which is what a receiver needs to read the values as one moment.

The default host is `127.0.0.1` rather than `localhost`, which resolves to the IPv6 address
first: `OscIn` listens on IPv4, so the pair would not reach each other out of the box.
"""

import numpy as np
from pythonosc.osc_bundle_builder import IMMEDIATELY, OscBundleBuilder
from pythonosc.osc_message_builder import OscMessageBuilder
from pythonosc.udp_client import UDPClient
import goofi


class OscOut(goofi.Node):
    """Send a table as OSC messages, one address per entry."""

    TAGS = ["output"]
    INPUTS = {"input": goofi.InputSlot(goofi.DataType.TABLE, required=True)}
    PARAMS = {
        "osc": {
            "host": goofi.StringParam("127.0.0.1", doc="Where to send to."),
            "port": goofi.IntParam(8000, 1, 65535, doc="The port to send to."),
            "prefix": goofi.StringParam("/goofi", doc="The address every key hangs under."),
            "bundle": goofi.BoolParam(False, doc="Send the whole table at once, under one timestamp."),
        }
    }

    def setup(self):
        self.client = None
        self.sending_to = None

    def process(self, input):
        p = self.params.osc
        if (p.host, p.port) != self.sending_to:
            self.client = UDPClient(p.host, p.port)
            self.sending_to = (p.host, p.port)

        messages = []
        walk(input.table, "/" + p.prefix.strip("/"), messages)
        if p.bundle:
            bundle = OscBundleBuilder(IMMEDIATELY)
            for m in messages:
                bundle.add_content(m)
            self.client.send(bundle.build())
        else:
            for m in messages:
                self.client.send(m)
        return None


def walk(table, address, into):
    """Every leaf of the table as one message, its address the path that reached it."""
    for key, value in table.items():
        here = f"{address}/{key}"
        if value.kind == "TABLE":
            walk(value.table, here, into)
            continue
        builder = OscMessageBuilder(address=here)
        if value.kind == "STRING":
            builder.add_arg(value.text)
        else:
            for v in np.asarray(value.data, dtype=np.float32).ravel():
                builder.add_arg(float(v))
        into.append(builder.build())
