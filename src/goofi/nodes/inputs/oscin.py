import numpy as np
from oscpy.server import OSCThreadServer

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, IntParam, StringParam


class OSCIn(Node):
    """
    Receives incoming OSC (Open Sound Control) messages over the network and makes them available as output. Each OSC message received is stored and organized by its address. Messages containing string data are output as strings, while other types are represented as arrays. This node acts as a bridge between OSC sources (such as sensors, controllers, or other software) and Goofi-Pipe, enabling real-time signal and data integration.

    Outputs:
    - message: A table containing the latest received OSC messages, organized by address. Each entry holds the received data, which may be a string or an array, depending on the message content.
    """

    def config_params():
        return {
            "osc": {
                "address": StringParam("0.0.0.0"),
                "port": IntParam(9000, 0, 65535),
                "keep_messages": BoolParam(True, doc="Keep all received messages"),
                "clear": BoolParam(trigger=True, doc="Clear all stored messages"),
            },
            "common": {"autotrigger": True},
        }

    def config_output_slots():
        return {"message": DataType.TABLE}

    def setup(self):
        self.server = OSCThreadServer(advanced_matching=True)
        self.server.listen(address=self.params.osc.address.value, port=self.params.osc.port.value, default=True)

        # bind to possible addresses of depth 10 (is there a better way to do this?)
        for i in range(1, 11):
            self.server.bind(b"/*" * i, self.callback, get_address=True)

        self.messages = {}

    def callback(self, address, *args):
        args = ["None".encode() if a is None else a for a in args]
        if isinstance(args[0], bytes):
            if len(args) > 1:
                raise ValueError(
                    "OSCIn currently doesn't support multiple string message per OSC address but "
                    f"received {list(map(bytes.decode, args))}... If you think this is wrong, "
                    "open an issue: https://github.com/dav0dea/goofi-pipe/issues"
                )
            val = Data(DataType.STRING, args[0].decode(), {})
        else:
            val = Data(DataType.ARRAY, np.array(args), {})

        self.messages[address.decode()] = val

    def process(self):
        if len(self.messages) == 0:
            return None

        data = self.messages

        if not self.params.osc.keep_messages.value or self.params.osc.clear.value:
            self.messages = {}

        return {"message": (data, {})}

    def osc_address_changed(self, address):
        self.server.stop_all()
        self.server.terminate_server()
        self.server.join_server()
        self.setup()

    def osc_port_changed(self, port):
        self.server.stop_all()
        self.server.terminate_server()
        self.server.join_server()
        self.setup()
