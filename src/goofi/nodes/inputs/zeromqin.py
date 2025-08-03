import zmq

from goofi.data import DataType
from goofi.node import Node
from goofi.params import IntParam


class ZeroMQIn(Node):
    """
    This node receives data from a ZeroMQ socket using the PAIR pattern and provides this data as output in real time. It connects to a specified address and port, then waits for incoming Python objects sent via ZeroMQ, which it passes on for processing by other nodes.

    Outputs:
    - data: The data object received from the ZeroMQ socket, provided as an array.
    """

    def config_params():
        return {"zero_mq": {"address": "127.0.0.1", "port": IntParam(6543)}, "common": {"autotrigger": True}}

    def config_output_slots():
        return {"data": DataType.ARRAY}

    def setup(self):
        if not hasattr(self, "context"):
            self.context = zmq.Context()

        if hasattr(self, "socket"):
            try:
                self.socket.close()
            except Exception:
                pass

        # bind a publisher socket
        self.socket = self.context.socket(zmq.PAIR)
        self.socket.connect(f"tcp://{self.params.zero_mq.address.value}:{self.params.zero_mq.port.value}")

    def process(self):
        data = self.socket.recv_pyobj()
        return {"data": (data, {})}

    def zero_mq_address_changed(self, value):
        # TODO: make sure socket stuff only happens on the main thread
        self.setup()

    def zero_mq_port_changed(self, value):
        # TODO: make sure socket stuff only happens on the main thread
        self.setup()
