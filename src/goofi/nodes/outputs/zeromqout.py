import threading

import numpy as np
import zmq

from goofi.data import Data, DataType
from goofi.node import Node


class ZeroMQOut(Node):
    """
    This node sends array data to an external application or process over a network connection using the ZeroMQ library. It transmits data in real time via a TCP socket, allowing integration with remote systems or distributed processing setups.

    Inputs:
    - data: The array data to be transmitted, which is sent as a NumPy float32 array.

    Outputs:
    - None.
    """

    def config_params():
        return {"zero_mq": {"address": "127.0.0.1", "port": 6543}}

    def config_input_slots():
        return {"data": DataType.ARRAY}

    def setup(self):
        # The address/port param callbacks call setup() on the messaging thread, closing +
        # recreating the socket while process() may be sending on the processing thread. zmq
        # sockets are NOT thread-safe, so guard every socket touch with one shared lock
        # (created once, kept across setup() calls).
        if not hasattr(self, "_lock"):
            self._lock = threading.RLock()
        with self._lock:
            if not hasattr(self, "context"):
                self.context = zmq.Context()

            if hasattr(self, "socket"):
                try:
                    self.socket.close(linger=0)
                except Exception:
                    pass

            self.socket = self.context.socket(zmq.PAIR)
            self.socket.bind(f"tcp://{self.params.zero_mq.address.value}:{self.params.zero_mq.port.value}")

    def process(self, data: Data):
        if data is None:
            return
        data = data.data.astype(np.float32)
        # Serialize the send against a concurrent setup() (socket recreate) — see setup().
        with self._lock:
            self.socket.send_pyobj(data)

    def zero_mq_address_changed(self, value):
        # TODO: make sure socket stuff only happens on the main thread
        self.setup()

    def zero_mq_port_changed(self, value):
        # TODO: make sure socket stuff only happens on the main thread
        self.setup()

    def terminate(self):
        # Close the socket + context so the bound port/fd isn't leaked across shutdown.
        if getattr(self, "socket", None) is not None:
            try:
                self.socket.close(linger=0)
            except Exception:
                pass
        if getattr(self, "context", None) is not None:
            try:
                self.context.term()
            except Exception:
                pass
