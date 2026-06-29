import threading

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
        # The address/port param callbacks call setup() on the messaging thread, closing +
        # recreating the socket while process() may be using it on the processing thread.
        # zmq sockets are NOT thread-safe, so guard every socket touch with one lock (created
        # once, kept across setup() calls so both threads share the same lock).
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
            self.socket.connect(f"tcp://{self.params.zero_mq.address.value}:{self.params.zero_mq.port.value}")

    def process(self):
        # Poll briefly (non-blocking) under the lock instead of a blocking recv: holding the
        # lock during a blocking recv_pyobj() would wedge the node when no peer is sending and
        # deadlock every address/port change. Returns None when nothing is ready this tick
        # (the node self-triggers via autotrigger, so it polls again immediately).
        with self._lock:
            if not self.socket.poll(timeout=100):
                return None
            try:
                data = self.socket.recv_pyobj(flags=zmq.NOBLOCK)
            except zmq.Again:
                return None
        return {"data": (data, {})}

    def zero_mq_address_changed(self, value):
        # TODO: make sure socket stuff only happens on the main thread
        self.setup()

    def zero_mq_port_changed(self, value):
        # TODO: make sure socket stuff only happens on the main thread
        self.setup()

    def terminate(self):
        # Close the socket + context so the connected fd isn't leaked across shutdown.
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
