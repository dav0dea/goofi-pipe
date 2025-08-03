import socket  # For setting broadcast options
from typing import Any, List, Tuple

from oscpy.client import send_bundle, send_message

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, IntParam


class OSCOut(Node):
    """
    This node sends incoming table data as OSC (Open Sound Control) messages to a specified network address and port. Input data is serialized to OSC message bundles or individual messages and transmitted, with options for broadcasting and message change detection.

    Inputs:
    - data: Table data to be transmitted as OSC messages.

    Outputs:
    - None.
    """

    def config_input_slots():
        return {"data": DataType.TABLE}

    def config_params():
        return {
            "osc": {
                "address": "localhost",
                "port": IntParam(8000, 0, 65535),
                "prefix": "/goofi",
                "bundle": BoolParam(False, doc="Some software doesn't deal well with OSC bundles"),
                "broadcast": BoolParam(False, doc="Enable broadcasting OSC messages"),
                "require_change": BoolParam(False, doc="Only send OSC messages when the data changes"),
            }
        }

    def setup(self):
        self.sock = None
        self.last_messages = {}

    def process(self, data: Data):
        if data is None or len(data.data) == 0:
            return

        # determine the address and configure the socket if broadcasting is enabled
        address = self.params.osc.address.value
        port = self.params.osc.port.value
        broadcast = self.params.osc.broadcast.value

        if broadcast:
            address = "255.255.255.255"
            if self.sock is None:
                # create a socket with broadcasting enabled if needed
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        # convert the data to a list of OSC messages
        messages = generate_messages(data, self.params.osc.prefix.value)

        if self.params.osc.require_change.value and self.last_messages is not None:
            messages = [
                (addr, msg) for addr, msg in messages if addr not in self.last_messages or self.last_messages[addr] != msg
            ]

        self.last_messages.update(dict(messages))

        if self.params.osc.bundle.value:
            # send the data as an OSC bundle
            send_bundle(messages, address, port, sock=self.sock)
        else:
            # send the data as individual OSC messages
            for addr, val in messages:
                if self.sock is None:
                    send_message(addr, val, address, port)
                else:
                    send_message(addr, val, address, port, sock=self.sock)

    def osc_broadcast_changed(self, value: bool):
        if self.sock is not None:
            self.sock.close()
            self.sock = None


def generate_messages(data: Data, prefix: str = "") -> List[Tuple[bytes, List[Any]]]:
    messages = []
    for key, val in data.data.items():
        # generate the OSC address
        addr = sanitize_address(prefix + "/" + key)

        if val.dtype == DataType.ARRAY:
            # convert the array to a list
            assert val.data.ndim < 2, "Numerical arrays must at most be one-dimensional."
            val = val.data.tolist()
        elif val.dtype == DataType.STRING:
            # simply use the string
            val = val.data.encode("utf-8")
        elif val.dtype == DataType.TABLE:
            # recursively convert the table to a list of messages
            messages.extend(generate_messages(val, addr))
            continue
        else:
            raise ValueError(f"Unsupported data type {val.dtype} for OSC output.")

        # oscpy expects the message to be a list
        if not isinstance(val, list):
            val = [val]

        # add the message to the list
        messages.append((addr.encode(), val))

    return messages


def sanitize_address(address: str) -> str:
    """
    Sanitize an OSC address. This function removes leading and trailing slashes and replaces multiple slashes with a
    single slash.

    ### Parameters
    `address` : str
        The OSC address to sanitize.

    ### Returns
    str
        The sanitized OSC address.
    """
    return "/" + "/".join(a for a in address.split("/") if len(a) > 0).strip("/")
