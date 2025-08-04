import socket
from threading import Thread, current_thread
from typing import Any, Dict, Tuple

import numpy as np
from tabulate import tabulate

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam


class LSLClient(Node):
    """
    This node connects to a Lab Streaming Layer (LSL) stream and receives real-time data from it. It discovers available LSL streams on the network, connects to the specified source and stream, reads chunks of incoming data, and outputs this data along with relevant metadata such as channel names and sampling frequency. The node is suitable for live signal acquisition from any source that publishes data via LSL.

    Inputs:
    - source_name: The LSL source ID to connect to.
    - stream_name: The LSL stream name within the specified source.

    Outputs:
    - out: The acquired data as an array, along with metadata including sampling frequency and channel names.
    """

    def config_params():
        return {
            "lsl_stream": {
                "source_name": "goofi",
                "stream_name": "",
                "refresh": BoolParam(False, trigger=True),
            },
            "common": {"autotrigger": True},
        }

    def config_input_slots():
        return {"source_name": DataType.STRING, "stream_name": DataType.STRING}

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def setup(self):
        """Initialize and start the LSL client."""
        import pylsl

        self.pylsl = pylsl

        if hasattr(self, "client"):
            self.disconnect()

        self.client = None
        self.lsl_discover_thread = None
        self.ch_names = None

        if not hasattr(self, "available_streams"):
            self.available_streams = None

        # initialize list of streams
        self.connect()

    def process(self, source_name: Data, stream_name: Data) -> Dict[str, Tuple[np.ndarray, Dict[str, Any]]]:
        """Fetch the next chunk of data from the client."""
        if source_name is not None:
            self.params.lsl_stream.source_name.value = source_name.data
            self.lsl_stream_source_name_changed(source_name.data)
            self.input_slots["source_name"].clear()
        if stream_name is not None:
            self.params.lsl_stream.stream_name.value = stream_name.data
            self.lsl_stream_stream_name_changed(stream_name.data)
            self.input_slots["stream_name"].clear()

        if self.client is None:
            if not self.connect():
                return None

        try:
            # fetch data
            samples, _ = self.client.pull_chunk()
        except Exception as e:
            print(f"Error fetching data from LSL stream: {e}")
            self.setup()
            return

        samples = np.array(samples).T

        if samples.size == 0:
            return

        try:
            ch_info = self.client.info().desc().child("channels").child("channel")
            ch_type = self.client.info().type().lower()
            ch_names = []
            for k in range(1, self.client.info().channel_count() + 1):
                ch_names.append(ch_info.child_value("label") or "{} {:03d}".format(ch_type.upper(), k))
                ch_info = ch_info.next_sibling()
            self.ch_names = ch_names
        except Exception as e:
            print(f"Error fetching channel names from LSL stream: {e}")
            self.setup()
            return

        meta = {
            "sfreq": self.client.info().nominal_srate(),
            "channels": {"dim0": self.ch_names},
        }
        return {"out": (samples, meta)}

    def connect(self) -> bool:
        """Connect to the LSL stream."""
        if self.client is not None:
            self.disconnect()
        if self.available_streams is None:
            self.lsl_stream_refresh_changed(True)

        # find the stream
        source_name = self.params.lsl_stream.source_name.value
        stream_name = self.params.lsl_stream.stream_name.value

        matches = {}
        for info in self.available_streams:
            h, s, n = info.hostname(), info.source_id(), info.name()
            if s == source_name and (len(stream_name) == 0 or n == stream_name):
                if (s, n) in matches and h == socket.gethostname():
                    # prefer local streams
                    matches[(s, n)] = info
                elif (s, n) not in matches:
                    # otherwise, prefer the first match
                    matches[(s, n)] = info

        if len(matches) != 1:
            if self.lsl_discover_thread is None:
                # check if new streams arrived
                self.lsl_discover_thread = Thread(
                    target=self.lsl_stream_refresh_changed, args=(True,), daemon=True, name="lsl_discover_thread"
                )
                self.lsl_discover_thread.start()

                if len(matches) == 0:
                    print(f'\nCould not find source "{source_name}" with stream "{stream_name}".')
                else:
                    ms = tabulate(
                        [list(m) for m in matches],
                        headers=["Source ID", "Stream Name"],
                        tablefmt="simple_outline",
                    )
                    print(f'\nFound multiple streams matching source="{source_name}", name="{stream_name}":\n{ms}.')
            return False

        # connect to the stream
        self.client = self.pylsl.StreamInlet(info=list(matches.values())[0], recover=True)
        return True

    def disconnect(self) -> None:
        """Disconnect from the LSL stream."""
        if self.client is not None:
            try:
                self.client.close_stream()
            except:
                pass
            self.client = None

    def lsl_stream_refresh_changed(self, value: bool) -> None:
        self.available_streams = self.pylsl.resolve_streams()
        stream_data = sorted(
            [[info.source_id(), info.name(), info.hostname()] for info in self.available_streams], key=lambda x: x[0]
        )

        print("\nAvailable LSL streams:")
        print(tabulate(stream_data, headers=["Source ID", "Stream Name", "Host Name"], tablefmt="simple_outline"))
        print()

        if current_thread().name == "lsl_discover_thread":
            self.lsl_discover_thread = None

    def lsl_stream_source_name_changed(self, value: str) -> None:
        try:
            if self.client is not None and value != self.client.info().source_id():
                self.setup()
        except:
            # stream might have been lost
            self.setup()

    def lsl_stream_stream_name_changed(self, value: str) -> None:
        try:
            if self.client is not None and value != self.client.info().name():
                self.setup()
        except:
            # stream might have been lost
            self.setup()
