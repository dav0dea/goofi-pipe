import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node


class LSLOut(Node):
    """
    This node outputs incoming array data as a Lab Streaming Layer (LSL) stream, allowing real-time transmission of signals (such as EEG, sensor data, etc.) to other software or machines compatible with LSL. The node automatically creates and manages an LSL outlet, configuring its channels and parameters to match the input array.

    Inputs:
    - data: A 1D or 2D array of floating-point data to be sent over LSL. The array can represent multi-channel or single-channel time series data. The expected channel names and sample frequency may be specified in metadata.

    Outputs:
    - None. (This node transmits data to an external LSL stream and does not produce an output within the goofi-pipe node graph.)
    """

    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_params():
        return {
            "lsl": {
                "source_name": "goofi",
                "stream_name": "stream",
            }
        }

    def setup(self):
        from mne_lsl import lsl

        self.lsl = lsl
        self.outlet = None

    def process(self, data: Data):
        if data is None or len(data.data) == 0:
            return

        if self.outlet is not None and self.outlet.n_channels != len(data.data):
            self.outlet = None

        if self.outlet is None:
            info = self.lsl.StreamInfo(
                self.params.lsl.stream_name.value,
                "Data",
                len(data.data),
                data.meta["sfreq"] if "sfreq" in data.meta else self.lsl.IRREGULAR_RATE,
                "float32",
                self.params.lsl.source_name.value,
            )
            if "dim0" in data.meta["channels"]:
                info.set_channel_names(data.meta["channels"]["dim0"])

            self.outlet = self.lsl.StreamOutlet(info)

        try:
            if data.data.ndim == 1:
                self.outlet.push_sample(data.data.astype(np.float32))
            elif data.data.ndim == 2:
                self.outlet.push_chunk(np.ascontiguousarray(data.data.T.astype(np.float32)))
            else:
                raise ValueError("Only one- and two-dimensional arrays are supported.")
        except Exception as e:
            self.outlet = None
            raise e

    def lsl_stream_name_changed(self, value: str):
        if self.outlet is not None:
            self.outlet = None
