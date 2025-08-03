from collections import deque

import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, FloatParam, IntParam, StringParam


class Filter(Node):
    """
    This node applies digital signal filtering to incoming data arrays, allowing for the removal or attenuation of specific frequency components. The node supports real-time bandpass and notch filtering, with options for causal or zero-phase filters. It can also perform detrending and demeaning operations on the signal. Internally, it can buffer incoming data to enable more robust filtering.

    Inputs:
    - data: An array of numerical signal data to be filtered. The data should include associated metadata such as the sampling frequency ("sfreq").

    Outputs:
    - filtered_data: The filtered version of the input data array, returned with original metadata. This output contains the signal after the selected filtering and optional preprocessing steps have been applied.
    """

    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {"filtered_data": DataType.ARRAY}

    def config_params():
        return {
            "bandpass": {
                "apply": BoolParam(False),
                "type": StringParam("butterworth", options=["butterworth", "chebyshev", "elliptic"]),
                "method": StringParam(
                    "Causal",
                    options=["Causal", "Zero-phase"],
                    doc="Zero-phase refers to applying the filter twice, once forward and once backward",
                ),
                "order": IntParam(3, 1, 10, doc="Order of the bandpass filter"),
                "f_low": FloatParam(1.0, 0.01, 9999.0),
                "f_high": FloatParam(50.0, 1.0, 10000.0),
                "ripple": FloatParam(1.0, 0.1, 10.0, doc="Ripple refers to the maximum loss in the passband of the filter"),
                "padding": FloatParam(
                    0.1,
                    0.01,
                    1.0,
                    doc="Padding refers to the fraction of the signal to pad at the beginning and end of the signal",
                ),
            },
            "notch": {
                "apply": BoolParam(False),
                "type": StringParam("butterworth", options=["butterworth", "chebyshev", "elliptic"]),
                "method": StringParam(
                    "Causal",
                    options=["Causal", "Zero-phase"],
                    doc="Zero-phase refers to applying the filter twice, once forward and once backward",
                ),
                "order": IntParam(1, 1, 4, doc="Order of the notch filter"),
                "f_center": FloatParam(60.0, 0.01, 10000.0, doc="Center frequency of the notch filter"),
                "Q": FloatParam(10.0, 0.1, 30.0, doc="Intensity of the notch filter"),
                "ripple": FloatParam(1.0, 0.1, 10.0, doc="Ripple refers to the maximum loss in the passband of the filter"),
                "padding": FloatParam(
                    0.1,
                    0.01,
                    1.0,
                    doc="Padding refers to the fraction of the signal to pad at the beginning and end of the signal",
                ),
            },
            "signal": {
                "detrend": False,
                "demean": False,
                "internal_buffer": False,
                "buffer_size": 512,
            },
        }

    def setup(self):
        from scipy import signal

        self.signal = signal

        self.filter_state_bp = None
        self.filter_state_notch = None
        self.internal_buffer = deque(maxlen=self.params["signal"]["buffer_size"].value)

    def process(self, data: Data):
        if data is None or data.data is None:
            return None

        sfreq = data.meta["sfreq"]
        nyq = 0.5 * sfreq

        self.internal_buffer.extend(data.data.T)
        if self.params["signal"]["internal_buffer"].value:
            filtered_data = np.array(self.internal_buffer).T
        else:
            filtered_data = data.data

        # Bandpass Filtering
        if self.params["bandpass"]["apply"].value:
            filter_type = self.params["bandpass"]["type"].value
            f_low = self.params["bandpass"]["f_low"].value
            f_high = self.params["bandpass"]["f_high"].value
            ripple = self.params["bandpass"]["ripple"].value
            padding = self.params["bandpass"]["padding"].value
            method = self.params["bandpass"]["method"].value
            order = self.params["bandpass"]["order"].value
            low = f_low / nyq
            high = f_high / nyq

            if filter_type == "butterworth":
                b, a = self.signal.butter(order, [low, high], btype="band")
            elif filter_type == "chebyshev":
                b, a = self.signal.cheby1(order, ripple, [low, high], btype="band")
            elif filter_type == "elliptic":
                b, a = self.signal.ellip(order, ripple, ripple, [low, high], btype="band")

            filtered_data = self.apply_filter(b, a, filtered_data, method, padding, "bandpass")

        # Notch Filtering
        if self.params["notch"]["apply"].value:
            filter_type = self.params["notch"]["type"].value
            f_center = self.params["notch"]["f_center"].value
            ripple = self.params["notch"]["ripple"].value
            padding = self.params["notch"]["padding"].value
            method = self.params["notch"]["method"].value
            order = self.params["notch"]["order"].value
            Q = self.params["notch"]["Q"].value
            w0 = f_center / nyq
            bw = f_center / Q
            bw /= nyq

            w_low = np.clip(w0 - bw / 2, 0.01, 0.99)
            w_high = np.clip(w0 + bw / 2, 0.01, 0.99)

            if filter_type == "butterworth":
                b, a = self.signal.butter(order, [w_low, w_high], btype="bandstop")
            elif filter_type == "chebyshev":
                b, a = self.signal.cheby1(order, ripple, [w_low, w_high], btype="bandstop")
            elif filter_type == "elliptic":
                b, a = self.signal.ellip(order, ripple, ripple, [w_low, w_high], btype="bandstop")

            filtered_data = self.apply_filter(b, a, filtered_data, method, padding, "notch")

        if self.params["signal"]["internal_buffer"].value:
            filtered_data = filtered_data[..., -data.data.shape[-1] :]

        # Detrend and demean
        if self.params["signal"]["detrend"].value:
            filtered_data = self.signal.detrend(filtered_data, type="linear")
        if self.params["signal"]["demean"].value:
            filtered_data = self.signal.detrend(filtered_data, type="constant")

        return {"filtered_data": (filtered_data, data.meta)}

    def apply_filter(self, b, a, data, method, padding, filter_type):
        if method == "Zero-phase":
            if data.ndim == 1:
                padlen = int(padding * len(data))
            else:
                padlen = int(padding * data.shape[-1])
            return self.signal.filtfilt(b, a, data, padlen=padlen)
        elif method == "Causal":
            if filter_type == "bandpass":
                filter_state = self.filter_state_bp
            elif filter_type == "notch":
                filter_state = self.filter_state_notch

            try:
                if data.ndim == 1:
                    zi = self.signal.lfilter_zi(b, a)
                    if filter_state is None or filter_state.shape != zi.shape:
                        filter_state = zi * data[0]
                    filtered_data, new_state = self.signal.lfilter(b, a, data, zi=filter_state)
                elif data.ndim == 2:
                    num_channels = data.shape[0]
                    zi = self.signal.lfilter_zi(b, a).reshape(-1, 1)
                    if filter_state is None or filter_state.shape[1] != num_channels:
                        filter_state = np.tile(zi, (1, num_channels)) * data[:, 0]
                    filtered_data, new_state = self.signal.lfilter(b, a, data.T, axis=0, zi=filter_state)
                    filtered_data = filtered_data.T
            except:
                raise
            finally:
                if filter_type == "bandpass":
                    self.filter_state_bp = None
                elif filter_type == "notch":
                    self.filter_state_notch = None

            # Update the appropriate filter state
            if filter_type == "bandpass":
                self.filter_state_bp = new_state
            elif filter_type == "notch":
                self.filter_state_notch = new_state

            return filtered_data
        else:
            return data

    def signal_buffer_size_changed(self, buffer_size):
        self.internal_buffer = deque(maxlen=buffer_size)
