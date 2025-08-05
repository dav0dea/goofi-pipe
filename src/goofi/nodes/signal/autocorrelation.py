import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, IntParam


class Autocorrelation(Node):
    """
    Computes the autocorrelation of input array signals along a specified axis. The autocorrelation measures the similarity of a signal with delayed versions of itself, and can reveal repeating patterns or periodicity in data.

    Inputs:
    - signal: Input array (may be one-dimensional or multi-dimensional) to compute the autocorrelation from.

    Outputs:
    - autocorr: The resulting autocorrelation array, with the same or reduced dimensionality depending on processing.
    """

    def config_input_slots():
        return {"signal": DataType.ARRAY}

    def config_output_slots():
        return {"autocorr": DataType.ARRAY}

    def config_params():
        return {
            "autocorrelation": {
                "normalize": BoolParam(True, doc="Normalize the autocorrelation result"),
                "biased": BoolParam(False, doc="Use biased (divide by N) or unbiased (divide by N-lag) estimator"),
                "cutoff": IntParam(-1, -500, -1, doc="Cut off the autocorrelation result at this lag (use -1 for no cutoff)"),
                "axis": IntParam(-1, doc="Axis along which to compute the autocorrelation"),
            },
        }

    def process(self, signal: Data):
        x = signal.data
        if x is None or x.size == 0:
            return None

        axis = self.params.autocorrelation.axis.value
        axis = axis if axis >= 0 else x.ndim + axis

        N = x.shape[axis]

        def autocorr_func(m):
            result = np.correlate(m, m, mode="full")
            return result[result.size // 2 :]

        autocorr = np.apply_along_axis(autocorr_func, axis=axis, arr=x)

        # Apply unbiased or biased normalization
        if self.params.autocorrelation.biased.value:
            autocorr = autocorr / N
        else:
            lags = np.arange(N, 0, -1)
            shape = [1] * autocorr.ndim
            shape[axis] = -1
            autocorr = autocorr / lags.reshape(shape)

        # Normalize if requested
        if self.params.autocorrelation.normalize.value:
            norm_factor = np.take(autocorr, indices=0, axis=axis)
            norm_factor = np.where(norm_factor == 0, 1, norm_factor)
            autocorr = autocorr / np.expand_dims(norm_factor, axis=axis)

        # Apply cutoff if set
        cutoff = self.params.autocorrelation.cutoff.value
        if cutoff != -1:
            slices = [slice(None)] * autocorr.ndim
            slices[axis] = slice(0, cutoff)
            autocorr = autocorr[tuple(slices)]

        if "channels" in signal.meta and f"dim{axis}" in signal.meta["channels"]:
            del signal.meta["channels"][f"dim{axis}"]
        return {"autocorr": (autocorr, signal.meta)}
