import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, IntParam, StringParam


class Padding(Node):
    """
    Pads an input array with values along specified dimensions. Supports various padding modes including constant, edge, reflect, and wrap.

    Inputs:
    - array: The input array to be padded.

    Outputs:
    - out: The padded array with updated dimensions.
    """

    def config_input_slots():
        return {"array": DataType.ARRAY}

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def config_params():
        return {
            "padding": {
                "pad_before": IntParam(0, 0, 1000),
                "pad_after": IntParam(0, 0, 1000),
                "target_size": IntParam(0, 0, 10000),
                "axis": IntParam(-1, -10, 10),
                "mode": StringParam("constant"),
                "constant_value": FloatParam(0.0),
            }
        }

    def process(self, array: Data):
        if array is None:
            return None

        pad_before = self.params.padding.pad_before.value
        pad_after = self.params.padding.pad_after.value
        target_size = self.params.padding.target_size.value
        axis = self.params.padding.axis.value
        mode = self.params.padding.mode.value
        constant_value = self.params.padding.constant_value.value

        # Handle negative axis
        ndim = array.data.ndim
        if axis < 0:
            axis = ndim + axis

        current_size = array.data.shape[axis]

        # If target_size is set (> 0), calculate padding to reach that size
        if target_size > 0:
            if target_size > current_size:
                total_pad = target_size - current_size
                # Distribute padding: add to the end by default
                pad_before = 0
                pad_after = total_pad
            else:
                # Target size is smaller or equal, no padding needed
                pad_before = 0
                pad_after = 0

        # Build pad_width tuple for np.pad
        pad_width = [(0, 0)] * ndim
        pad_width[axis] = (pad_before, pad_after)

        # Apply padding
        if mode == "constant":
            result = np.pad(array.data, pad_width, mode=mode, constant_values=constant_value)
        else:
            result = np.pad(array.data, pad_width, mode=mode)

        # Update metadata channels if they exist
        meta = array.meta.copy()
        if "channels" in meta and meta["channels"] is not None:
            channels = meta["channels"]
            if isinstance(channels, dict):
                # Check if this axis has channel names
                axis_key = f"dim{axis}"
                if axis_key in channels and channels[axis_key] is not None:
                    original_names = list(channels[axis_key])
                    # Add placeholder names for padded elements
                    before_names = [f"pad_{i}" for i in range(pad_before)]
                    after_names = [f"pad_{len(original_names) + i}" for i in range(pad_after)]
                    channels[axis_key] = before_names + original_names + after_names
                meta["channels"] = channels

        return {"out": (result, meta)}
