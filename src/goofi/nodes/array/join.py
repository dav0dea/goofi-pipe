from copy import deepcopy

import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, StringParam


class Join(Node):
    """
    This node combines two array inputs into a single array output. It supports two methods of combining: joining the arrays along an existing axis or stacking them along a new dimension. The node manages the merging or updating of metadata from both inputs as needed.

    Inputs:
    - a: The first input array and its associated metadata.
    - b: The second input array and its associated metadata.

    Outputs:
    - out: The combined array resulting from joining or stacking the two input arrays, along with updated metadata.
    """

    def config_input_slots():
        return {"a": DataType.ARRAY, "b": DataType.ARRAY}

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def config_params():
        return {"join": {"method": StringParam("concatenate", options=["concatenate", "stack"]), "axis": 0}}

    def process(self, a: Data, b: Data):
        axis = self.params.join.axis.value

        if a is not None and b is None:
            if self.params.join.method.value == "stack":
                a.data = np.expand_dims(a.data, axis=axis)
            return {"out": (a.data, a.meta)}
        elif a is None and b is not None:
            if self.params.join.method.value == "stack":
                b.data = np.expand_dims(b.data, axis=axis)
            return {"out": (b.data, b.meta)}
        elif a is None and b is None:
            return None

        result_meta = deepcopy(a.meta)
        if self.params.join.method.value == "concatenate":
            # concatenate a and b
            result = np.concatenate([a.data, b.data], axis=axis)

            axis = axis if axis >= 0 else axis + a.data.ndim
            if f"dim{axis}" in a.meta["channels"] and f"dim{axis}" in b.meta["channels"]:
                result_meta["channels"][f"dim{axis}"] = (
                    a.meta["channels"][f"dim{axis}"] + b.meta["channels"][f"dim{axis}"]
                )
        elif self.params.join.method.value == "stack":
            # stack a and b
            result = np.stack([a.data, b.data], axis=axis)

            axis = axis if axis >= 0 else axis + a.data.ndim
            for i in range(a.data.ndim, axis - 1, -1):
                if f"dim{i}" in result_meta["channels"]:
                    result_meta["channels"][f"dim{i+1}"] = result_meta["channels"].pop(f"dim{i}")

        else:
            raise ValueError(
                f"Unknown join method {self.params.join.method.value}. Supported are 'concatenate' and 'stack'."
            )

        # TODO: properly combine metadata from both inputs
        # TODO: update metadata information after stack
        # TODO: check if inputs are compatible
        return {"out": (result, result_meta)}
