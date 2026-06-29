from copy import deepcopy

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import StringParam


class Reshape(Node):
    """
    Reshapes an input array to a specified shape. The output array will have the new shape
    while preserving the original data. Channel names can't survive an arbitrary reshape, so
    they are dropped from the output (other metadata, e.g. sfreq, is preserved).

    Inputs:
    - array: The array to be reshaped.

    Outputs:
    - out: The reshaped array with updated metadata.
    """

    def config_input_slots():
        return {"array": DataType.ARRAY}

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def config_params():
        return {"reshape": {"shape": StringParam("-1")}}

    def process(self, array: Data):
        if array is None:
            return None

        shape = list(map(int, self.params.reshape.shape.value.split(",")))
        result = array.data.reshape(shape)

        # Channel names don't survive an arbitrary reshape; drop them on a COPY so the
        # producer's meta is never mutated (the old `del` aliased it across fan-out).
        meta = deepcopy(array.meta)
        meta.pop("channels", None)
        return {"out": (result, meta)}
