from goofi.data import Data, DataType
from goofi.node import Node


class NullArray(Node):
    """
    This node passes the input array directly to the output without modification. It is useful for situations where an explicit bypass or null operation on array data is needed.

    Inputs:
    - array_in: The input array data to be passed through unchanged.

    Outputs:
    - array_out: The unmodified array data from the input.
    """

    def config_input_slots():
        return {"array_in": DataType.ARRAY}

    def config_output_slots():
        return {"array_out": DataType.ARRAY}

    def process(self, array_in: Data):
        return {"array_out": (array_in.data, array_in.meta)}
