from goofi.data import Data, DataType
from goofi.node import Node


class NullString(Node):
    """
    This node receives a string input and passes it through unchanged to the output. It does not alter the string data or its associated metadata.

    Inputs:
    - string_in: The input string data to be passed through.

    Outputs:
    - string_out: The output string data, identical to the input.
    """

    def config_input_slots():
        return {"string_in": DataType.STRING}

    def config_output_slots():
        return {"string_out": DataType.STRING}

    def process(self, string_in: Data):
        return {"string_out": (string_in.data, string_in.meta)}
