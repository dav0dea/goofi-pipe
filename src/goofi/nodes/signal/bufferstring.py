import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, IntParam, StringParam


class BufferString(Node):
    """
    This node accumulates incoming string values into a rolling buffer and outputs the concatenated result. Each new string input is split according to a chosen separator, and the resulting pieces are appended to the buffer. If the buffer exceeds its maximum size, the oldest entries are removed to maintain the limit. The output is the joined contents of the current buffer as a single string.

    Inputs:
    - val: String data to be added to the buffer.

    Outputs:
    - out: The concatenated string of buffer contents after appending the latest input.
    """

    def config_input_slots():
        return {"val": DataType.STRING}

    def config_output_slots():
        return {"out": DataType.STRING}

    def config_params():
        return {
            "buffer": {
                "size": IntParam(10, 1, 5000),
                "separator": StringParam(" ", options=["[space]", ",", "[paragraph]"]),
                "reset": BoolParam(False, trigger=True),
            }
        }

    def setup(self):
        self.buffer = []

    def process(self, val: Data):
        if val is None or val.data is None:
            return None

        if self.params.buffer.reset.value:
            # reset buffer
            self.buffer = []

        maxlen = self.params.buffer.size.value
        separator = self.params.buffer.separator.value
        if separator == "[space]":
            separator = " "
        elif separator == "[paragraph]":
            separator = "\n"

        # Split the input string into words based on the separator
        words = val.data.split(separator)

        # Add words to the buffer
        self.buffer.extend(words)

        # Ensure the buffer does not exceed the maximum length
        if len(self.buffer) > maxlen:
            self.buffer = self.buffer[-maxlen:]

        # Join the buffer into a single string
        output_string = separator.join(self.buffer)

        return {"out": (output_string, val.meta)}
