from copy import deepcopy

import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import StringParam


class Operation(Node):
    """
    This node performs a specified element-wise or matrix operation on two input arrays. It supports common arithmetic operations such as addition, subtraction, multiplication, division, matrix multiplication, maximum, minimum, average, and cosine similarity. The operation is applied to the data from the two input arrays, and the output retains the metadata from the first input, adjusting dimensions and sampling frequency as needed.

    Inputs:
    - a: The first input array, with associated metadata including dimensions and channels.
    - b: The second input array, with associated metadata.

    Outputs:
    - out: The result of applying the selected operation to the two input arrays, along with the merged and adjusted metadata.
    """

    def config_input_slots():
        return {"a": DataType.ARRAY, "b": DataType.ARRAY}

    def config_params():
        return {
            "operation": {
                "operation": StringParam(
                    "add",
                    options=["add", "subtract", "multiply", "divide", "matmul", "max", "min", "avg", "cosine_similarity"],
                    doc="Operation to perform on the input arrays",
                )
            }
        }

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def process(self, a: Data, b: Data):
        if a is None or b is None:
            return None

        operation = self.params.operation.operation.value
        if operation == "matmul":
            new_meta = deepcopy(a.meta)
            # matmul is n x m * m x p -> n x p
            if "dim0" in a.meta["channels"]:
                new_meta["channels"]["dim0"] = a.meta["channels"]["dim0"]
            elif "dim0" in new_meta["channels"]:
                del new_meta["channels"]["dim0"]

            if "dim1" in b.meta["channels"]:
                new_meta["channels"]["dim1"] = b.meta["channels"]["dim1"]
            elif "dim1" in new_meta["channels"]:
                del new_meta["channels"]["dim1"]
        else:
            if "channels" in a.meta and "channels" in b.meta:
                if a.meta["channels"] != b.meta["channels"]:
                    print("Channels are not the same, metadata from a is used")
                new_meta = deepcopy(a.meta)

        if "sfreq" in a.meta and "sfreq" in b.meta:
            if a.meta["sfreq"] != b.meta["sfreq"]:
                print("Sampling frequencies are not the same, metadata from a is used")
            new_meta["sfreq"] = a.meta["sfreq"]
        elif "sfreq" in a.meta:
            new_meta["sfreq"] = a.meta["sfreq"]
        elif "sfreq" in b.meta:
            new_meta["sfreq"] = b.meta["sfreq"]

        if operation == "add":
            result = a.data + b.data
        elif operation == "subtract":
            result = a.data - b.data
        elif operation == "multiply":
            result = a.data * b.data
        elif operation == "divide":
            result = a.data / b.data
        elif operation == "matmul":
            result = np.dot(a.data, b.data)
        elif operation == "max":
            result = np.maximum(a.data, b.data)
        elif operation == "min":
            result = np.minimum(a.data, b.data)
        elif operation == "avg":
            result = (a.data + b.data) / 2
        elif operation == "cosine_similarity":
            from sklearn.metrics.pairwise import cosine_similarity

            result = cosine_similarity(a.data, b.data)
        else:
            raise ValueError(f"Invalid operation: {operation}")

        return {"out": (result, new_meta)}
