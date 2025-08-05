import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, IntParam, StringParam


class TimeDelayEmbedding(Node):
    """
    This node performs time delay embedding on an input array. Time delay embedding reconstructs the state space of a signal by creating multiple delayed copies of the input and stacking them together, enabling the analysis of temporal structures and dynamics within the data.

    Inputs:
    - input_array: A one-dimensional array representing the input signal to be embedded.

    Outputs:
    - embedded_array: An array where each element contains values from the original signal at different delay steps, enabling state space reconstruction, along with updated metadata.
    """

    def config_input_slots():
        return {"input_array": DataType.ARRAY}

    def config_output_slots():
        return {"embedded_array": DataType.ARRAY}

    def config_params():
        return {
            "embedding": {
                "delay": IntParam(1, 1, 100),
                "unit": StringParam(
                    "samples",
                    options=["samples", "seconds"],
                    doc=(
                        "Unit of the buffer size. If 'seconds', the metadata must specify "
                        "the sampling frequency via the 'sfreq' key."
                    ),
                ),
                "embedding_dimension": IntParam(2, 2, 100),
                "moire_embedding": False,
                "exponent": FloatParam(1.0, 0.0, 10.0, doc="Exponent for time delay embedding"),
            }
        }

    def process(self, input_array: Data):
        if input_array is None or input_array.data is None:
            return None

        delay = self.params.embedding.delay.value
        if self.params.embedding.unit.value == "seconds":
            if "sfreq" not in input_array.meta:
                raise ValueError("If unit is 'seconds', the metadata must contain 'sfreq'.")
            delay = int(delay * input_array.meta["sfreq"])

        embedding_dimension = self.params.embedding.embedding_dimension.value
        moire_embedding = self.params.embedding.moire_embedding.value

        array = input_array.data
        arrays = [array]
        exponent = self.params.embedding.exponent.value
        max_shift = int((embedding_dimension - 1) ** exponent * delay)

        # Generating delayed versions
        for i in range(1, embedding_dimension):
            shift = int((i**exponent) * delay)
            delayed = np.roll(array, shift=shift, axis=0)
            arrays.append(delayed)

        # prevent wrapping around
        arrays = [arr[max_shift:] for arr in arrays]

        if moire_embedding:
            embedded_array = np.stack(arrays, axis=-1)
        else:
            embedded_array = np.array(arrays)

        if "dim0" in input_array.meta["channels"]:
            input_array.meta["channels"]["dim1"] = input_array.meta["channels"]["dim0"]
            del input_array.meta["channels"]["dim0"]

        return {"embedded_array": (embedded_array, input_array.meta)}
