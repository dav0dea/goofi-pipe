import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, IntParam, StringParam


class Reservoir(Node):
    """
    This node implements a recurrent neural reservoir module. It maintains an internal state vector that evolves over time according to a configurable nonlinear transformation and a connectivity (weight) matrix. At each step, the state is updated by multiplying it with the connectivity matrix, optionally adding a bias, and applying a nonlinear activation function. This dynamic transformation is typically useful for tasks involving temporal or sequential data processing.

    Inputs:
    - connectivity: An optional square connectivity matrix (as an array) matching the size of the reservoir. If provided, it determines how each node in the reservoir is connected to others.

    Outputs:
    - data: The current state vector of the reservoir, after updating and applying the activation function. The output includes the array and associated sample frequency information.
    """

    def config_params():
        return {
            "reservoir": {
                "size": IntParam(100, 2, 1000),
                "function": StringParam("tanh", options=["tanh", "sigmoid"]),
                "add_bias": True,
                "mean": FloatParam(0.0, -0.2, 0.2),
                "std": FloatParam(1.0, 0.7, 1.3),
            },
            "common": {"autotrigger": True},
        }

    def config_input_slots():
        return {"connectivity": DataType.ARRAY}

    def config_output_slots():
        return {"data": DataType.ARRAY}

    def setup(self):
        self.reservoir = np.zeros(self.params.reservoir.size.value)
        self.weights = np.random.uniform(-1, 1, (self.params.reservoir.size.value, self.params.reservoir.size.value))
        self.bias = np.random.uniform(-1, 1, self.params.reservoir.size.value)

    def process(self, connectivity: Data):
        if connectivity is None:
            w = self.weights
        else:
            assert connectivity.data.ndim == 2, "Connectivity matrix must be two-dimensional."
            assert connectivity.data.shape[0] == connectivity.data.shape[1], "Connectivity matrix must be square."
            assert (
                connectivity.data.shape[0] == self.params.reservoir.size.value
            ), "Connectivity matrix must have the same size as the reservoir."
            conn = connectivity.data
            conn[np.diag_indices_from(conn)] = 0
            w = conn

        # scale the weights
        w = w * self.params.reservoir.std.value + self.params.reservoir.mean.value

        self.reservoir = np.dot(w, self.reservoir)
        if self.params.reservoir.add_bias.value:
            self.reservoir += self.bias

        if self.params.reservoir.function.value == "tanh":
            self.reservoir = np.tanh(self.reservoir)
        elif self.params.reservoir.function.value == "sigmoid":
            self.reservoir = 1 / (1 + np.exp(-self.reservoir))
        else:
            raise ValueError(f"Unsupported function {self.params.reservoir.function.value} for reservoir output.")

        # reshape into matrix of size greatest common divisor of reservoir size
        return {"data": (self.reservoir, {"sfreq": self.params.common.max_frequency.value})}

    def reservoir_size_changed(self, value):
        self.setup()
