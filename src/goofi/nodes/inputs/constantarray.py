import time

import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, StringParam


class ConstantArray(Node):

    def config_params():
        return {
            "constant": {
                "value": FloatParam(1.0, -10.0, 10.0),
                "shape": "1",
                "graph": StringParam("none", options=["none", "ring", "random"]),
                "overwrite_timeout": FloatParam(
                    5,
                    0,
                    30,
                    doc="Duration within which the overwrite input data is used, revert to constant data after (0 never clears the overwrite).",
                ),
            },
            "common": {"autotrigger": True},
        }

    def config_input_slots():
        return {"overwrite": DataType.ARRAY}

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def setup(self):
        self.last_overwrite_time = None
        self.last_overwrite_data = None

    def process(self, overwrite: Data):
        if overwrite is not None:
            self.last_overwrite_data = overwrite
            self.last_overwrite_time = time.time()
            self.input_slots["overwrite"].clear()

        if self.last_overwrite_data is not None:
            timeout_val = self.params.constant.overwrite_timeout.value
            if timeout_val > 0 and (time.time() - self.last_overwrite_time) > timeout_val:
                self.last_overwrite_data = None
                self.last_overwrite_time = None
            else:
                return {"out": (self.last_overwrite_data.data, self.last_overwrite_data.meta)}

        if self.params.constant.graph.value == "ring":
            matrix = ring_graph_adjacency_matrix(int(self.params.constant.shape.value))
            return {"out": (matrix, {"sfreq": self.params.common.max_frequency.value})}

        if self.params.constant.graph.value == "random":
            return {
                "out": (
                    np.random.rand(int(self.params.constant.shape.value), int(self.params.constant.shape.value)),
                    {"sfreq": self.params.common.max_frequency.value},
                )
            }

        parts = [p for p in self.params.constant.shape.value.split(",") if len(p) > 0]
        shape = list(map(int, parts))
        return {
            "out": (
                np.ones(shape) * self.params.constant.value.value,
                {"sfreq": self.params.common.max_frequency.value},
            )
        }


def ring_graph_adjacency_matrix(n):
    # Create an nxn zero matrix
    adjacency = np.zeros((n, n), dtype=int)

    # Set values for the ring connections
    for i in range(n):
        adjacency[i][(i + 1) % n] = 1  # Next vertex in the ring
        adjacency[i][(i - 1) % n] = 1  # Previous vertex in the ring

    return adjacency
