from collections import defaultdict

import numpy as np

from goofi.data import DataType
from goofi.node import Node
from goofi.params import BoolParam


class ProbabilityMatrix(Node):
    """
    This node computes and updates a state transition probability matrix based on an incoming sequence of data values. Each data value is discretized (rounded to 2 decimal places), and the node tracks transitions between consecutive values, incrementally building a transition matrix representing the empirical probabilities of moving from one state to another within the observed sequence.

    Inputs:
    - input_data: Array of numerical data points, which are discretized and used to build the transition probability matrix by tracking transitions between consecutive values.

    Outputs:
    - data: The current transition probability matrix as a NumPy array, along with associated metadata such as sampling frequency.
    """

    def config_input_slots():
        return {"input_data": DataType.ARRAY}

    def config_params():
        return {
            "Probability": {
                "Reset": BoolParam(trigger=True),
            }
        }

    def config_output_slots():
        return {"data": DataType.ARRAY}

    def setup(self):
        self.state_to_idx = {}  # Mapping of state to index
        self.idx_to_state = []  # List of states, where index corresponds to state's index
        self.transition_count = defaultdict(lambda: defaultdict(int))
        self.transition_matrix = None
        self.last_state = None

    def update_transition_matrix(self):
        """Update the transition matrix based on transition counts."""
        num_states = len(self.idx_to_state)
        self.transition_matrix = np.zeros((num_states, num_states))
        for i, targets in self.transition_count.items():
            total_transitions = sum(targets.values())
            for j, count in targets.items():
                self.transition_matrix[i][j] = count / total_transitions

    def process(self, input_data):
        if input_data is None:
            return None

        # Discretize the input data by rounding to 2 decimal places
        data = [round(val, 2) for val in input_data.data]

        # Update state-to-index mapping and index-to-state list
        for state in data:
            if state not in self.state_to_idx:
                self.state_to_idx[state] = len(self.idx_to_state)
                self.idx_to_state.append(state)

        # Handle sequential data
        for current_state in data:
            if self.last_state is not None:
                i = self.state_to_idx[self.last_state]
                j = self.state_to_idx[current_state]
                self.transition_count[i][j] += 1
            self.last_state = current_state

        # Update the transition matrix
        self.update_transition_matrix()

        if self.params.Probability.Reset.value:
            self.setup()
        return {"data": (self.transition_matrix, {"sfreq": self.params.common.autotrigger.value})}

    def autotrigger_changed(self, value):
        self.setup()
