import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import StringParam


class TuningMatrix(Node):
    """
    This node computes a tuning matrix and related metrics from an input tuning array using a selectable metric function. It processes the tuning data to generate a matrix that characterizes relationships or similarities between tunings, as well as metrics measuring aspects of the tuning structure.

    Inputs:
    - tuning: An array containing tuning data to be analyzed.

    Outputs:
    - matrix: The computed tuning matrix, expressing inter-relations between elements in the input tuning array. Depending on configuration, this may be a normalized matrix or may be set to None for certain ratio types.
    - metric_per_step: An array containing the metric score calculated for each step or pair in the tuning.
    - metric: An array containing overall metric values summarizing properties of the tuning.
    """

    def config_input_slots():
        return {
            "tuning": DataType.ARRAY,
        }

    def config_output_slots():
        return {
            "matrix": DataType.ARRAY,
            "metric_per_step": DataType.ARRAY,
            "metric": DataType.ARRAY,
        }

    def config_params():
        return {
            "Tuning_Matrix": {
                "function": StringParam("dyad_similarity", options=["dyad_similarity", "consonance", "metric_denom"]),
                "ratio_type": StringParam("all", options=["all", "pos_harm", "sub_harm"]),
            }
        }

    def setup(self):
        from biotuner.metrics import (
            compute_consonance,
            dyad_similarity,
            metric_denom,
            tuning_cons_matrix,
        )

        self.tuning_cons_matrix = tuning_cons_matrix
        self.dyad_similarity = dyad_similarity
        self.compute_consonance = compute_consonance
        self.metric_denom = metric_denom

    def process(self, tuning: Data):
        if tuning is None or tuning.data is None:
            return None
        tuning = tuning.data
        function = self.params.Tuning_Matrix.function.value
        ratio_type = self.params.Tuning_Matrix.ratio_type.value
        if function == "dyad_similarity":
            metric_per_step, metric, matrix = self.tuning_cons_matrix(tuning, self.dyad_similarity, ratio_type=ratio_type)
        if function == "consonance":
            metric_per_step, metric, matrix = self.tuning_cons_matrix(tuning, self.compute_consonance, ratio_type=ratio_type)
        if function == "metric_denom":
            metric_per_step, metric, matrix = self.tuning_cons_matrix(
                tuning, self.metric_denom, ratio_type=ratio_type, metric_denom=True
            )
        metric_per_step = np.array(metric_per_step)
        metric = np.array(metric)
        matrix = np.array(matrix)
        if ratio_type == "pos_harm" or ratio_type == "sub_harm":
            scaled_matrix = None
        else:
            scaled_matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
        return {
            "matrix": (scaled_matrix, {}),
            "metric_per_step": (metric_per_step, {}),
            "metric": (metric, {}),
        }
