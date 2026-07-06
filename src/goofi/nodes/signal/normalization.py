import numpy as np
from sklearn import preprocessing

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, FloatParam, IntParam, StringParam


def _normalization_params():
    """Shared param declaration for Normalization and TableNormalization —
    a function (not a constant) so every configure() gets fresh Param objects."""
    return {
        "normalization": {
            "method": StringParam("z-score", options=["z-score", "quantile", "robust", "minmax"]),
            "buffer_size": IntParam(1024, 2, 10000),
            "reset": BoolParam(trigger=True),
            "axis": -1,
        },
        "quantile": {
            "n_quantiles": IntParam(1000, 100, 10000),
            "output_distribution": StringParam("uniform", options=["uniform", "normal"]),
        },
        "robust": {
            "quantile_min": IntParam(25, 0, 100),
            "quantile_max": IntParam(75, 0, 100),
            "unit_variance": BoolParam(False),
        },
        "minmax": {
            "feature_min": FloatParam(0.0, -1.0, 1.0),
            "feature_max": FloatParam(1.0, -1.0, 1.0),
        },
    }


class Normalization(Node):
    """
    This node performs normalization on array data using various normalization methods such as z-score, quantile, robust, or min-max scaling. It manages a buffer of incoming data along a specified axis to compute statistics and applies the selected normalization technique in real-time. The node is designed for use with multi-dimensional array data and is suitable for pre-processing input signals to have consistent statistical properties.

    Inputs:
    - data: Array input data to be normalized.

    Outputs:
    - normalized: The normalized array data, transformed according to the selected normalization method and matching the structure and size of the input.
    """

    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {"normalized": DataType.ARRAY}

    def config_params():
        return _normalization_params()

    def setup(self):
        self.buffer = None

    def process(self, data: Data):
        if data is None or data.data is None:
            return None

        array = data.data
        if self.params.normalization.axis.value != -1:
            array = np.moveaxis(array, self.params.normalization.axis.value, -1)

        # handle reset trigger
        if self.params.normalization.reset.value:
            self.buffer = None

        if self.buffer is None:
            self.buffer = array
        else:
            try:
                self.buffer = np.concatenate((self.buffer, array), axis=-1)
            except Exception as e:
                print(f"Failed to extend buffer, resetting ({e})")
                self.buffer = None
                return

        # limit buffer size
        if self.buffer.shape[-1] > self.params.normalization.buffer_size.value:
            self.buffer = self.buffer[..., -self.params.normalization.buffer_size.value :]

        # normalize data
        normalized = np.zeros_like(self.buffer)
        for idxs in np.ndindex(self.buffer.shape[:-1]):
            current_slice = self.buffer[idxs]

            # replace NaNs and Infs with zeros
            current_slice = np.nan_to_num(current_slice, posinf=0, neginf=0)

            if self.params.normalization.method.value == "z-score":
                normalized[idxs] = preprocessing.scale(current_slice)
            elif self.params.normalization.method.value == "quantile":
                normalized[idxs] = preprocessing.quantile_transform(
                    current_slice.reshape(-1, 1),
                    n_quantiles=min(self.params.quantile.n_quantiles.value, current_slice.size),
                    output_distribution=self.params.quantile.output_distribution.value,
                ).squeeze(1)
            elif self.params.normalization.method.value == "robust":
                normalized[idxs] = preprocessing.robust_scale(
                    current_slice,
                    quantile_range=(self.params.robust.quantile_min.value, self.params.robust.quantile_max.value),
                    unit_variance=self.params.robust.unit_variance.value,
                )
            elif self.params.normalization.method.value == "minmax":
                normalized[idxs] = preprocessing.minmax_scale(
                    current_slice,
                    feature_range=(self.params.minmax.feature_min.value, self.params.minmax.feature_max.value),
                )

        # return only as much data as was passed in
        normalized = normalized[..., -array.shape[-1] :]

        # move axis back
        if self.params.normalization.axis.value != -1:
            normalized = np.moveaxis(normalized, -1, self.params.normalization.axis.value)

        return {"normalized": (normalized, data.meta)}


class TableNormalization(Node):
    """
    This node performs normalization on each column of a table independently. For each key (column) in the input table, it applies a normalization process and produces a table with normalized data columns. Each column is normalized using its own normalization state, ensuring independent treatment per column.

    Inputs:
    - table: A table containing multiple columns (keys), where each column will be normalized separately.

    Outputs:
    - normalized: A table with the same structure as the input, where each column has been normalized.
    """

    def config_input_slots():
        return {"table": DataType.TABLE}

    def config_output_slots():
        return {"normalized": DataType.TABLE}

    def config_params():
        return _normalization_params()

    def setup(self):
        self.normalizers = {}

    def process(self, table: Data):
        params = self.params.serialize()

        result = {}
        for key, data in table.data.items():
            if key not in self.normalizers:
                # initialize a new normalizer for this key
                self.normalizers[key] = Normalization.create_standalone()
                self.normalizers[key].setup()

            # update params of the normalizer
            self.normalizers[key].params.update(params)

            # process the data
            normalized = self.normalizers[key].process(data)["normalized"]
            result[key] = Data(data.dtype, *normalized)

        # make sure to "use" the reset trigger so it doesn't get stuck
        self.params.normalization.reset.value

        # delete normalizers for keys that are not in the table anymore
        self.normalizers = {key: norm for key, norm in self.normalizers.items() if key in table.data}

        return {"normalized": (result, table.meta)}
