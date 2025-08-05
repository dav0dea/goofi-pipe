import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, IntParam


class VAMP(Node):
    """
    This node computes a variational approach for Markov processes (VAMP) dimensionality reduction on incoming 2D array data. It accumulates epochs of data, fits a VAMP model using a sliding time-lagged window, and projects input data onto learned VAMP components in real-time. The node is typically used to extract meaningful low-dimensional temporal features from multivariate time series.

    Inputs:
    - data: 2D array input data representing multichannel time series.

    Outputs:
    - comps: 2D array of VAMP components for the input data, with updated metadata.
    """

    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {
            "comps": DataType.ARRAY,
        }

    def config_params():
        return {
            "vamp": {
                "time_lag": IntParam(10, 1, 300, doc="Time lag tau (in samples)"),
                "n_dims": IntParam(2, 1, 8, doc="Number of dimensions to keep"),
                "collect": BoolParam(True, doc="Collect data and fit the model"),
                "epoch_size": IntParam(256, 2, 1000, doc="Size of one epoch (in samples)"),
                "reset": BoolParam(False, trigger=True, doc="Reset the model"),
            }
        }

    def setup(self):
        self.current_epoch = None
        self.buffer = []
        self.internal_model = self.make_model()
        self.model = None

    def process(self, data: Data):
        if data is None:
            return None

        assert data.data.ndim == 2, "Data must be 2D"

        if self.params.vamp.reset.value:
            self.buffer = []
            self.current_epoch = None
            self.internal_model = self.make_model()
            self.model = None

        if self.params.vamp.collect.value:
            # extend current epoch
            if self.current_epoch is None:
                self.current_epoch = data.data
            else:
                self.current_epoch = np.concatenate([self.current_epoch, data.data], axis=1)

            if self.current_epoch.shape[1] >= (self.params.vamp.epoch_size.value + self.params.vamp.time_lag.value):
                # epoch is full, update model and start new epoch
                self.vamp_collect_changed(True)

        if self.model is None:
            return None

        # remove channel names, which are now components
        if "dim0" in data.meta["channels"]:
            del data.meta["channels"]["dim0"]

        # transform current data
        comps = self.model.transform(data.data.T).T
        return {"comps": (comps, data.meta)}

    def vamp_collect_changed(self, value):
        if self.current_epoch is not None:
            self.buffer.append(self.current_epoch.T)

            # update model
            tau = self.params.vamp.time_lag.value
            self.internal_model.partial_fit((self.current_epoch.T[:-tau], self.current_epoch.T[tau:]))
            self.model = self.internal_model.fetch_model()

            self.current_epoch = None

    def vamp_time_lag_changed(self, value):
        # refit a new model with updated time lag
        self.internal_model = self.make_model()
        self.internal_model.fit_from_timeseries(self.buffer)
        self.model = self.internal_model.fetch_model()

    def vamp_n_dims_changed(self, value):
        # refit a new model with updated number of dimensions
        self.internal_model = self.make_model()
        self.internal_model.fit_from_timeseries(self.buffer)
        self.model = self.internal_model.fetch_model()

    def make_model(self):
        from deeptime.decomposition import VAMP as VAMP_Model

        return VAMP_Model(lagtime=self.params.vamp.time_lag.value, dim=self.params.vamp.n_dims.value)
