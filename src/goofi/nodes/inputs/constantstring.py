import time

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, StringParam


class ConstantString(Node):

    def config_params():
        return {
            "constant": {
                "value": StringParam("default_value"),
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
        return {"overwrite": DataType.STRING}

    def config_output_slots():
        return {"out": DataType.STRING}

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
                return {"out": (self.last_overwrite_data.data, {})}

        return {"out": (self.params.constant.value.value, {})}
