from goofi.data import Data, DataType
from goofi.node import Node
from goofi.nodes.inputs._overwrite import _OverwriteHold
from goofi.params import FloatParam, StringParam


class ConstantString(_OverwriteHold, Node):

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
        self.setup_overwrite()

    def process(self, overwrite: Data):
        held = self.held_override(overwrite)
        if held is not None:
            return {"out": (held.data, {})}
        return {"out": (self.params.constant.value.value, {})}
