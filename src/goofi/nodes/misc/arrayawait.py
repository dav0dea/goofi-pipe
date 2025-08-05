from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, StringParam


class ArrayAwait(Node):
    """
    Waits for an incoming trigger to pass through the most recent array data. The node outputs the array only when both new data is present and a trigger event occurs. If the trigger input is not received, the array data will not be forwarded to the output.

    Inputs:
    - data: Array to be held until a trigger is received.
    - trigger: Array input used solely as a trigger event to release the latest data.

    Outputs:
    - out: The latest array data passed through when a trigger is received.
    """

    def config_input_slots():
        return {"data": DataType.ARRAY, "trigger": DataType.ARRAY}

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def config_params():
        return {
            "array_await": {
                "require_change": BoolParam(True, doc="Only output when the data changes, and we have an unconsumed trigger")
            }
        }

    def setup(self):
        self.last_data = None

    def process(self, data: Data, trigger: Data):
        if trigger is None or data is None:
            return

        if self.params.array_await.require_change.value and self.last_data is not None and (self.last_data == data.data).all():
            return

        self.input_slots["trigger"].clear()

        self.last_data = data.data
        return {"out": (data.data, data.meta)}
