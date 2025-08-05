from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam


class StringAwait(Node):
    """
    Waits for a trigger signal before outputting the provided string message. The node only outputs the message in response to a trigger, and can be set to only emit when the message content has changed. Once triggered, the output is generated and the trigger is consumed.

    Inputs:
    - message: The string to be output when triggered.
    - trigger: An array acting as the trigger signal. The presence of a value triggers the output.

    Outputs:
    - out: The provided string message, passed through when the trigger is received.
    """

    def config_input_slots():
        return {"message": DataType.STRING, "trigger": DataType.ARRAY}

    def config_output_slots():
        return {"out": DataType.STRING}

    def config_params():
        return {
            "string_await": {
                "require_change": BoolParam(True, doc="Only output when the message changes, and we have an unconsumed trigger")
            }
        }

    def setup(self):
        self.last_message = None

    def process(self, message: Data, trigger: Data):
        if trigger is None or message is None:
            return

        if self.params.string_await.require_change.value and self.last_message == message.data:
            return

        self.input_slots["trigger"].clear()

        self.last_message = message.data
        return {"out": (message.data, message.meta)}
