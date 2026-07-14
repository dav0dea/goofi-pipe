import time
from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam


class Delay(Node):
    """
    Introduces a configurable time delay into the data stream, pausing the forwarding of incoming array data for a specified duration before outputting it unchanged.

    Inputs:
    - data: An array of data to be delayed.

    Outputs:
    - output: The same array data as received on input, passed after the specified delay.
    """

    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {"output": DataType.ARRAY}

    def config_params():
        return {"delay": {"time": FloatParam(0.1, 0.0, 100.0)}}  # Delay time in seconds (default 0.1s, max 10s)

    def process(self, data: Data):
        if data is None:
            return None

        # Get delay time from parameter
        delay_time = self.params.delay.time.value

        # Apply the delay in small chunks so terminate() (which clears self.alive) can
        # interrupt a long delay instead of blocking the processing thread for the whole
        # window — otherwise the node is unresponsive to shutdown for up to delay_time.
        remaining = delay_time
        while remaining > 0 and self.alive:
            step = min(0.05, remaining)
            time.sleep(step)
            remaining -= step

        return {"output": (data.data, data.meta)}
