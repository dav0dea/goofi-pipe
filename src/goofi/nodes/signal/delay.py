import time
from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam

class Delay(Node):
    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {"output": DataType.ARRAY}

    def config_params():
        return {
            "delay": {
                "time": FloatParam(0.1, 0.0, 100.0)  # Delay time in seconds (default 0.1s, max 10s)
            }
        }

    def process(self, data: Data):
        if data is None or data.data is None:
            return None
        
        # Get delay time from parameter
        delay_time = self.params.delay.time.value
        
        # Apply delay
        time.sleep(delay_time)
        
        return {"output": (data.data, data.meta)}
