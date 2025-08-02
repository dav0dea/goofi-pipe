from goofi.data import Data, DataType
from goofi.node import Node


class NullArray(Node):
    def config_input_slots():
        return {"array_in": DataType.ARRAY}

    def config_output_slots():
        return {"array_out": DataType.ARRAY}

    def process(self, array_in: Data):
        return {"array_out": (array_in.data, array_in.meta)}
