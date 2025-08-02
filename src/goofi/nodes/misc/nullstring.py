from goofi.data import Data, DataType
from goofi.node import Node


class NullString(Node):
    def config_input_slots():
        return {"string_in": DataType.STRING}

    def config_output_slots():
        return {"string_out": DataType.STRING}

    def process(self, string_in: Data):
        return {"string_out": (string_in.data, string_in.meta)}
