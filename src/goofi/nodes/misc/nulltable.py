from goofi.data import Data, DataType
from goofi.node import Node


class NullTable(Node):
    def config_input_slots():
        return {"table_in": DataType.TABLE}

    def config_output_slots():
        return {"table_out": DataType.TABLE}

    def process(self, table_in: Data):
        return {"table_out": (table_in.data, table_in.meta)}
