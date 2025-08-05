from goofi.data import Data, DataType
from goofi.node import Node


class NullTable(Node):
    """
    This node receives a table and passes it through unchanged. It does not modify the table's data or metadata. This node can be used when a direct passthrough of table data is needed in a processing graph.

    Inputs:
    - table_in: The input table to be passed through.

    Outputs:
    - table_out: The same table as the input, unchanged.
    """

    def config_input_slots():
        return {"table_in": DataType.TABLE}

    def config_output_slots():
        return {"table_out": DataType.TABLE}

    def process(self, table_in: Data):
        return {"table_out": (table_in.data, table_in.meta)}
