from goofi.data import Data, DataType
from goofi.node import Node


class AppendTables(Node):
    """
    This node combines two tables by merging their data and metadata dictionaries into a single output table. If only one input table is provided, it passes that table to the output unchanged. If both inputs are absent, the output is None.

    Inputs:
    - table1: The first input table to be combined.
    - table2: The second input table to be combined.

    Outputs:
    - output_table: The resulting table containing merged data and metadata from both input tables. If only one table is provided, it outputs that table as is.
    """

    def config_input_slots():
        return {"table1": DataType.TABLE, "table2": DataType.TABLE}

    def config_output_slots():
        return {"output_table": DataType.TABLE}

    def config_params():
        return {}

    def process(self, table1: Data, table2: Data):
        if table1 is None and table2 is None:
            return None
        if table2 is None:
            return {"output_table": (table1.data, table1.meta)}
        if table1 is None:
            return {"output_table": (table2.data, table2.meta)}

        # Combine the two tables' data dictionaries
        combined_data = {**table1.data, **table2.data}

        # For simplicity, I'm just combining the meta data dictionaries here,
        # but you might want to handle meta data merging more carefully.
        combined_meta = {**table1.meta, **table2.meta}

        return {"output_table": (combined_data, combined_meta)}
