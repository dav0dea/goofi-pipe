from goofi.data import DataType, Data
from goofi.node import Node
from goofi.params import StringParam


class TableSelectString(Node):
    """
    This node extracts a specific string value from an input table based on a selected key. If the value associated with the key is not already a string, it is converted to a string before output.

    Inputs:
    - input_table: A table containing key-value pairs where each value is a data object.

    Outputs:
    - output_string: The string value retrieved from the table using the selected key.
    """

    def config_input_slots():
        return {"input_table": DataType.TABLE}

    def config_output_slots():
        return {"output_string": DataType.STRING}

    def config_params():
        return {
            "selection": {"key": StringParam("default_key")},
            "common": {"autotrigger": False},
        }

    def process(self, input_table: Data):
        if input_table is None:
            return None

        # Retrieve the selected key
        selected_key = self.params["selection"]["key"].value

        if selected_key not in input_table.data:
            return

        selected_value = input_table.data[selected_key]
        # print(selected_value)
        # print(type(selected_value))
        if selected_value.dtype != DataType.STRING:
            selected_value.data = str(selected_value.data)

        return {"output_string": (selected_value.data, input_table.meta)}
