from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import StringParam


class TableSelectString(Node):
    def config_input_slots():
        return {"input_table": DataType.TABLE, "key": DataType.STRING}

    def config_output_slots():
        return {"output_string": DataType.STRING}

    def config_params():
        return {
            "selection": {"key": StringParam("default_key")},
            "common": {"autotrigger": False},
        }

    def process(self, input_table: Data, key: Data):
        if input_table is None:
            return None

        if key is not None:
            self.params.selection.key.value = key.data
            self.input_slots["key"].clear()

        selected_key = self.params.selection.key.value

        if selected_key not in input_table.data:
            return

        selected_value = input_table.data[selected_key]
        if selected_value.dtype != DataType.STRING:
            selected_value.data = str(selected_value.data)

        return {"output_string": (selected_value.data, input_table.meta)}
