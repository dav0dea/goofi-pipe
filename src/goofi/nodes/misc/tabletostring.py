import json
import yaml
import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, StringParam


class TableToString(Node):
    """
    Converts a table data structure into a text representation in either JSON or YAML format. The node serializes the input table and outputs it as a string, enabling easy inspection or further text-based processing.

    Inputs:
    - table: The input table to be converted to text.

    Outputs:
    - text: The string representation of the input table, serialized in the chosen format.
    """

    def config_input_slots():
        return {"table": DataType.TABLE}

    def config_output_slots():
        return {"text": DataType.STRING}

    def config_params():
        return {
            "table_to_string": {
                "format": StringParam("json", options=["json", "yaml"], doc="Output text format"),
                "add_backslashes": BoolParam(True, doc="Add backslashes before quotes"),
            }
        }

    def process(self, table):
        if table.data is None:
            return None

        meta = table.meta
        table = table.data

        if self.params.table_to_string.format.value == "json":
            try:
                text = json.dumps(table, default=convert_data_to_serializable)
            except TypeError as e:
                raise ValueError(f"Error encoding JSON: {e}")
        elif self.params.table_to_string.format.value == "yaml":
            try:
                text = yaml.safe_dump(table, default_flow_style=False)
            except yaml.YAMLError as e:
                raise ValueError(f"Error encoding YAML: {e}")
        else:
            raise ValueError(f"Unsupported format: {self.params['table_to_string']['format'].value}")

        if self.params.table_to_string.add_backslashes.value:
            text = text.replace('"', '\\"')

        return {"text": (text, meta)}


def convert_data_to_serializable(data):
    if isinstance(data, Data):
        if data.dtype == DataType.TABLE:
            return {key: convert_data_to_serializable(value) for key, value in data.data.items()}
        elif data.dtype == DataType.ARRAY:
            return data.data.tolist()
        elif data.dtype == DataType.STRING:
            return data.data
    elif isinstance(data, np.ndarray):
        return data.tolist()
    return data
