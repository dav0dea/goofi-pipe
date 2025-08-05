import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, StringParam


class StringToTable(Node):
    """
    Converts a text containing structured data (in JSON or YAML format) into a Goofi table. The node parses the input string, automatically handling nested objects and converting them into nested table structures. Strings become string fields, arrays become array fields, and objects become subtables.

    Inputs:
    - text: The input string containing data in a structured format (JSON or YAML).

    Outputs:
    - table: The resulting table parsed from the input text, with fields mapped to Goofi TABLE, STRING, or ARRAY types as appropriate.
    """

    def config_input_slots():
        return {"text": DataType.STRING}

    def config_output_slots():
        return {"table": DataType.TABLE}

    def config_params():
        return {
            "string_to_table": {
                "format": StringParam("json", options=["json", "yaml"], doc="Input text format"),
                "clean_backslashes": BoolParam(True, doc="Remove backslashes before quotes"),
            }
        }

    def process(self, text):
        if text.data is None:
            return None

        meta = text.meta
        text = text.data
        if self.params.string_to_table.clean_backslashes.value:
            text = text.replace('\\"', '"')

        if self.params.string_to_table.format.value == "json":
            try:
                import json

                table = json.loads(text)
            except json.JSONDecodeError as e:
                raise ValueError(f"Error decoding JSON: {e}")
        elif self.params.string_to_table.format.value == "yaml":
            try:
                import yaml

                table = yaml.safe_load(text)
            except yaml.YAMLError as e:
                raise ValueError(f"Error decoding YAML: {e}")
        else:
            raise ValueError(f"Unsupported format: {self.params['string_to_table']['format'].value}")

        # parse the table and return it
        return {"table": (parse_table(table, meta), meta)}


def parse_table(table, meta):
    for key, value in table.items():
        if isinstance(value, dict):
            table[key] = Data(DataType.TABLE, parse_table(value, meta), meta)
            continue

        if isinstance(value, str):
            table[key] = Data(DataType.STRING, value, meta)
            continue

        table[key] = Data(DataType.ARRAY, np.array(value), meta)

    return table
