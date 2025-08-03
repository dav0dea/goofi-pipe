from goofi.data import Data, DataType
from goofi.node import Node


class ExtendedTable(Node):
    """
    This node extends an input table by adding up to ten new entries, five from array inputs and five from string inputs. For each input, if a value is provided, it is added to the resulting table with a corresponding key. If no new entries are provided, the original base table is returned unchanged.

    Inputs:
    - base: The original table to extend.
    - array_input1: First array to add to the table.
    - array_input2: Second array to add to the table.
    - array_input3: Third array to add to the table.
    - array_input4: Fourth array to add to the table.
    - array_input5: Fifth array to add to the table.
    - string_input1: First string to add to the table.
    - string_input2: Second string to add to the table.
    - string_input3: Third string to add to the table.
    - string_input4: Fourth string to add to the table.
    - string_input5: Fifth string to add to the table.

    Outputs:
    - table: The extended table containing the original entries plus any new arrays and strings added.
    """

    def config_input_slots():
        # Define 10 input slots, 5 are Arrays and 5 are Strings
        return {
            "base": DataType.TABLE,
            "array_input1": DataType.ARRAY,
            "array_input2": DataType.ARRAY,
            "array_input3": DataType.ARRAY,
            "array_input4": DataType.ARRAY,
            "array_input5": DataType.ARRAY,
            "string_input1": DataType.STRING,
            "string_input2": DataType.STRING,
            "string_input3": DataType.STRING,
            "string_input4": DataType.STRING,
            "string_input5": DataType.STRING,
        }

    def config_output_slots():
        # Keep the output slot same as the Table Node if needed
        return {"table": DataType.TABLE}

    def config_params():
        return {
            "table": {
                "param1": "value1",
                "param2": "value2",
                "param3": "value3",
                "param4": "value4",
                "param5": "value5",
                "param6": "value6",
                "param7": "value7",
                "param8": "value8",
                "param9": "value9",
                "param10": "value10",
            }
        }

    def process(self, **inputs):
        base = inputs.get("base", None)

        if base is None:
            # if no base is given, use an empty table
            base = Data(DataType.TABLE, DataType.TABLE.empty(), {})

        # Initialize an empty list to store new_entries
        new_entries = []

        # Loops to get the new entries from the inputs and store them in new_entries list
        for i in range(1, 6):
            array_key = f"array_input{i}"
            array_entry = inputs.get(array_key, None)
            new_entries.append(array_entry)

        for j in range(1, 6):
            string_key = f"string_input{j}"

            string_entry = inputs.get(string_key, None)

            new_entries.append(string_entry)

        # if no new entries are given, return the base table
        if all(entry is None for entry in new_entries):
            return {"table": (base.data, base.meta)}

        # add the new entries to the base table
        for i, entry in enumerate(new_entries):
            if entry is not None:
                key = self.params["table"][f"param{i+1}"].value  # Note the change here to access the parameter by name
                assert len(key) > 0, "New Entry Key cannot be empty."
                base.data[key] = entry  # Updated to entry.data to store the actual data

        return {"table": (base.data, base.meta)}
