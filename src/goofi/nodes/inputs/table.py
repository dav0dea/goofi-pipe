from goofi.data import Data, DataType
from goofi.node import Node


class Table(Node):
    """
    Creates or updates a table by adding a new entry under a specified key. If no base table is provided, an empty table is used. If no new entry is given, the current table is returned unchanged.

    Inputs:
    - base: The existing table to which a new entry can be added.
    - new_entry: The data (as an array) to insert into the table.

    Outputs:
    - table: The updated table after adding the new entry, or the original table if no new entry was given.
    """

    def config_input_slots():
        return {"base": DataType.TABLE, "new_entry": DataType.ARRAY}

    def config_output_slots():
        return {"table": DataType.TABLE}

    def config_params():
        return {"table": {"new_entry_key": "key"}}

    def process(self, base: Data, new_entry: Data):
        if base is None:
            # if no base is given, use an empty table
            base = Data(DataType.TABLE, DataType.TABLE.empty(), {})

        if new_entry is None:
            # if no new entry is given, return the base table
            return {"table": (base.data, base.meta)}

        # add the new entry to the base table
        assert len(self.params.table.new_entry_key.value) > 0, "New Entry Key cannot be empty."
        base.data[self.params.table.new_entry_key.value] = new_entry
        return {"table": (base.data, base.meta)}
