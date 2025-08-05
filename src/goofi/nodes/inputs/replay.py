import ast

import numpy as np

from goofi.data import DataType, to_data
from goofi.node import Node
from goofi.params import BoolParam, StringParam


def convert_to_numpy(val):
    try:
        parsed_val = ast.literal_eval(val)  # Convert string to Python list/dict
        if isinstance(parsed_val, list):
            return np.array(parsed_val, dtype=np.float32)  # Convert lists to NumPy arrays
        return parsed_val  # Keep as is if it's not a list
    except (ValueError, SyntaxError):
        return val  # Return as is if it can't be parsed


class Replay(Node):
    """
    This node replays data from a CSV file as a table, outputting one row at a time on each process step. It reads the specified CSV file, converts each row into a dictionary with appropriate data formats (including lists as NumPy arrays), and sequentially outputs the data row by row, looping back to the start after reaching the end of the file.

    Outputs:
    - table_output: A tuple containing a dictionary representation of the current CSV row and an empty dictionary. All columns in the CSV are included as fields in the output dictionary, with lists automatically converted to NumPy arrays where applicable.
    """

    @staticmethod
    def config_output_slots():
        return {"table_output": DataType.TABLE}

    @staticmethod
    def config_params():
        return {
            "Read": {
                "filename": StringParam("output.csv"),
                "play": BoolParam(False),
                "restart": BoolParam(False, trigger=True),
            }
        }

    def setup(self):
        self.df = None
        self.current_index = 0
        self.last_filename = None  # Track filename changes
        self.load_csv()

    def load_csv(self):
        import pandas as pd

        filename = self.params["Read"]["filename"].value
        if filename != self.last_filename:  # Only reload if filename changed
            self.df = pd.read_csv(
                filename,
                converters={col: convert_to_numpy for col in pd.read_csv(filename, nrows=1).columns},
            )
            self.current_index = 0
            self.last_filename = filename

    def process(self):
        # Reload CSV if filename has changed
        self.load_csv()

        if self.df is None or self.df.empty:
            return {"table_output": ({}, {})}  # Return empty table instead of None

        if not self.params["Read"]["play"].value:
            return

        # Extract the current row as a dictionary
        row_data = self.df.iloc[self.current_index].to_dict()

        # Convert each value to the appropriate Data format
        table_output = {key: to_data(value) for key, value in row_data.items()}

        # Increment index, loop back to the start when reaching the end
        self.current_index = (self.current_index + 1) % len(self.df)

        return {"table_output": (table_output, {})}

    def read_filename_changed(self):
        self.load_csv()

    def read_restart_changed(self):
        self.current_index = 0  # Reset index when restart is triggered
