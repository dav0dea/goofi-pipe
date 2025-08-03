import datetime
import os
import numpy as np
from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import StringParam, BoolParam, FloatParam
import json
import time


class WriteCsv(Node):
    """
    This node writes incoming table data to a CSV file, supporting both generic tables and EEG-specific formats. The node can append new rows to an existing CSV, generate unique filenames based on the current time, and optionally include timestamps. Two writing modes are supported: a default mode for general tabular data and an EEG mode that handles multidimensional arrays and sampling frequency metadata. Data is automatically flattened and serialized as needed to preserve structure in the CSV output.

    Inputs:
    - table_input: Table data to be written into the CSV file. The table can contain nested tables, arrays, or strings.
    - start: Array signal triggering the start of writing to the CSV file.
    - stop: Array signal triggering the stop of writing to the CSV file.
    - fname: String specifying the filename to use for the CSV output.

    Outputs:
    - None. This node writes data to disk but does not produce downstream data outputs.
    """

    @staticmethod
    def config_input_slots():
        # This node will accept a table as its input.
        return {"table_input": DataType.TABLE, "start": DataType.ARRAY, "stop": DataType.ARRAY, "fname": DataType.STRING}

    @staticmethod
    def config_params():
        # Parameters can include the CSV filename, write control, and timestamp option.
        return {
            "Write": {
                "filename": StringParam("output.csv"),
                "start": BoolParam(False, trigger=True),
                "stop": BoolParam(False, trigger=True),
                "duration": FloatParam(0.0, 0.0, 100.0),
                "timestamps": BoolParam(False),  # New timestamp parameter
                "writing_mode": StringParam("default", options=["default", "eeg"]),
            },
        }

    def setup(self):
        import pandas as pd

        self.pd = pd
        self.last_filename = None
        self.base_filename = None  # Track the base filename without timestamp
        self.written_files = set()  # Track files to ensure headers are written
        self.last_values = {}  # Store the last known value for each column
        self.is_writing = False

    def process(self, table_input: Data, start: Data, stop: Data, fname: Data):
        if start is not None and (start.data > 0).any() or self.params["Write"]["start"].value:
            self.is_writing = True
            self.start_time = time.time()
            if fname is not None:
                filename = fname.data
                self.params.Write.filename.value = filename
            else:
                # Use the filename from parameters if no custom filename is provided
                filename = self.params["Write"]["filename"].value
            # Generate a new filename each time writing starts
            basename, ext = os.path.splitext(filename)
            datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.last_filename = f"{basename}_{datetime_str}{ext}"
        if stop is not None or self.params["Write"]["stop"].value:
            self.is_writing = False

        self.input_slots["start"].clear()
        self.input_slots["stop"].clear()
        duration = self.params["Write"]["duration"].value
        if self.is_writing:
            if duration > 0:
                if time.time() - self.start_time > duration:
                    self.is_writing = False
                    return

            mode = self.params["Write"]["writing_mode"].value
            if mode == "default":
                self.default_mode(table_input, start, stop)
            elif mode == "eeg":
                self.eeg_mode(table_input, start, stop)

    def default_mode(self, table_input, start, stop):
        table_data = table_input.data

        # Extract actual data content, handling multiple columns
        actual_data = {key: (value.data if isinstance(value, Data) else value) for key, value in table_data.items()}

        def flatten(data):
            """Ensure lists and NumPy arrays are stored as JSON strings to keep their structure."""
            if isinstance(data, np.ndarray):
                return json.dumps(data.tolist())  # Convert ndarray to list before serializing
            elif isinstance(data, (list, tuple)):
                return json.dumps(data)  # Serialize list as a JSON string
            return data  # Return scalars as-is

        flattened_data = {
            col: [flatten(values)] if not isinstance(values, list) else [flatten(v) for v in values]
            for col, values in actual_data.items()
        }

        # Ensure all columns have the same length by padding with None
        max_length = max(map(len, flattened_data.values()), default=0)
        for col in flattened_data:
            flattened_data[col] += [None] * (max_length - len(flattened_data[col]))

        # Replace None with the last known value
        for col in flattened_data:
            if col not in self.last_values:
                self.last_values[col] = None  # Initialize with None if not present
            for i in range(len(flattened_data[col])):
                if flattened_data[col][i] is None:
                    flattened_data[col][i] = self.last_values[col]
                else:
                    self.last_values[col] = flattened_data[col][i]  # Update the last known value

        # Add timestamp column if enabled
        if self.params["Write"]["timestamps"].value:
            timestamps = [datetime.datetime.utcnow().isoformat()] * max_length
            flattened_data["timestamp"] = timestamps

        # Convert to DataFrame
        df = self.pd.DataFrame(flattened_data)

        # Check if filename has changed, then update with timestamp
        fn = self.last_filename

        # Determine if headers should be written
        write_header = fn not in self.written_files

        # Append new data to CSV
        df.to_csv(fn, mode="a", header=write_header, index=False)

        # Mark file as written to prevent duplicate headers
        if write_header:
            self.written_files.add(fn)

    def eeg_mode(self, table_input, start, stop):
        table_data = table_input.data

        column_data = get_column_data(table_data)

        max_samples = None
        for key, val in column_data.items():
            if val.dtype == DataType.ARRAY and max_samples is None:
                max_samples = len(val.data)
            elif val.dtype == DataType.ARRAY:
                assert max_samples == len(val.data), f"Column {key} has inconsistent length."
        if max_samples is None:
            max_samples = 1

        sfreq = None
        for key, val in column_data.items():
            if "sfreq" in val.meta and sfreq is None:
                sfreq = val.meta["sfreq"]
            elif "sfreq" in val.meta:
                assert sfreq == val.meta["sfreq"], f"Column {key} has inconsistent sampling frequency."
        if sfreq is None and max_samples > 1:
            raise ValueError("No sampling frequency found in the data.")

        # Add timestamp column if enabled
        if self.params["Write"]["timestamps"].value:
            timestamp = datetime.datetime.utcnow().isoformat()
            # interpolate timestamps based on sampling frequency
            timestamps = [timestamp] * max_samples
            for i in range(max_samples):
                timestamps[i] = (datetime.datetime.fromisoformat(timestamp) + datetime.timedelta(seconds=i / sfreq)).isoformat()
            column_data["timestamp"] = timestamps

        # Convert all data to lists
        for key, val in column_data.items():
            if isinstance(val, list):
                continue
            if val.dtype == DataType.ARRAY:
                column_data[key] = val.data.tolist()
            elif val.dtype == DataType.STRING:
                column_data[key] = [val.data] * max_samples
            else:
                raise ValueError(f"Unsupported data type: {val.dtype}")

        # Convert to DataFrame
        df = self.pd.DataFrame(column_data)
        # Check if filename has changed, then update with timestamp
        fn = self.last_filename
        # Determine if headers should be written
        write_header = fn not in self.written_files
        # Append new data to CSV
        df.to_csv(fn, mode="a", header=write_header, index=False)
        if write_header:
            self.written_files.add(fn)


def get_column_data(table_data, column_data=None, prefix=""):
    if column_data is None:
        column_data = {}
    for key, data in table_data.items():
        if data.dtype == DataType.TABLE:
            column_data = get_column_data(data.data, column_data, prefix + key + "-")
        elif data.dtype == DataType.ARRAY:
            arr = data.data
            assert arr.ndim <= 2, f"Array must be 1D or 2D, got {arr.ndim}."

            channel_names = [str(i) for i in range(arr.shape[0])]
            if "channels" in data.meta and "dim0" in data.meta["channels"]:
                channel_names = data.meta["channels"]["dim0"]

            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            for i in range(arr.shape[0]):
                meta = data.meta.copy()
                del meta["channels"]
                column_data[prefix + key + "-" + channel_names[i]] = Data(data.dtype, arr[i], meta)
        elif data.dtype == DataType.STRING:
            column_data[prefix + key] = data
        else:
            raise ValueError(f"Unsupported data type: {data.dtype}")
    return column_data
