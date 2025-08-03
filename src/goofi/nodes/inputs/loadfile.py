from copy import deepcopy

import numpy as np

from goofi.data import DataType
from goofi.node import Node
from goofi.params import FloatParam, StringParam


class LoadFile(Node):
    """
    This node loads data from a file and outputs the loaded data in array or string format, depending on the file type and content. It supports various file types such as spectrum data, time series, generic numpy arrays, embedding CSV files, and audio files. The node processes the input file according to the specified type and outputs the corresponding data structure with optional metadata.

    Inputs:
    - file: The filename (as a string) of the file to load.

    Outputs:
    - data_output: The primary data loaded from the file, such as an array with optional metadata depending on the file type.
    - string_output: The string representation of the data if applicable (e.g., if data is non-numeric and cannot be converted to an array).
    """

    def config_input_slots():
        return {"file": DataType.STRING}

    def config_output_slots():
        return {"data_output": DataType.ARRAY, "string_output": DataType.STRING}

    def config_params():
        return {
            "file": {
                "filename": StringParam("", doc="The name of the file to load with extension"),
                "type": StringParam(
                    "spectrum",
                    options=["spectrum", "time_series", "ndarray", "embedding_csv", "audio"],
                    doc="Type of file to load",
                ),
                "select": StringParam("", doc="NumPy selection string"),
            },
            "spectrum": {"freq_multiplier": FloatParam(1.0, doc="Multiplier to adjust the frequency values")},
            "embedding_csv": {"header": 0, "name_column": False, "index_column": True},
            "common": {"autotrigger": True, "max_frequency": 1.0},
        }

    def setup(self):
        import librosa
        import pandas as pd

        self.pd = pd
        self.librosa = librosa

        self.data_output = None
        self.string_output = None
        self.last_params = None

    def process(self, file):
        if file is not None:
            self.params.file.filename.value = file.data
            self.file_filename_changed(file.data)
            self.input_slots["file"].clear()

        if self.last_params == self.params and not (self.data_output is None and self.string_output is None):
            # if the parameters are the same, return the previous output
            return {"data_output": self.data_output, "string_output": self.string_output}

        # if the parameters are different, load the file
        self.last_params = deepcopy(self.params)
        self.load_file()

        return {"data_output": self.data_output, "string_output": self.string_output}

    def load_file(self):
        if not self.params.file.filename.value:
            self.data_output = None
            self.string_output = None
            return

        file_type = self.params.file.type.value
        filename = self.params.file.filename.value

        if file_type == "audio":
            try:
                audio, sr = self.librosa.load(f"{filename}", sr=None)
                self.data_output = (audio.astype(np.float32), {"sfreq": sr})
                self.string_output = None
            except Exception as e:
                print(f"Error loading audio file: {e}")
                self.data_output = None
                self.string_output = None
            return

        extension = filename.split(".")[-1]

        df = None
        if extension == "npy":
            data = np.load(f"{filename}", allow_pickle=True)
        elif extension == "txt":
            data = np.loadtxt(f"{filename}")
        elif extension == "csv":
            df = self.pd.read_csv(
                f"{filename}",
                header=self.params.embedding_csv.header.value,
                index_col=0 if self.params.embedding_csv.index_column.value else None,
            )
            data = df.values

            if self.params.embedding_csv.name_column.value:
                meta = {"channels": {"dim0": list(df.iloc[:, 0].values)}}
                data = data[:, 1:]

        # apply selection
        selection = self.params.file.select.value
        dtypes = None
        if selection:
            try:
                # Parse the selection string into a tuple of slices
                slices = tuple(
                    (
                        slice(*map(lambda x: int(x.strip()) if x.strip() else None, dim.split(":")))
                        if ":" in dim
                        else int(dim.strip())
                    )
                    for dim in selection.split(",")
                )
                data = data[slices]
                if df is not None and len(slices) > 1:
                    dtypes = list(df.dtypes)[slices[1]]
                    if not isinstance(dtypes, list):
                        dtypes = [dtypes]
            except (ValueError, IndexError):
                print(f"Invalid selection string: {selection}")

        if dtypes is not None and any([dtype == "object" for dtype in dtypes]):
            self.data_output = None
            self.string_output = ("\n".join(data), {})
            return

        data = data.astype(np.float32)

        # Handle time_series type
        if file_type == "time_series":
            assert data.shape[1] == 2, "Invalid time series shape"
            time_series, meta = data[0], data[1]
            assert isinstance(meta, dict), "Metadata should be a dictionary"

            self.data_output = (time_series, meta)
            self.string_output = None
            return

        # Handle spectrum type
        elif file_type == "spectrum":
            freq_multiplier = self.params.spectrum.freq_multiplier.value
            freq_vector = data[0] * freq_multiplier  # Multiply the frequency values
            spectrums = data[1]

            self.data_output = (np.array(spectrums), {"freq": freq_vector})
            self.string_output = None
            return

        elif file_type == "ndarray":
            self.data_output = (data, {})
            self.string_output = None
            return

        elif file_type == "embedding_csv":
            self.data_output = (data, meta if self.params.embedding_csv.name_column.value else {})
            self.string_output = None
            return

    def file_filename_changed(self, filename):
        self.setup()
