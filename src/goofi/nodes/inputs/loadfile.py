from copy import deepcopy
from glob import glob
from pathlib import Path

import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, FloatParam, StringParam


class LoadFile(Node):

    def config_input_slots():
        return {"file": DataType.STRING, "reload": DataType.ARRAY}

    def config_output_slots():
        return {"data_output": DataType.ARRAY, "string_output": DataType.STRING}

    def config_params():
        return {
            "file": {
                "filename": StringParam("", doc="The name of the file to load with extension"),
                "type": StringParam(
                    "spectrum",
                    options=["spectrum", "time_series", "ndarray", "embedding_csv", "audio", "pickle", "image"],
                    doc="Type of file to load",
                ),
                "select": StringParam("", doc="NumPy selection string"),
                "glob_resolution": StringParam(
                    "newest", options=["newest", "oldest", "alphabet_first", "alphabet_last"]
                ),
                "reload": BoolParam(False, trigger=True, doc="Force reload the file"),
                "require_reload": BoolParam(False, doc="Only return the data if a file was newly loaded"),
            },
            "spectrum": {"freq_multiplier": FloatParam(1.0, doc="Multiplier to adjust the frequency values")},
            "embedding_csv": {"header": 0, "name_column": False, "index_column": True},
            "common": {"autotrigger": True, "max_frequency": 1.0},
        }

    def setup(self):
        import librosa
        import pandas as pd
        from PIL import Image

        self.pd = pd
        self.librosa = librosa
        self.Image = Image

        self.data_output = None
        self.string_output = None
        self.last_params = None
        self.last_mtime = None
        self.resolved_filename = None

    def process(self, file: Data, reload: Data):
        self.just_loaded = False

        if (reload is not None and np.any(reload.data > 0)) or self.params.file.reload.value:
            self.input_slots["reload"].clear()
            self.data_output = None
            self.string_output = None
            self.last_params = None
            self.last_mtime = None
            self.resolved_filename = None

        if file is not None:
            self.params.file.filename.value = file.data
            self.file_filename_changed(file.data)
            self.input_slots["file"].clear()

        # Check if file has been modified on disk
        if self.last_mtime is not None and self.resolved_filename:
            try:
                current_mtime = Path(self.resolved_filename).stat().st_mtime
                if current_mtime > self.last_mtime:
                    self.data_output = None
                    self.string_output = None
                    self.last_params = None
                    self.last_mtime = None
                    self.resolved_filename = None
            except (FileNotFoundError, OSError):
                pass

        if self.last_params == self.params and not (self.data_output is None and self.string_output is None):
            if self.params.file.require_reload.value:
                return {"data_output": None, "string_output": None}
            else:
                return {"data_output": self.data_output, "string_output": self.string_output}

        # if the parameters are different, load the file
        self.last_params = deepcopy(self.params)
        self.load_file()
        self.just_loaded = True

        # Set last_mtime after successful load
        if (self.data_output is not None or self.string_output is not None) and self.resolved_filename:
            self.last_mtime = Path(self.resolved_filename).stat().st_mtime

        if self.params.file.require_reload.value and not self.just_loaded:
            return {"data_output": None, "string_output": None}
        else:
            return {"data_output": self.data_output, "string_output": self.string_output}

    def load_file(self):
        if not self.params.file.filename.value:
            self.data_output = None
            self.string_output = None
            self.last_mtime = None
            self.resolved_filename = None
            return

        file_type = self.params.file.type.value
        filename = self.params.file.filename.value

        matches = glob(filename)
        if len(matches) == 0:
            raise FileNotFoundError(f"File does not exist or no matches found: {filename}")
        elif len(matches) == 1:
            filename = matches[0]
        else:
            glob_resolution = self.params.file.glob_resolution.value
            if glob_resolution == "newest":
                filename = max(matches, key=lambda f: Path(f).stat().st_mtime)
            elif glob_resolution == "oldest":
                filename = min(matches, key=lambda f: Path(f).stat().st_mtime)
            elif glob_resolution == "alphabet_first":
                filename = min(matches)
            elif glob_resolution == "alphabet_last":
                filename = max(matches)
            else:
                filename = matches[0]

        self.resolved_filename = filename

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

        elif file_type == "image":
            try:
                img = self.Image.open(filename)
                img_array = np.array(img) / 255.0
                self.data_output = (
                    img_array.astype(np.float32),
                    {"height": img.height, "width": img.width, "img_channels": len(img.getbands())},
                )
                self.string_output = None
            except Exception as e:
                print(f"Error loading image file: {e}")
                self.data_output = None
                self.string_output = None
            return

        extension = filename.split(".")[-1]

        df, data = None, None
        if extension == "npy":
            data = np.load(f"{filename}", allow_pickle=True)
        elif extension == "txt":
            data = np.loadtxt(f"{filename}")
        elif extension == "csv":
            header = self.params.embedding_csv.header.value
            df = self.pd.read_csv(
                f"{filename}",
                header=None if header < 0 else header,
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

        self.resolved_filename = filename  # Store the resolved filename

    def file_filename_changed(self, filename):
        self.setup()
