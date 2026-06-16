import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, IntParam, StringParam

MONTAGE_OPTIONS = [
    "standard_1005",
    "standard_1020",
    "standard_alphabetic",
    "standard_postfixed",
    "standard_prefixed",
    "standard_primed",
    "biosemi16",
    "biosemi32",
    "biosemi64",
    "biosemi128",
    "biosemi160",
    "biosemi256",
    "easycap-M1",
    "easycap-M10",
    "EGI_256",
    "GSN-HydroCel-32",
    "GSN-HydroCel-64_1.0",
    "GSN-HydroCel-65_1.0",
    "GSN-HydroCel-128",
    "GSN-HydroCel-129",
    "GSN-HydroCel-256",
    "GSN-HydroCel-257",
    "mgh60",
    "mgh70",
    "artinis-octamon",
    "artinis-brite23",
]


class ChannelPositions(Node):
    """
    Looks up 3D electrode positions for the channel names of an input array using an MNE standard montage.
    The axis carrying the channel names is configurable. If normalization is enabled, the entire montage is
    centered at the origin and rescaled so that the largest spatial extent spans [-1, 1], preserving the
    aspect ratios on the smaller axes.

    Inputs:
    - array: An array whose channel names along the selected axis are looked up in the montage.

    Outputs:
    - positions: An (n_channels, 3) array of (x, y, z) electrode positions. Channels not present in the
      montage are filled with NaN.
    """

    def config_input_slots():
        return {"array": DataType.ARRAY}

    def config_output_slots():
        return {"positions": DataType.ARRAY}

    def config_params():
        return {
            "montage": {
                "name": StringParam("standard_1005", options=MONTAGE_OPTIONS),
                "axis": IntParam(0, vmin=-5, vmax=5, doc="Axis along which channel names are stored in the input meta."),
                "normalize": BoolParam(
                    True,
                    doc=(
                        "Center the montage at the origin and rescale so the largest spatial extent spans [-1, 1], "
                        "preserving aspect ratios on the smaller axes."
                    ),
                ),
            }
        }

    def setup(self):
        import mne

        self.mne = mne
        self._cache = {}

    def _get_positions(self, name, normalize):
        key = (name, normalize)
        if key in self._cache:
            return self._cache[key]

        montage = self.mne.channels.make_standard_montage(name)
        pos_dict = {ch: np.asarray(p, dtype=np.float64) for ch, p in montage.get_positions()["ch_pos"].items()}

        if normalize and pos_dict:
            coords = np.stack(list(pos_dict.values()), axis=0)
            valid = np.isfinite(coords).all(axis=1)
            if np.any(valid):
                lo = coords[valid].min(axis=0)
                hi = coords[valid].max(axis=0)
                center = (lo + hi) / 2.0
                scale = np.max((hi - lo) / 2.0)
                if scale > 0:
                    pos_dict = {ch: (p - center) / scale for ch, p in pos_dict.items()}
                else:
                    pos_dict = {ch: p - center for ch, p in pos_dict.items()}

        self._cache[key] = pos_dict
        return pos_dict

    def process(self, array: Data):
        if array is None:
            return None

        axis = self.params.montage.axis.value
        if axis < 0:
            axis = array.data.ndim + axis
        if axis < 0 or axis >= array.data.ndim:
            raise ValueError(
                f"Axis {self.params.montage.axis.value} is out of range for array of ndim {array.data.ndim}."
            )

        ch_key = f"dim{axis}"
        channels = array.meta.get("channels", {}).get(ch_key)
        if not channels:
            raise ValueError(f"Input array does not have channel names along axis {axis} (meta 'channels.{ch_key}').")

        pos_dict = self._get_positions(
            self.params.montage.name.value,
            self.params.montage.normalize.value,
        )

        positions = np.full((len(channels), 3), np.nan, dtype=np.float64)
        for i, ch in enumerate(channels):
            if ch in pos_dict:
                positions[i] = pos_dict[ch]

        meta = {"channels": {"dim0": list(channels), "dim1": ["x", "y", "z"]}}
        return {"positions": (positions, meta)}

    def montage_name_changed(self, _):
        self._cache = {}

    def montage_normalize_changed(self, _):
        self._cache = {}
