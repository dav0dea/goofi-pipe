from copy import deepcopy

import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node


class Transpose(Node):

    def config_input_slots():
        return {"array": DataType.ARRAY}

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def config_params():
        return {"transpose": {"axes": "1,0"}}

    def process(self, array: Data):
        if array is None:
            return None

        if array.data.ndim == 1:
            # 1D -> add a leading axis: the single dim0 moves to dim1.
            result = array.data.reshape(1, -1)
            axes = [1, 0]
        else:
            axes = list(map(int, self.params.transpose.axes.value.split(",")))
            result = np.transpose(array.data, axes=axes)

        # Permute the channel names to match (output dim i sources from input dim axes[i]),
        # on a COPY — never mutate the producer's meta (fan-out aliasing, backlog #6).
        meta = deepcopy(array.meta)
        ch = meta.get("channels")
        if ch is not None:
            meta["channels"] = {
                f"dim{new_dim}": ch[f"dim{old_dim}"]
                for new_dim, old_dim in enumerate(axes)
                if f"dim{old_dim}" in ch
            }
        return {"out": (result, meta)}
