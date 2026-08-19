import sys

import goofi


class GilFlip(goofi.Node):
    """A node that re-enables the GIL from a declaration hook."""

    def config_input_slots(self):
        return {"data": goofi.DataType.ARRAY}

    def config_output_slots(self):
        return {"out": goofi.DataType.ARRAY}

    def config_params(self):
        # Stands in for the real cause: a node whose declaration-time import (see
        # device_options.py in goofi-pymod's fixtures — importing inside a hook is the
        # documented pattern) pulls in a C extension built without free-threading
        # support, which flips the GIL back on mid-probe. Flipping the very oracle the
        # probe reads keeps this deterministic and free of a wheel dependency.
        sys._is_gil_enabled = lambda: True
        return {}
