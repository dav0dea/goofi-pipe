import sys

import goofi

# Stands in for the real cause: a node whose declaration-time import pulls in a C extension built
# without free-threading support, which flips the GIL back on while the module is being imported.
# Flipping the very oracle the probe reads keeps this deterministic and free of a wheel dependency.
sys._is_gil_enabled = lambda: True


class GilFlip(goofi.Node):
    """A node that re-enables the GIL while its module is imported."""

    INPUTS = {"data": goofi.DataType.ARRAY}
    OUTPUTS = {"out": goofi.DataType.ARRAY}
