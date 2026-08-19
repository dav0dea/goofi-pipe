import goofi


class PSD(goofi.Node):
    """Power spectral density."""

    INPUTS = {"data": goofi.DataType.ARRAY}
    OUTPUTS = {"psd": goofi.DataType.ARRAY}
    PARAMS = {
        "welch": {
            "nperseg": goofi.IntParam(256, 16, 4096, doc="Window length in samples."),
            "average": goofi.BoolParam(True),
        }
    }
