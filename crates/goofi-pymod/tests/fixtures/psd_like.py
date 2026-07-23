import goofi


class PSD(goofi.Node):
    """Power spectral density."""

    def config_input_slots(self):
        return {"data": goofi.DataType.ARRAY}

    def config_output_slots(self):
        return {"psd": goofi.DataType.ARRAY}

    def config_params(self):
        return {
            "welch": {
                "nperseg": goofi.IntParam(256, 16, 4096, doc="Window length in samples."),
                "average": goofi.BoolParam(True),
            }
        }
