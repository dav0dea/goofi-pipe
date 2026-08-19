import goofi


class PSD(goofi.Node):
    """Power spectral density."""

    manifest = goofi.Manifest(
        inputs={"data": goofi.DataType.ARRAY},
        outputs={"psd": goofi.DataType.ARRAY},
        params={
            "welch": {
                "nperseg": goofi.IntParam(256, 16, 4096, doc="Window length in samples."),
                "average": goofi.BoolParam(True),
            }
        },
    )
