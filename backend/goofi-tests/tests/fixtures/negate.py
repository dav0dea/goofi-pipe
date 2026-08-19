import goofi


class Negate(goofi.Node):
    """Negate the input."""

    manifest = goofi.Manifest(
        inputs={"data": goofi.DataType.ARRAY},
        outputs={"out": goofi.DataType.ARRAY},
    )
