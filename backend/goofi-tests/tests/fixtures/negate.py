import goofi


class Negate(goofi.Node):
    """Negate the input."""

    INPUTS = {"data": goofi.DataType.ARRAY}
    OUTPUTS = {"out": goofi.DataType.ARRAY}
