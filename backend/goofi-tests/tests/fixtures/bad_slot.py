import goofi


class BadSlot(goofi.Node):
    """Declares a slot named by a Python keyword."""

    INPUTS = {"in": goofi.DataType.ARRAY}
    OUTPUTS = {"out": goofi.DataType.ARRAY}
