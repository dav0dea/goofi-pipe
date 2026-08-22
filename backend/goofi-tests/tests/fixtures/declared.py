import goofi


class Declared(goofi.Node):
    """Declares its input slots rather than taking the defaults."""

    INPUTS = {
        "bare": goofi.DataType.ARRAY,
        "needed": goofi.InputSlot(goofi.DataType.ARRAY, required=True),
        "passive": goofi.InputSlot(goofi.DataType.ARRAY, trigger=False),
    }
    OUTPUTS = {"out": goofi.DataType.ARRAY}
