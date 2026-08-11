import goofi


class Negate(goofi.Node):
    """Negate the input."""

    def config_input_slots(self):
        return {"data": goofi.DataType.ARRAY}

    def config_output_slots(self):
        return {"out": goofi.DataType.ARRAY}
