import goofi


class Sources(goofi.Node):
    """Names the senders on its multi slot."""

    INPUTS = {"input": goofi.InputSlot(goofi.DataType.ARRAY, multi=True)}
    OUTPUTS = {"out": goofi.DataType.STRING}

    def process(self, input):
        return ",".join(src for src, _ in input)
