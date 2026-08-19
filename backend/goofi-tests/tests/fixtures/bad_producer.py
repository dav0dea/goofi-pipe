import goofi


class BadProducer(goofi.Node):
    """A node whose `producer` is not a bool."""

    OUTPUTS = {"out": goofi.DataType.ARRAY}
    # Not a bool. Reading this as False would make the node never run, with nothing anywhere saying
    # why, so the probe refuses it the way a bad slot or param declaration is refused.
    PRODUCER = "yes"

    def process(self):
        return {"out": [1.0]}
