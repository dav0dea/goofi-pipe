import goofi


class BadProducer(goofi.Node):
    """A node whose `producer` is not a bool."""

    # Not a bool. Silently reading this as False would make the node never run, with nothing
    # anywhere saying why — so `Manifest` refuses it where it is written, and the import that
    # evaluates this class body is what fails.
    manifest = goofi.Manifest(outputs={"out": goofi.DataType.ARRAY}, producer="yes")

    def process(self):
        return {"out": [1.0]}
