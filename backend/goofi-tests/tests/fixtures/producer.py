import goofi


class Producer(goofi.Node):
    """A source that declares itself."""

    manifest = goofi.Manifest(outputs={"out": goofi.DataType.ARRAY}, producer=True)

    def process(self):
        return {"out": [1.0]}
