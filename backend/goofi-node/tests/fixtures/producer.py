import goofi


class Producer(goofi.Node):
    """A source that declares itself."""

    producer = True

    @staticmethod
    def config_output_slots():
        return {"out": goofi.DataType.ARRAY}

    def process(self):
        return {"out": [1.0]}
