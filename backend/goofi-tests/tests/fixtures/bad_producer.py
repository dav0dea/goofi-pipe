import goofi


class BadProducer(goofi.Node):
    """A node whose `producer` is not a bool."""

    # Not a bool. Silently reading this as False would make the node never run once implicit
    # free-run is gone, with nothing anywhere saying why — so the probe must refuse it instead.
    producer = "yes"

    @staticmethod
    def config_output_slots():
        return {"out": goofi.DataType.ARRAY}

    def process(self):
        return {"out": [1.0]}
