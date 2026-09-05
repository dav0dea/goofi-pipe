import goofi


class BadTag(goofi.Node):
    """Declares a tag the vocabulary does not hold."""

    TAGS = ["sparkly"]
    OUTPUTS = {"out": goofi.DataType.ARRAY}
