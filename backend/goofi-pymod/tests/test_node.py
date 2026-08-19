import goofi


def test_defaults_are_empty():
    n = goofi.Node()
    assert n.INPUTS == {}
    assert n.OUTPUTS == {}
    assert n.PARAMS == {}
    assert n.PRODUCER is False
    assert n.setup() is None
    assert n.process() is None


def test_a_subclass_declares_itself_in_constants():
    class PSD(goofi.Node):
        INPUTS = {"data": goofi.DataType.ARRAY}
        OUTPUTS = {"psd": goofi.DataType.ARRAY}
        PARAMS = {"welch": {"nperseg": goofi.IntParam(256, 16, 4096)}}
        PRODUCER = True

    n = PSD()
    assert isinstance(n, goofi.Node)
    assert n.INPUTS["data"].value == "ARRAY"
    assert n.OUTPUTS["psd"].value == "ARRAY"
    assert n.PARAMS["welch"]["nperseg"].default == 256
    assert n.PRODUCER is True


def test_an_omitted_constant_falls_back_to_the_base_class():
    class OutputOnly(goofi.Node):
        OUTPUTS = {"out": goofi.DataType.ARRAY}

    n = OutputOnly()
    assert n.OUTPUTS["out"].value == "ARRAY"
    assert n.INPUTS == {}
    assert n.PARAMS == {}
    assert n.PRODUCER is False
