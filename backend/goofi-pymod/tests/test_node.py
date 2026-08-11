import goofi


def test_defaults_are_empty():
    n = goofi.Node()
    assert n.config_input_slots() == {}
    assert n.config_output_slots() == {}
    assert n.config_params() == {}
    assert n.setup() is None
    assert n.process() is None


def test_subclass_overrides():
    class PSD(goofi.Node):
        def config_input_slots(self):
            return {"data": goofi.DataType.ARRAY}

        def config_params(self):
            return {"welch": {"nperseg": goofi.IntParam(256, 16, 4096)}}

    n = PSD()
    assert isinstance(n, goofi.Node)
    assert n.config_input_slots()["data"].value == "ARRAY"
    assert n.config_params()["welch"]["nperseg"].default == 256
