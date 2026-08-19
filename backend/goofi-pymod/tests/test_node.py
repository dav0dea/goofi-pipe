import goofi


def test_defaults_are_empty():
    n = goofi.Node()
    assert n.manifest.inputs == {}
    assert n.manifest.outputs == {}
    assert n.manifest.params == {}
    assert n.manifest.producer is False
    assert n.setup() is None
    assert n.process() is None


def test_subclass_declares_one_manifest():
    class PSD(goofi.Node):
        manifest = goofi.Manifest(
            inputs={"data": goofi.DataType.ARRAY},
            outputs={"psd": goofi.DataType.ARRAY},
            params={"welch": {"nperseg": goofi.IntParam(256, 16, 4096)}},
            producer=True,
        )

    n = PSD()
    assert isinstance(n, goofi.Node)
    assert n.manifest.inputs["data"].value == "ARRAY"
    assert n.manifest.outputs["psd"].value == "ARRAY"
    assert n.manifest.params["welch"]["nperseg"].default == 256
    assert n.manifest.producer is True


def test_producer_must_be_a_bool():
    # Refused where it is WRITTEN, so a mistyped pacing declaration cannot reach the probe as a
    # node that silently never runs.
    try:
        goofi.Manifest(producer="yes")
    except TypeError:
        return
    raise AssertionError("a non-bool producer must not construct")
