import goofi


def test_datatype_values():
    assert goofi.DataType.ARRAY.value == "ARRAY"
    assert goofi.DataType.STRING.value == "STRING"
    assert goofi.DataType.TABLE.value == "TABLE"


def test_numeric_params():
    p = goofi.IntParam(256, 16, 4096)
    assert (p.default, p.min, p.max) == (256, 16, 4096)
    f = goofi.FloatParam(1.0, 0.0, 10.0)
    assert (f.default, f.min, f.max) == (1.0, 0.0, 10.0)


def test_bool_and_string_params():
    b = goofi.BoolParam(True)
    assert b.default is True
    s = goofi.StringParam("a", options=["a", "b"], refresh=True)
    assert (s.default, s.options, s.refresh) == ("a", ["a", "b"], True)
    d = goofi.StringParam("x")
    assert (d.options, d.refresh) == ([], False)
