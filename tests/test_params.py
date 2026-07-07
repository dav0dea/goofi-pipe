import pytest
import yaml

from goofi import params
from goofi.params import (
    DEFAULT_PARAMS,
    BoolParam,
    FloatParam,
    IntParam,
    NodeParams,
    StringParam,
    coerce_to_param_type,
)

from .utils import list_param_types


def test_coerce_to_param_type_truncates_and_rejects():
    # int field gets a fractional value -> truncate; float gets an int -> widen;
    # a non-numeric string into an int field is uncoercible -> None (skip).
    assert coerce_to_param_type(IntParam(0), 2.5) == 2
    assert coerce_to_param_type(FloatParam(0.0), 3) == 3.0
    assert coerce_to_param_type(StringParam(""), 5) == "5"
    assert coerce_to_param_type(IntParam(0), 4) == 4
    assert coerce_to_param_type(IntParam(0), "abc") is None


def test_nodeparams_update_truncates_float_into_int_param():
    # A fractional value must NOT poison an IntParam and make the saved patch
    # unloadable — it is truncated, and a serialize()->load round-trip is clean.
    p = NodeParams({"g": {"n": IntParam(4, 0, 10)}})
    p.update({"g": {"n": 2.5}})
    assert p["g"]["n"].value == 2
    assert p.serialize()["g"]["n"] == 2


def test_nodeparams_update_contains_uncoercible_value():
    # One bad param must not abort the whole update loop (the STATE_UPDATE mirror
    # / .gfi load path): later params still apply, and the bad one is left as-is.
    p = NodeParams({"g": {"n": IntParam(4, 0, 10), "m": IntParam(7, 0, 10)}})
    with pytest.raises(params.InvalidParamError):
        p.update({"g": {"n": "not a number", "m": 9}})
    assert p["g"]["n"].value == 4
    assert p["g"]["m"].value == 9


def test_trigger_boolparam_serializes_false_even_when_currently_true():
    # A used-but-not-yet-consumed trigger must persist as its resting False, so a
    # saved patch does not silently re-fire the trigger on every load.
    p = NodeParams({"g": {"fire": BoolParam(True, trigger=True), "flag": BoolParam(True)}})
    ser = p.serialize()
    assert ser["g"]["fire"] is False
    assert ser["g"]["flag"] is True


def test_trigger_boolparam_with_expression_serializes_value_false():
    p = NodeParams({"g": {"fire": BoolParam(True, trigger=True)}})
    fire = p["g"]["fire"]
    fire.expression = "True"
    fire.expression_enabled = True
    ser = p.serialize()
    assert ser["g"]["fire"]["value"] is False


def full_node_params(n_groups: int = 2) -> NodeParams:
    """
    Returns a NodeParams object with all parameter types.

    ### Parameters
    `n_groups` : int
        The number of parameter groups to create.

    ### Returns
    NodeParams
        A NodeParams object with all parameter types.
    """
    params = {}
    for i in range(n_groups):
        params[f"group{i}"] = {}
        for param_type in list_param_types():
            params[f"group{i}"][f"param{param_type.__name__}"] = param_type.default()
    return NodeParams(params)


def test_default_members():
    assert "common" in DEFAULT_PARAMS, "Missing default group 'common'."
    # autotrigger
    assert "autotrigger" in DEFAULT_PARAMS["common"], "Missing default param 'autotrigger' in group 'common'."
    assert isinstance(
        DEFAULT_PARAMS["common"]["autotrigger"], params.BoolParam
    ), "Wrong type for default param 'autotrigger' in group 'common'"
    # max_frequency
    assert "max_frequency" in DEFAULT_PARAMS["common"], "Missing default param 'max_frequency' in group 'common'."
    assert isinstance(
        DEFAULT_PARAMS["common"]["max_frequency"], params.FloatParam
    ), "Wrong type for default param 'max_frequency' in group 'common'."


@pytest.mark.parametrize("param_type", list_param_types())
def test_param_types(param_type):
    p = param_type()
    assert p.value == param_type.default(), f"Wrong default value for param type {param_type.__name__}."

    for type2 in list_param_types():
        if type2 is param_type:
            continue

        if isinstance(type2.default(), type(param_type.default())):
            # some types may be instances of other types, e.g. bool is an instance of int
            continue

        if isinstance(param_type.default(), float) and isinstance(type2.default(), int):
            # we accept int as default value for float
            continue

        with pytest.raises(TypeError):
            param_type(type2.default())


def test_node_params():
    params = NodeParams(DEFAULT_PARAMS)

    try:
        params.common
    except AttributeError:
        pytest.fail("NodeParams should support attribute access (e.g. params.common).")
    try:
        params.common.autotrigger
    except AttributeError:
        pytest.fail("Parameter groups should support attribute access (e.g. params.common.autotrigger).")

    with pytest.raises(AttributeError):
        params.nonexistent

    for name, group in DEFAULT_PARAMS.items():
        assert getattr(params, name)._asdict() == group, f"Wrong parameter group {name} in NodeParams."


def test_node_params_repr():
    repr(NodeParams(DEFAULT_PARAMS))


@pytest.mark.parametrize("param_type", list_param_types())
def test_type_param_map(param_type):
    assert any(
        param_type == t for t in params.TYPE_PARAM_MAP.values()
    ), f"Param type '{param_type.__name__}' is not in TYPE_PARAM_MAP."


@pytest.mark.parametrize("type_cls", params.TYPE_PARAM_MAP.keys())
def test_raw_value(type_cls):
    data = {"test": {"param": type_cls()}}
    NodeParams(data)


def test_node_params_iter():
    params = DEFAULT_PARAMS.copy()
    params["test"] = {"param": True}

    p = NodeParams(params)
    assert len(p) == 2, "NodeParams should have 2 groups."

    it = iter(p)
    assert next(it) == "common", "First group should be 'common'."
    assert next(it) == "test", "Second group should be 'test'."
    with pytest.raises(StopIteration):
        next(it)

    expected = ["common", "test"]
    groups = list(p)
    assert groups == expected, "NodeParams should have groups 'common' and 'test'."

    for i, group in enumerate(p):
        assert group == expected[i], f"NodeParams group {i} should be '{expected[i]}'."


def test_node_params_contains():
    params = DEFAULT_PARAMS.copy()
    params["test"] = {"param": True}

    p = NodeParams(params)
    assert "common" in p, "NodeParams should contain group 'common'."
    assert "test" in p, "NodeParams should contain group 'test'."
    assert "nonexistent" not in p, "NodeParams should not contain group 'nonexistent'."


def test_node_params_getitem():
    p = NodeParams(DEFAULT_PARAMS)
    assert p[0] == "common", "NodeParams group 0 should be 'common'."
    assert p["common"] == p.common, "NodeParams group 'common' should be accessible by name."


def test_param_group_getitem():
    p = NodeParams(DEFAULT_PARAMS)
    assert p.common["autotrigger"].value is False, "NodeParams group 'common' should have the correct value for 'autotrigger'."


def test_param_group_keys():
    p = NodeParams(DEFAULT_PARAMS)
    assert p.common.keys() == DEFAULT_PARAMS["common"].keys(), "NodeParams group 'common' should have the correct keys."


def test_param_group_values():
    p = NodeParams(DEFAULT_PARAMS)
    assert isinstance(
        p.common.values(), type(DEFAULT_PARAMS["common"].values())
    ), "NodeParams group 'common' should have a dict_values object."
    assert list(p.common.values()) == list(
        DEFAULT_PARAMS["common"].values()
    ), "NodeParams group 'common' should have the correct values."


def test_param_group_items():
    p = NodeParams(DEFAULT_PARAMS)
    assert p.common.items() == DEFAULT_PARAMS["common"].items(), "NodeParams group 'common' should have the correct items."


def test_serialize():
    p = full_node_params()

    # make sure the serialized data can be converted to YAML
    serialized = p.serialize()
    serialized_str = yaml.dump(serialized, sort_keys=False)

    # reconstruct from the serialized data
    reconstructed = yaml.load(serialized_str, Loader=yaml.FullLoader)
    p2 = NodeParams(reconstructed)

    # make sure the reconstructed object is equal to the original
    assert p == p2, "NodeParams should be equal after serialization and reconstruction."


def test_update():
    p = full_node_params()
    p.update({"group0": {"paramFloatParam": 5.0}})
    assert p.group0.paramFloatParam.value == 5.0, "NodeParams should update the value of a parameter."

    p.update({"group0": {"paramFloatParam": FloatParam(-3.0)}})
    assert p.group0.paramFloatParam.value == -3.0, "NodeParams should update the value of a parameter."


@pytest.mark.parametrize("param_type", list_param_types())
def test_update_types(param_type):
    p = full_node_params()
    p.update({"group0": {"param" + param_type.__name__: param_type.default()}})


@pytest.mark.parametrize("param_type", list_param_types())
def test_doc(param_type):
    p = param_type()
    assert p.doc is None, f"Param type {param_type.__name__} should have no doc string."

    p = param_type(doc="test")
    assert p.doc == "test", f"Param type {param_type.__name__} should have the correct doc string."

    p = param_type(param_type.default(), doc="test")
    assert p.doc == "test", f"Param type {param_type.__name__} should have the correct doc string."


@pytest.mark.parametrize("trigger", [True, False])
def test_toggle_bool_param(trigger):
    p = BoolParam(True, trigger=trigger)
    assert p.value, "BoolParam should have the correct value."
    assert p.value == (not trigger), "Accessing BoolParam should set the value to False if trigger is True."


# ---- expression-binding gating rule (single source of truth) -----------------

def test_normalize_expression_binding_empty_source_is_inactive():
    from goofi.params import normalize_expression_binding

    # empty / whitespace-only source counts as "no source" -> inactive
    assert normalize_expression_binding("", True, False, False) == (None, False, False, False)
    assert normalize_expression_binding("   ", True, False, False) == (None, False, False, False)
    assert normalize_expression_binding(None, True, False, False) == (None, False, False, False)


def test_normalize_expression_binding_active_requires_source_and_enabled():
    from goofi.params import normalize_expression_binding

    # a real source + enabled -> active
    assert normalize_expression_binding("x*2", True, False, False) == ("x*2", True, False, False)
    # a real source but disabled -> source kept, inactive (toggle off without losing it)
    assert normalize_expression_binding("x*2", False, False, False) == ("x*2", False, False, False)


def test_normalize_expression_binding_coerces_flag_types():
    from goofi.params import normalize_expression_binding

    src, active, trig, auto = normalize_expression_binding("a+b", 1, 1, 0)
    assert (src, active, trig, auto) == ("a+b", True, True, False)
    assert isinstance(active, bool) and isinstance(trig, bool) and isinstance(auto, bool)


def test_common_and_expression_fields_are_keyword_only():
    # The shared fields (doc/save_param/expression*) must be KEYWORD-ONLY so they
    # never consume a subclass's positional slots (value, vmin, vmax, ...). This is
    # the contract the kw_only fields preserve (previously via an __init__ monkey-patch).
    from dataclasses import asdict

    p = FloatParam(2.5, -1.0, 10.0, doc="d", expression="x*2", expression_enabled=True)
    assert (p.value, p.vmin, p.vmax) == (2.5, -1.0, 10.0)  # positionals untouched
    assert p.doc == "d" and p.expression == "x*2" and p.expression_enabled is True
    assert p.save_param is True and p.expression_autoeval is False  # defaults

    # asdict carries all fields and **dict reconstructs (the legacy .gfi load path)
    p2 = FloatParam(**asdict(p))
    assert p2 == p

    # the extras are NOT positional: a 4th positional would be a TypeError
    with pytest.raises(TypeError):
        FloatParam(2.5, -1.0, 10.0, "doc-as-positional")


def test_update_options_refreshes_stringparam():
    # The node process is the authority on runtime-enumerated options (device
    # lists): update_options merges its report into the manager-side params.
    from goofi.params import NodeParams, StringParam

    p = NodeParams({"g": {"device": StringParam("default", options=["default"]), "free": StringParam("x")}})
    p.update_options({"g": {"device": ["hw:0", "hw:1"]}})
    assert p["g"]["device"].options == ["hw:0", "hw:1"]
    # unknown groups/names and empty lists are ignored (schema drift must not crash)
    p.update_options({"g": {"nonexistent": ["x"], "device": []}, "nope": {"a": ["b"]}})
    assert p["g"]["device"].options == ["hw:0", "hw:1"]
    # a free-form StringParam (options=None) stays free-form
    p.update_options({"g": {"free": ["a"]}})
    assert p["g"]["free"].options is None
