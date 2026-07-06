from abc import ABC, abstractmethod
from collections import namedtuple
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class Param(ABC):
    """
    Parameter container that has a specific type and potentially constrains the range of allowed values.
    """

    _value: Any = None

    # Common attributes shared by every Param subclass. Declared KEYWORD-ONLY so a
    # base-class field with a default doesn't shift each subclass's positional
    # fields (value, vmin/vmax, options, trigger, ...). `expression` is the source
    # (None if unauthored); `expression_enabled` gates evaluation — toggling the fx
    # button off keeps the source so the user can flip back. The cadence flags ride
    # along (see normalize_expression_binding).
    doc: Optional[str] = field(default=None, kw_only=True)
    save_param: bool = field(default=True, kw_only=True)
    expression: Optional[str] = field(default=None, kw_only=True)
    expression_enabled: bool = field(default=False, kw_only=True)
    expression_triggers_process: bool = field(default=False, kw_only=True)
    expression_autoeval: bool = field(default=False, kw_only=True)

    def __post_init__(self):
        if self._value is None:
            self._value = self.default()

        if not isinstance(self._value, type(self.default())):
            if isinstance(self.default(), float) and isinstance(self._value, int):
                # it's okay if we wanted a float but got int
                self._value = float(self._value)
            else:
                raise TypeError(
                    f"Parameter {self.__class__.__name__} expected type {type(self.default())} but got {type(self._value)}."
                )

    @staticmethod
    @abstractmethod
    def default() -> Any:
        """
        Return the default value for the parameter.
        """
        pass

    @property
    def value(self) -> Any:
        return self._value

    @value.setter
    def value(self, value: Any):
        self._value = value

    def copy(self):
        """
        Return a deep copy of the parameter object.
        """
        return deepcopy(self)


@dataclass
class BoolParam(Param):
    trigger: bool = False

    @staticmethod
    def default() -> bool:
        return False

    @property
    def value(self) -> bool:
        """If trigger is True, return the current value and reset it to False."""
        val = self._value
        if self.trigger:
            self._value = False
        return val

    @value.setter
    def value(self, value: bool):
        self._value = value


@dataclass
class FloatParam(Param):
    vmin: float = 0.0
    vmax: float = 1.0

    @staticmethod
    def default() -> float:
        return 0.0


@dataclass
class IntParam(Param):
    vmin: int = -1
    vmax: int = 3

    @staticmethod
    def default() -> int:
        return 0


@dataclass
class StringParam(Param):
    # if options is None, the string is free-form, otherwise it is a dropdown
    options: List[str] = None

    @staticmethod
    def default() -> str:
        return ""


def normalize_expression_binding(expression, enabled, triggers_process, autoeval):
    """Canonical expression-binding gating rule — the SINGLE source of truth.

    An empty / whitespace-only source counts as "no source"; the engine is ACTIVE
    only when there is a source AND the user enabled it (toggling off keeps the
    source so it isn't lost). Returns the normalized
    ``(source, active, triggers_process, autoeval)`` tuple. This is purposely PURE
    (it writes nothing): the node applies it to its Param as the authority, while
    the manager-side proxy uses it only to shape the outgoing RPC — so the rule
    can never diverge between the two sides.
    """
    source = expression if (expression and expression.strip()) else None
    active = bool(enabled) and source is not None
    return source, active, bool(triggers_process), bool(autoeval)


DEFAULT_PARAMS = {
    "common": {
        "autotrigger": BoolParam(
            False,
            doc=(
                "If set, the node will run its processing function automatically "
                "at the specified Max Frequency, even if none of the inputs have changed."
            ),
        ),
        "max_frequency": FloatParam(
            30.0, 0.0, 60.0, doc="The maximum frequency at which the node will run its processing function."
        ),
        "frequency_mode": StringParam(
            "updates-per-second",
            options=["updates-per-second", "seconds-per-update"],
            doc=(
                "The frequency mode of the node. If 'updates-per-second', the node will run at the specified Max Frequency.\n"
                "If 'seconds-per-update', the node will run once every specified number of seconds."
            ),
        ),
        "process_group": StringParam(
            "",
            doc=(
                "If empty, spawn the node in its own process. If non-empty, register the node to the process with this name "
                "(restart required to take effect)."
            ),
        ),
    },
}

TYPE_PARAM_MAP = {
    bool: BoolParam,
    float: FloatParam,
    int: IntParam,
    str: StringParam,
}


class InvalidParamError(Exception):
    pass


def _split_saved(saved: Any) -> tuple:
    """Pull (value, expression, enabled, triggers_process, autoeval) from
    a saved param entry.

    Accepts the rich ``{value, expression, ...}`` dict shape and falls
    back to treating ``saved`` as a flat scalar otherwise. The flag
    fields default to False / None when absent so legacy .gfi files
    keep their behavior. ``enabled`` defaults to True when a saved
    expression is present (older .gfi files predate the flag).
    """
    if isinstance(saved, dict) and "value" in saved and "_value" not in saved:
        expr = saved.get("expression")
        return (
            saved["value"],
            expr,
            bool(saved.get("expression_enabled", expr is not None)),
            bool(saved.get("expression_triggers_process", False)),
            bool(saved.get("expression_autoeval", False)),
        )
    return saved, None, False, False, False


class NodeParams:
    """
    A class for storing node parameters, which store the configuration state of the node, and are exposed
    to the user through the GUI. The parameters are stored in named groups with each group containing a
    number of parameters. The parameters are stored as named tuples, and can be accessed as attributes.

    When initializing a `NodeParams` object, a set of default parameters is inserted if they are not provided.

    ### Parameters
    `data` : Dict[str, Dict[str, Any]]
        A dictionary of parameter groups, where each group is a dictionary of parameter names and values.
    """

    def __init__(self, data: Dict[str, Dict[str, Any]]):
        # insert default parameters if they are not present
        for group_name, group in deepcopy(DEFAULT_PARAMS).items():
            if group_name not in data:
                # group is not present, insert it
                data[group_name] = group
            else:
                # group is present, insert missing parameters
                for param_name, param in group.items():
                    if param_name in data[group_name]:
                        # make sure we have a Param and not just a deserialized value
                        saved = data[group_name][param_name]
                        if not isinstance(saved, Param):
                            value, expression, enabled, trig, auto = _split_saved(saved)
                            param._value = value
                            if expression is not None:
                                param.expression = expression
                                param.expression_enabled = enabled
                                param.expression_triggers_process = trig
                                param.expression_autoeval = auto
                            data[group_name][param_name] = param
                    else:
                        data[group_name][param_name] = param

        # convert values to Param objects
        for group, params in data.items():
            if not isinstance(params, dict):
                raise TypeError(f"Expected dict, got {type(params)}.")
            for param_name, param in params.items():
                if not isinstance(param, Param):
                    if isinstance(param, dict):
                        # `{value, expression, ...}` is the expression-aware
                        # shape; `{_value, ...}` is the legacy goofi-patch
                        # reconstruction.
                        if "value" in param and "_value" not in param:
                            value, expression, enabled, trig, auto = _split_saved(param)
                            param_type = TYPE_PARAM_MAP[type(value)]
                            new_param = param_type(value)
                            new_param.expression = expression
                            new_param.expression_enabled = enabled
                            new_param.expression_triggers_process = trig
                            new_param.expression_autoeval = auto
                            data[group][param_name] = new_param
                            continue
                        # reconstruct serialized param object (legacy)
                        param_type = TYPE_PARAM_MAP[type(param["_value"])]
                        data[group][param_name] = param_type(**param)
                        continue

                    # convert to Param object
                    if type(param) not in TYPE_PARAM_MAP:
                        raise TypeError(
                            f"Invalid parameter type {type(param).__name__}. Must be one of "
                            f"{list(map(lambda x: x.__name__, TYPE_PARAM_MAP.keys()))}."
                        )
                    data[group][param_name] = TYPE_PARAM_MAP[type(param)](param)

        # convert to named tuples
        self._data = self._generate_data_dict(data)

    def _generate_data_dict(self, data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Convert the parameter dictionary to a dictionary of named tuples.

        ### Parameters
        `data` : Dict[str, Dict[str, Any]]
            A dictionary of parameter groups, where each group is a dictionary of parameter names and values.

        ### Returns
        `Dict[str, Dict[str, Any]]`
            A dictionary of parameter groups, where each group is a dictionary of parameter names and values.
        """
        # convert to named tuples
        result = {}
        for group, params in data.items():
            # create the named tuple class for the current group
            NamedTupleClass = namedtuple(group.capitalize(), params.keys())

            # implement __contains__ for the named tuple class
            NamedTupleClass = type(
                NamedTupleClass.__name__,
                (NamedTupleClass,),
                {
                    "__contains__": lambda self, item: hasattr(self, item),
                    "__getitem__": lambda self, item: getattr(self, item),
                    "keys": lambda self: self._asdict().keys(),
                    "values": lambda self: self._asdict().values(),
                    "items": lambda self: self._asdict().items(),
                },
            )

            result[group] = NamedTupleClass(**params)
        return result

    def update_options(self, options: Dict[str, Dict[str, list]]) -> None:
        """Refresh StringParam option lists from a node's authoritative report
        (runtime-enumerated device lists ride STATE_UPDATE as `param_options`).

        Only params that DECLARED options are refreshed — a free-form
        StringParam stays free-form. Unknown groups/names and empty lists are
        ignored: schema drift must not crash the manager-side ref."""
        for group, names in options.items():
            if group not in self._data:
                continue
            for name, opts in names.items():
                if name not in self._data[group]._fields:
                    continue
                param = self._data[group][name]
                if isinstance(param, StringParam) and param.options is not None and opts:
                    param.options = list(opts)

    def update(self, params: Dict[str, Dict[str, Any]]):
        """
        Update the parameters with new values.

        ### Parameters
        `params` : Dict[str, Dict[str, Any]]
            A dictionary of parameter groups, where each group is a dictionary of parameter names and values.
        """
        encountered_invalid_params = False

        # TODO: avoid code duplication with __init__
        for group, params in params.items():
            if group not in self._data:
                encountered_invalid_params = True
                print(f"Invalid parameter group '{group}'.")
                continue

            for name, param in params.items():
                if name not in self._data[group]._fields:
                    encountered_invalid_params = True
                    print(f"Invalid parameter '{name}' in group '{group}'.")
                    continue

                if not isinstance(param, Param):
                    if isinstance(param, dict):
                        # New {value, expression, ...} shape — preserve all
                        # type-specific fields (vmin/vmax/options) on the
                        # existing Param by mutating in place.
                        if "value" in param and "_value" not in param:
                            existing = self._data[group][name]
                            value, expression, enabled, trig, auto = _split_saved(param)
                            existing._value = value
                            existing.expression = expression
                            existing.expression_enabled = enabled
                            existing.expression_triggers_process = trig
                            existing.expression_autoeval = auto
                            continue
                        # !!! LOADING PARAMS FROM DICT IS A LEGACY FEATURE TO LOAD OLD GOOFI PATCHES !!!
                        # reconstruct serialized param object
                        param_type = type(self._data[group][name])
                        self._data[group] = self._data[group]._replace(**{name: param_type(**param)})
                        continue

                    # update only the value of the parameter, leave the rest unchanged
                    if not isinstance(self._data[group][name], TYPE_PARAM_MAP[type(param)]):
                        # it's okay if we wanted a float but got int
                        if isinstance(self._data[group][name], FloatParam) and isinstance(param, int):
                            param = float(param)
                        else:
                            raise TypeError(
                                f"Expected parameter type {TYPE_PARAM_MAP[type(param)].__name__} but got "
                                f"{type(self._data[group][name]).__name__}."
                            )
                    self._data[group][name]._value = param
                else:
                    # update the entire parameter object
                    self._data[group] = self._data[group]._replace(**{name: param})

        if encountered_invalid_params:
            raise InvalidParamError("Invalid parameters encountered.")

    def serialize(self) -> Dict[str, Dict[str, Any]]:
        """
        Serialize the parameter values to a dictionary.

        ### Returns
        `Dict[str, Dict[str, Any]]`
            A dictionary of parameter groups, where each group is a dictionary of parameter names and values.
        """
        serialized_data = {}
        for group, params in self._data.items():
            serialized_params = {}
            for name, param in params.items():
                if param.save_param:
                    expr = getattr(param, "expression", None)
                    if expr is not None:
                        # Asymmetric on purpose: dict shape only when an
                        # expression is bound, flat value otherwise. Keeps
                        # legacy .gfi files byte-identical in git.
                        serialized_params[name] = {
                            "value": param._value,
                            "expression": expr,
                            "expression_enabled": bool(
                                getattr(param, "expression_enabled", False)
                            ),
                            "expression_triggers_process": bool(
                                getattr(param, "expression_triggers_process", False)
                            ),
                            "expression_autoeval": bool(
                                getattr(param, "expression_autoeval", False)
                            ),
                        }
                    else:
                        serialized_params[name] = param._value
            serialized_data[group] = serialized_params
        return serialized_data

    def keys(self):
        return self._data.keys()

    def __getattr__(self, group: str):
        # don't allow access to the _data attribute
        if group == "_data":
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{group}'")

        # return the group if it exists
        if group in self._data:
            return self._data[group]
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{group}'")

    def __contains__(self, group: str) -> bool:
        return group in self._data

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NodeParams):
            return False
        return self._data == other._data

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(map(repr,self._data.values()))})"

    def __getitem__(self, group: Union[int, str]):
        if isinstance(group, int):
            return list(self._data.keys())[group]
        return self._data[group]

    def __len__(self) -> int:
        return len(self._data)

    def __getstate__(self):
        return {"_data": {k: v._asdict() for k, v in self._data.items()}}

    def __setstate__(self, state):
        self._data = self._generate_data_dict(state["_data"])
