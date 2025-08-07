from abc import ABC, abstractmethod
from collections import namedtuple
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Union


@dataclass
class Param(ABC):
    """
    Parameter container that has a specific type and potentially constrains the range of allowed values.
    """

    _value: Any = None

    # NOTE: The `doc` and `save_param` attributes are added to the subclasses via monkey-patching.
    # The kw_only=True argument is not supported in Python<3.10, so we have to use this hacky solution.
    # doc: str = field(default=None, kw_only=True)
    # save_param: bool = field(default=True, kw_only=True)

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


# NOTE: Monkey-patching the `doc` and `save_param` attribute into the Param subclasses is a hacky solution
# to include keyword-only arguments in the __init__ method of the subclasses. Python>=3.10 can use the
# kw_only=True argument in the field decorator.


# decorator for adjusting the __init__ method of the Param subclasses
def adjusted_init(original_init):
    def new_init(self, *args, **kwargs):
        self.doc = kwargs.pop("doc", None)
        self.save_param = kwargs.pop("save_param", True)
        original_init(self, *args, **kwargs)

    return new_init


# monkey-patch the 'doc' attribute into the classes
def add_extra_attributes(cls):
    # doc attribute
    setattr(cls, "doc", None)
    cls.__dataclass_fields__["doc"] = field(default=None)
    # save_param attribute
    setattr(cls, "save_param", True)
    cls.__dataclass_fields__["save_param"] = field(default=True)

    # adjust the __init__ method
    cls.__init__ = adjusted_init(cls.__init__)


# apply the monkey-patching
add_extra_attributes(Param)
for subclass in Param.__subclasses__():
    add_extra_attributes(subclass)


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
                        if not isinstance(data[group_name][param_name], Param):
                            param._value = data[group_name][param_name]
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
                        # reconstruct serialized param object
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
                    param = param._value
                    serialized_params[name] = param
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
