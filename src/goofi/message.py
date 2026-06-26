from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict


class MessageType(Enum):
    """
    Message types carried on the control (manager → node) and status (node →
    manager) iceoryx2 pub/sub channels. The data plane is handled by a
    separate per-output-slot pub/sub channel — `DATA` is no longer a message
    type.

    Control plane (manager → node):

    - `SUBSCRIBE_INPUT`: instruct a node to start consuming from the named
      data service into one of its input slots.
        - `slot_name_in` (str): the input slot to bind.
        - `service_name` (str): the iceoryx2 service to subscribe to.
        - `in_process` (bool): true iff the upstream publisher lives in the
          same process group; selects thread vs. iceoryx2 transport.
    - `UNSUBSCRIBE_INPUT`: stop consuming on the named input slot.
        - `slot_name_in` (str)
    - `REGISTER_SUBSCRIBER`: increment subscriber count on a node's output
      slot so the node knows a consumer exists and the slot is worth
      publishing. Purely informational.
        - `slot_name_out` (str)
    - `UNREGISTER_SUBSCRIBER`: decrement subscriber count.
        - `slot_name_out` (str)
    - `PARAMETER_UPDATE`: update a parameter on the node.
        - `group` (str): the parameter group.
        - `param_name` (str): the parameter name.
        - `param_value` (Any): the new value.
    - `SET_EXPRESSION`: bind or unbind a Python expression on a parameter.
      When set, the node spins up an `ExpressionEngine` that re-evaluates
      whenever any referenced output slot emits new data. ``None`` clears.
        - `group` (str): the parameter group.
        - `param_name` (str): the parameter name.
        - `expression` (str | None): the expression source, or None to clear.
    - `CLEAR_DATA`: clear the cached value on an input slot.
        - `slot_name` (str)
    - `NODE_DIRECTORY`: the manager's current map of display name -> stable
      node_id. Pushed to every live node on each add/remove so a node's
      expression engine can resolve `nd('name')` references to the producing
      node's transport id (which the data services are keyed on, not the
      reusable display name).
        - `directory` (dict[str, str])
    - `REGISTER_VIEWER`: a browser viewer attached to a node's output slot
      (Option C node-side reduction). Increments `viewer_count` so the node
      produces + offers reduced frames on the slot's `.view` service even with
      no node consumer. Manager-driven (one per (node, slot), folded across
      browsers).
        - `slot_name_out` (str)
    - `UNREGISTER_VIEWER`: a browser viewer detached; decrements `viewer_count`
      (evicts the reducer state on the 1->0 transition).
        - `slot_name_out` (str)
    - `SET_VIEWSPEC`: set the folded per-axis ViewSpec the node reduces a slot's
      viewer frames to (last-received-wins).
        - `slot_name_out` (str)
        - `spec` (dict): `{axes:[{axis,max,method}], version}`.
    - `TERMINATE`: empty; instructs the node to shut down.

    Status plane (node → manager):

    - `STATE_UPDATE`: pushed by the node whenever its serialized state
      changes (dirty/clean tracking — replaces the old request/response
      `SERIALIZE_REQUEST`/`SERIALIZE_RESPONSE` dance).
        - `_type` (str): node class name.
        - `category` (str): node category.
        - `params` (dict): full param tree as a plain dict.
        - `output_subscribers` (dict[str, int]): subscriber count per output
          slot — manager uses this to verify wiring without round-tripping.
    - `PROCESSING_ERROR`: most recent error from `process()` (or `None` to
      clear a previously reported error).
        - `error` (str | None)
    - `SHUTDOWN`: node requests full goofi-pipe shutdown.
    """

    # control plane
    SUBSCRIBE_INPUT = 1
    UNSUBSCRIBE_INPUT = 2
    REGISTER_SUBSCRIBER = 3
    UNREGISTER_SUBSCRIBER = 4
    PARAMETER_UPDATE = 5
    CLEAR_DATA = 6
    TERMINATE = 7
    SET_EXPRESSION = 11
    NODE_DIRECTORY = 12
    REGISTER_VIEWER = 13
    UNREGISTER_VIEWER = 14
    SET_VIEWSPEC = 15

    # status plane
    STATE_UPDATE = 8
    PROCESSING_ERROR = 9
    SHUTDOWN = 10


@dataclass
class Message:
    """
    A message carried on the control or status plane. The `content` dict
    holds the per-type fields documented on `MessageType`.
    """

    type: MessageType
    content: Dict[str, Any]

    def require_fields(self, **fields: Any) -> None:
        for field, field_type in fields.items():
            if field not in self.content:
                raise ValueError(f"Message content must contain field {field}.")
            if not isinstance(self.content[field], field_type):
                raise ValueError(
                    f"Message content field {field} must be of type {field_type} "
                    f"but got {type(self.content[field])}."
                )

    def __post_init__(self):
        if self.type is None:
            raise ValueError("Expected message type, got None.")
        if not isinstance(self.content, dict):
            raise ValueError(f"Expected dict, got {type(self.content)}.")

        t = self.type
        if t == MessageType.SUBSCRIBE_INPUT:
            self.require_fields(slot_name_in=str, service_name=str, in_process=bool)
        elif t == MessageType.UNSUBSCRIBE_INPUT:
            self.require_fields(slot_name_in=str)
        elif t == MessageType.REGISTER_SUBSCRIBER:
            self.require_fields(slot_name_out=str)
        elif t == MessageType.UNREGISTER_SUBSCRIBER:
            self.require_fields(slot_name_out=str)
        elif t == MessageType.PARAMETER_UPDATE:
            self.require_fields(group=str, param_name=str)
            if "param_value" not in self.content:
                raise ValueError("PARAMETER_UPDATE content must contain field param_value.")
        elif t == MessageType.SET_EXPRESSION:
            self.require_fields(group=str, param_name=str)
            # expression is allowed to be None (clears the binding)
            if "expression" in self.content:
                expr = self.content["expression"]
                if expr is not None and not isinstance(expr, str):
                    raise ValueError("SET_EXPRESSION 'expression' must be str or None.")
        elif t == MessageType.CLEAR_DATA:
            self.require_fields(slot_name=str)
        elif t == MessageType.NODE_DIRECTORY:
            self.require_fields(directory=dict)
        elif t == MessageType.REGISTER_VIEWER:
            self.require_fields(slot_name_out=str)
        elif t == MessageType.UNREGISTER_VIEWER:
            self.require_fields(slot_name_out=str)
        elif t == MessageType.SET_VIEWSPEC:
            self.require_fields(slot_name_out=str, spec=dict)
        elif t == MessageType.STATE_UPDATE:
            self.require_fields(_type=str, category=str, params=dict, output_subscribers=dict)
        elif t == MessageType.PROCESSING_ERROR:
            self.require_fields(error=(str, type(None)))
