import functools
import importlib
import inspect
import os
import pkgutil
import time
import traceback
from dataclasses import dataclass, field
from multiprocessing import Pipe, Process
from multiprocessing.connection import _ConnectionBase
from threading import Thread
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from goofi import nodes as goofi_nodes
from goofi.connection import Connection
from goofi.data import Data, DataType
from goofi.message import Message, MessageType
from goofi.params import NodeParams


class NodeLoadingError(Exception):
    """Custom exception for node loading errors."""


@functools.lru_cache(maxsize=1)
def list_nodes(verbose: bool = False) -> List[Type]:
    """
    Gather a list of all available nodes in the goofi.nodes module.

    ### Parameters
    `verbose` : bool
        If True, print the names of all available nodes to the console.

    ### Returns
    List[Type[None]]
        A list containing the classes of all available nodes.
    """
    first_module = True

    def _list_nodes_recursive(nodes=None, parent_module=goofi_nodes):
        from goofi.node import Node

        nonlocal first_module

        if nodes is None:
            # first call, initialize the list
            nodes = []

        # iterate over all modules in the parent module
        for info in pkgutil.walk_packages(parent_module.__path__):
            if "goofi" + os.sep + "nodes" not in info.module_finder.path:
                # skip modules that are not in the goofi.nodes package
                continue
            if info.name.startswith("_"):
                # skip modules that start with an underscore
                continue

            # import the module and measure the time it takes
            start = time.time()
            try:
                module = importlib.import_module(f"{parent_module.__name__}.{info.name}")
            except ModuleNotFoundError as e:
                print()
                raise NodeLoadingError(
                    f"Missing dependency for node '{info.name}' -> {e}; make sure to install the required dependencies."
                ) from e
            except Exception as e:
                print()
                raise NodeLoadingError(f"Failed to load node '{info.name}' -> {e}") from e

            module_annot = "" if time.time() - start < 0.2 else "!!!"

            if verbose:
                # print module name
                parts = module.__name__.split(".")
                if len(parts) == 3:
                    if first_module:
                        first_module = False
                    else:
                        print()
                    # module is a node category
                    print(f"- {parts[2]}:", end="")

            if info.ispkg:
                # recursively list nodes in submodules
                _list_nodes_recursive(nodes, module)
                continue

            # current module is a node, add it to the list
            members = inspect.getmembers(module, inspect.isclass)
            new_nodes = [cls for _, cls in members if issubclass(cls, Node) and cls is not Node]

            if len(new_nodes) != 1:
                raise ValueError(f"Expected exactly one node in module {module.__name__}, got {len(new_nodes)}")

            if verbose:
                print(f"  {module_annot}{new_nodes[0].__name__}{module_annot}", end="")

            nodes.extend(new_nodes)
        return nodes

    if verbose:
        print("Discovering goofi-pipe nodes...")
    res = _list_nodes_recursive()
    if verbose:
        print("\n")
    return res


@dataclass
class InputSlot:
    """
    Nodes have a number of input slots that can be connected to output slots of other nodes. Each input slot
    has a specific data type (`dtype`) and a `trigger_process` flag that indicates if the node's process
    function should be triggered when data is received on this input slot. The input slot also internally
    handles the data (`data`) received on the input slot.

    ### Parameters
    `dtype` : DataType
        The data type of the input slot.
    `trigger_process` : Optional[bool]
        Optional flag to indicate if the node's process function should be triggered when data is received on
        this input slot. Defaults to True.
    """

    dtype: DataType
    trigger_process: bool = True
    data: Optional[Data] = None

    def __post_init__(self):
        if isinstance(self.dtype, str):
            # convert dtype from string to DataType
            self.dtype = DataType[self.dtype]

    def clear(self):
        """Clears the data stored in the input slot."""
        self.data = None


@dataclass
class OutputSlot:
    """
    Nodes have a number of output slots that can be connected to input slots of other nodes. Each output slot
    has a specific data type (`dtype`) and internally handles a list of connections (`connections`).

    ### Parameters
    `dtype` : DataType
        The data type of the output slot.
    """

    dtype: DataType
    connections: List[Tuple[str, Connection, bool]] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.dtype, str):
            # convert dtype from string to DataType
            self.dtype = DataType[self.dtype]


@dataclass
class NodeRef:
    """
    A reference to a node, implementing the node's counterpart in the manager. The reference contains the
    connection object to the node, and the input and output slots of the node. NodeRef instances should
    be created along with a node.

    NodeRef instances will handle the following message types with default behavior, which can be overridden
    by registering a message handler function with the `set_message_handler()` method:

    - `PING`: Responds with a `PONG` message.
    - `TERMINATE`: Terminates the node by closing the connection to it.

    ### Parameters
    `connection` : Connection
        The connection object to the node.
    `input_slots` : Dict[str, DataType]
        A dictionary of input slots and their data types.
    `output_slots` : Dict[str, DataType]
        A dictionary of output slots and their data types.
    `params` : NodeParams
        The parameters of the node.
    `category` : str
        The category of the node.
    `process` : Optional[Process]
        If the node is running in a separate process, this should be the process object. Defaults to None.
    """

    connection: Connection
    input_slots: Dict[str, DataType]
    output_slots: Dict[str, DataType]
    params: NodeParams
    node_class: Type

    category: str = None
    process: Optional[Process] = None
    callbacks: Dict[MessageType, Callable] = field(default_factory=dict)
    serialization_pending: bool = False
    serialized_state: Optional[Dict[str, Any]] = None
    gui_kwargs: Dict[str, Any] = field(default_factory=dict)
    create_initialized: bool = True

    def __post_init__(self):
        if self.connection is None:
            raise ValueError("Expected Connection, got None.")

        self.__doc__ = self.node_class.docstring()
        self.category = self.node_class.category()

        if self.create_initialized:
            self.initialize()

        # register the node reference as an output pipe for each output slot
        for name in self.output_slots.keys():
            self.connection.send(
                Message(
                    MessageType.ADD_OUTPUT_PIPE,
                    {"slot_name_out": name, "slot_name_in": name, "node_connection": None},
                )
            )

    def initialize(self):
        self._alive = True
        self._messaging_thread = Thread(target=self._messaging_loop, daemon=True)
        self._messaging_thread.start()

        # request the initial serialized state
        self.serialize()

    def set_message_handler(self, msg_type: MessageType, callback: Optional[Callable]) -> None:
        """
        Registers a message handler function that is called when the node sends this type of message
        to the manager.

        ### Parameters
        `msg_type` : MessageType
            The type of message to register the handler for.
        `callback` : Optional[Callable]
            Callback function: `callback(node: NodeRef, msg: Message) -> None`
        """
        self.callbacks[msg_type] = callback

    def update_param(self, group, param_name, param_value):
        """
        Updates the value of a parameter in the local node reference, and sends a message to the node to
        update the parameter value.

        ### Parameters
        `group` : str
            The name of the parameter group.
        `param_name` : str
            The name of the parameter.
        `param_value` : Any
            The new value of the parameter.
        """
        if group not in self.params:
            raise ValueError(f"Parameter group '{group}' doesn't exist.")
        if param_name not in self.params[group]:
            raise ValueError(f"Parameter '{param_name}' doesn't exist in group '{group}'.")
        self.params[group][param_name].value = param_value
        self.connection.try_send(
            Message(
                MessageType.PARAMETER_UPDATE,
                {
                    "group": group,
                    "param_name": param_name,
                    "param_value": param_value,
                },
            )
        )

    def serialize(self) -> None:
        """
        Sets the serialization_pending flag and requests a new serialized state from the node.
        The serialized state can be accessed through the `serialized_state` attribute.

        Make sure to wait for the node's response as this is an asynchronous operation.
        """
        self.serialization_pending = True
        self.connection.send(Message(MessageType.SERIALIZE_REQUEST, {}))

    def terminate(self) -> None:
        """Terminates the node by closing the connection to it."""
        self.connection.try_send(Message(MessageType.TERMINATE, {}))
        self.connection.close()

    def _messaging_loop(self) -> None:
        """This method runs in a separate thread and handles incoming messages the node."""
        while self._alive:
            try:
                msg = self.connection.recv()
            except ConnectionError:
                # the connection was closed, consider the node dead
                self._alive = False
                self.connection.close()
                continue

            if not isinstance(msg, Message):
                raise TypeError(f"Expected Message, got {type(msg)}")

            # if the message type has a registered callback, call it and skip built-in message handling
            if msg.type in self.callbacks:
                try:
                    self.callbacks[msg.type](self, msg)
                except Exception:
                    # TODO: add proper logging
                    error_msg = traceback.format_exc()
                    print(f"Message callback for {msg.type} failed: {error_msg}")
                continue

            # built-in message handling
            if msg.type == MessageType.PING:
                # respond with PONG
                self.connection.send(Message(MessageType.PONG, {}))
            elif msg.type == MessageType.TERMINATE:
                # the node has terminated itself, consider it dead
                self._alive = False
                self.connection.close()
            elif msg.type == MessageType.SERIALIZE_RESPONSE:
                # store the serialized state
                self.serialized_state = msg.content
                self.serialization_pending = False


class NodeProcess:
    def __init__(self, name: str):
        self.name = name
        self.conn, recv = Pipe()
        self.proc = Process(target=self.messaging_loop, name=name, args=(recv,), daemon=True)
        self.proc.start()

    def messaging_loop(self, conn: _ConnectionBase):
        nodes = []
        while True:
            cls, conns, args = conn.recv()
            ref, node = cls.create_local(initial_params=args, conns=conns, init_ref=False)
            nodes.append(node)
            conn.send(ref)


class NodeProcessRegistry:
    _instance = None
    _processes: Dict[str, NodeProcess] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NodeProcessRegistry, cls).__new__(cls)
        return cls._instance

    def get(self, name: str) -> NodeProcess:
        if name not in self._processes:
            self._processes[name] = NodeProcess(name)
        return self._processes.get(name)

    def all(self) -> Dict[str, NodeProcess]:
        return dict(self._processes)
