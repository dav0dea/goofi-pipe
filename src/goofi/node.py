import importlib.resources as pkg_resources
import time
import traceback
from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
from multiprocessing import Process
from os.path import dirname, join
from pathlib import PosixPath
from threading import Event, Thread
from typing import Any, Callable, Dict, Optional, Tuple, Union

from goofi import assets
from goofi.connection import Connection
from goofi.data import Data, DataType
from goofi.message import Message, MessageType
from goofi.node_helpers import InputSlot, NodeProcessRegistry, NodeRef, OutputSlot
from goofi.params import InvalidParamError, NodeParams


class MultiprocessingForbiddenError(Exception):
    pass


def require_init(func: Callable) -> Callable:
    """
    Decorator that checks if `super().__init__()` has been called in the `__init__()` method of the class
    that the decorated method belongs to. This is used to make sure that the base class is initialized before
    accessing any of its attributes.

    ### Parameters
    `func` : Callable
        The method to decorate.

    ### Returns
    `Callable`
        The decorated method.
    """

    def wrapper(self, *args, **kwargs):
        # check if the base class is initialized
        if not hasattr(self, "_alive"):
            raise RuntimeError("Make sure to call super().__init__() in your node's __init__ method.")
        return func(self, *args, **kwargs)

    return wrapper


class NodeEnv(Enum):
    """
    Enumeration of the possible node environments.
    """

    MULTIPROCESSING = 1  # the node owns its own process
    LOCAL = 2  # the node is running in the manager's process
    STANDALONE = 3  # running in the main process, but without a manager


class Node(ABC):
    """
    The base class for all nodes. A node is a processing unit that can receive data from other nodes, process the
    data, and send the processed data to other nodes. A node can have any number of input and output slots. Each
    input slot can receive data from one output slot, and each output slot can send data to any number of input
    slots.

    ### Parameters
    `connection` : Connection
        The input connection to the node. This is used to receive messages from the manager, or other nodes.
    `input_slots` : Dict[str, InputSlot]
        A dict containing the input slots of the node. The keys are the names of the input slots, and the values
        are the input slots themselves.
    `output_slots` : Dict[str, OutputSlot]
        A dict containing the output slots of the node. The keys are the names of the output slots, and the values
        are the output slots themselves.
    `params` : NodeParams
        An instance of the NodeParams class containing the parameters of the node.
    `environment` : NodeEnv
        The environment in which the node is running.
    """

    NO_MULTIPROCESSING = False
    MESSAGE_TIMEOUT = 500  # ms

    def __init__(
        self,
        connection: Optional[Connection],
        input_slots: Dict[str, InputSlot],
        output_slots: Dict[str, OutputSlot],
        params: NodeParams,
        environment: NodeEnv,
    ) -> None:
        # initialize the base class
        self._alive = True
        self._node_ready = False

        # store the arguments and validate them
        self.connection = connection
        self._input_slots = input_slots
        self._output_slots = output_slots
        self._params = params

        self._validate_attrs(environment)

        # NOTE: we avoid creating a threading.Event() in standalone mode to ensure nodes can be pickled
        if environment != NodeEnv.STANDALONE:
            # initialize node flags
            self.process_flag = Event()
            if self.params.common.autotrigger.value:
                self.process_flag.set()

        # set up dict of possibly timed out output connections
        self.pending_connections = {}

        if environment != NodeEnv.STANDALONE:
            # initialize data processing thread
            self.processing_thread = Thread(
                target=self._processing_loop,
                name=f"{self.__class__.__name__}-proc_loop",
                daemon=True,
            )
            self.processing_thread.start()

        if environment == NodeEnv.MULTIPROCESSING:
            # we are in a node-specific process, run the messaging loop in the current thread
            # NOTE: if we don't block the current thread, the node's process will die
            self._messaging_loop()
        elif environment == NodeEnv.LOCAL:
            # we are in the main process, create a new thread to not block it
            self.messaging_thread = Thread(
                target=self._messaging_loop,
                name=f"{self.__class__.__name__}-msg_loop",
                daemon=True,
            )
            self.messaging_thread.start()
        elif environment == NodeEnv.STANDALONE:
            # the node does not have a messaging, or processing loop when running in standalone mode
            pass
        else:
            raise ValueError(f"Invalid environment: {environment}")

    @require_init
    def _validate_attrs(self, environment: NodeEnv) -> None:
        """
        Check that all attributes are present and of the correct type.
        """
        # check connection type
        if environment == NodeEnv.STANDALONE:
            if self.connection is not None:
                raise ValueError("Running in standalone mode, connection should be None.")
        else:
            if not isinstance(self.connection, Connection):
                raise TypeError(f"Expected Connection, got {type(self.connection)}")

        # check input slots type
        for name, slot in self._input_slots.items():
            if not isinstance(name, str) or len(name) == 0:
                raise ValueError(f"Expected input slot name '{name}' to be a non-empty string.")
            if not isinstance(slot, InputSlot):
                raise TypeError(f"Expected InputSlot for input slot '{name}', got {type(slot)}")

        # check output slots type
        for name, slot in self._output_slots.items():
            if not isinstance(name, str) or len(name) == 0:
                raise ValueError(f"Expected output slot name '{name}' to be a non-empty string.")
            if not isinstance(slot, OutputSlot):
                raise TypeError(f"Expected OutputSlot for output slot '{name}', got {type(slot)}")

        # check params type
        if not isinstance(self._params, NodeParams):
            raise TypeError(f"Expected NodeParams, got {type(self._params)}")

    def _setup(self):
        """This method calls the node's setup method and handles any exceptions that may occur."""
        while not self._node_ready:
            try:
                self.setup()
                self._node_ready = True
                # clear any errors that may have occurred during setup
                self.connection.try_send(Message(MessageType.PROCESSING_ERROR, {"error": None}))
            except Exception:
                error_message = traceback.format_exc()
                self.connection.try_send(Message(MessageType.PROCESSING_ERROR, {"error": error_message}))
                time.sleep(0.1)

    def _serialize(self):
        """Serialize the node's type, output connections, and parameters, and send the serialized data to the manager."""
        # serialize output connections and the node's parameters
        out_conns = {name: slot.connections for name, slot in self.output_slots.items()}
        params = self.params.serialize()
        # return the serialized data
        self.connection.try_send(
            Message(
                MessageType.SERIALIZE_RESPONSE,
                {"_type": type(self).__name__, "category": self.category(), "out_conns": out_conns, "params": params},
            )
        )

    def _messaging_loop(self):
        """
        This method runs in a separate thread and handles incoming messages from the manager, or other nodes.
        """
        # run the node's setup method
        if self._setup.__self__.__class__._setup is not Node._setup:
            # the node implements _setup, which is not allowed
            error = (
                f"The {self.__class__.__name__} node implements the _setup() method, which is reserved for "
                "internal use. Use setup() instead."
            )
            self.connection.try_send(Message(MessageType.PROCESSING_ERROR, {"error": error}))
            raise RuntimeError(error)

        Thread(target=self._setup, name=f"{self.__class__.__name__}-setup", daemon=True).start()

        # run the messaging loop
        while self.alive:
            # receive the message
            try:
                msg = self.connection.recv()
            except ConnectionError:
                # the connection was closed, consider the node dead
                self._alive = False
                self.connection.close()
                continue
            except Exception:
                error_message = traceback.format_exc()
                self.connection.try_send(Message(MessageType.PROCESSING_ERROR, {"error": error_message}))
                continue

            # potentially restart the processing thread
            # TODO: this might be starting too many threads and not cleaning up properly
            if not self.processing_thread.is_alive():
                self.processing_thread = Thread(
                    target=self._processing_loop, name=f"{self.__class__.__name__}-proc_loop", daemon=True
                )
                self.processing_thread.start()

            if not isinstance(msg, Message):
                raise TypeError(f"Expected Message, got {type(msg)}")

            if msg.type == MessageType.PING:
                # respond to a ping message by sending a pong message
                self.connection.try_send(Message(MessageType.PONG, {}))
            elif msg.type == MessageType.TERMINATE:
                # terminate the node
                self._alive = False
                # clear data in connected downstream nodes
                for slot in self.output_slots.values():
                    for slot_name, conn, _ in slot.connections:
                        conn.try_send(Message(MessageType.CLEAR_DATA, {"slot_name": slot_name}))
            elif msg.type == MessageType.ADD_OUTPUT_PIPE:
                # add a connection to the output slot
                slot = self.output_slots[msg.content["slot_name_out"]]
                conn = msg.content["node_connection"]
                self_conn = False

                if conn is None:
                    # if no connection is specified, connect to the node's own node reference
                    conn = self.connection
                    self_conn = True

                slot.connections.append((msg.content["slot_name_in"], conn, self_conn))

                if not self_conn:
                    # notify the manager that the connection was added
                    self._serialize()
            elif msg.type == MessageType.REMOVE_OUTPUT_PIPE:
                # clear the data in the input slot
                msg.content["node_connection"].try_send(
                    Message(MessageType.CLEAR_DATA, {"slot_name": msg.content["slot_name_in"]})
                )
                # remove the connection
                slot = self.output_slots[msg.content["slot_name_out"]]
                try:
                    slot.connections.remove((msg.content["slot_name_in"], msg.content["node_connection"], False))
                except ValueError:
                    # connection doesn't exist
                    self.connection.try_send(
                        Message(
                            MessageType.PROCESSING_ERROR,
                            {
                                "error": f"Request to remove non-existent connection from "
                                f"{msg.content['slot_name_out']} to {msg.content['slot_name_in']}."
                            },
                        )
                    )

                # notify the manager of the updated connections
                self._serialize()
            elif msg.type == MessageType.DATA:
                # received data from another node
                if msg.content["slot_name"] not in self.input_slots:
                    raise ValueError(
                        f"Received DATA message but input slot '{msg.content['slot_name']}' doesn't exist."
                    )
                slot = self.input_slots[msg.content["slot_name"]]
                slot.data = msg.content["data"]
                if slot.trigger_process:
                    self.process_flag.set()
            elif msg.type == MessageType.CLEAR_DATA:
                # clear the data in the input slot (usually triggered by a REMOVE_OUTPUT_PIPE message)
                slot = self.input_slots[msg.content["slot_name"]]
                slot.clear()
            elif msg.type == MessageType.PARAMETER_UPDATE:
                # update a parameter
                group = msg.content["group"]
                param_name = msg.content["param_name"]
                param_value = msg.content["param_value"]
                if group not in self.params:
                    raise ValueError(f"Parameter group '{group}' doesn't exist.")
                if param_name not in self.params[group]:
                    raise ValueError(f"Parameter '{param_name}' doesn't exist in group '{group}'.")
                self.params[group][param_name].value = param_value

                # notify the manager that the parameter was updated
                self._serialize()

                # call the callback if it exists
                if hasattr(self, f"{group}_{param_name}_changed"):
                    try:
                        getattr(self, f"{group}_{param_name}_changed")(param_value)
                    except Exception as e:
                        # parameter callback raised an exception, send out an error message
                        self.connection.try_send(
                            Message(
                                MessageType.PROCESSING_ERROR,
                                {"error": f"Parameter callback for {group}.{param_name} failed: {e}"},
                            )
                        )

            elif msg.type == MessageType.SERIALIZE_REQUEST:
                self._serialize()
            else:
                # TODO: handle the incoming message
                raise NotImplementedError(f"Message type {msg.type} not implemented.")

        # run the node's terminate method
        try:
            self.terminate()
        except Exception as e:
            self.connection.try_send(Message(MessageType.PROCESSING_ERROR, {"error": str(e)}))

        # close input connection
        self.connection.close()

    def _processing_loop(self):
        """
        This method runs in a separate thread and handles the processing of input data and sending of
        output data to other nodes.
        """
        # wait until the node's setup is complete
        while not self._node_ready:
            time.sleep(0.1)

        last_update = 0
        while self.alive:
            # wait for a trigger
            self.process_flag.wait()
            # clear the trigger if autotrigger is False
            if not self.params.common.autotrigger.value:
                self.process_flag.clear()

            # limit the update rate
            if self.params.common.max_frequency.value > 0:
                max_freq = self.params.common.max_frequency.value
                if self.params.common.frequency_mode.value == "updates-per-second":
                    # convert frequency to seconds per update
                    max_freq = 1 / max_freq
                sleep_time = max_freq - (time.time() - last_update)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                last_update = time.time()

            # gather input data
            input_data = {name: slot.data for name, slot in self.input_slots.items()}

            try:
                # process data
                output_data = self.process(**input_data)
            except Exception:
                error_message = traceback.format_exc()
                self.connection.try_send(Message(MessageType.PROCESSING_ERROR, {"error": error_message}))
                continue

            if not self.alive:
                # the node was terminated during processing
                break

            # if process returns None, skip sending output data
            if output_data is None:
                # TODO: make sure this is the correct behavior
                continue

            # check that the process method returned a dict
            if not isinstance(output_data, dict):
                raise TypeError(
                    f"The process method should return a dictionary with one entry per output slot. "
                    f"Got {type(output_data)}."
                )

            # check that the output data contains the correct fields
            if missing := set(self.output_slots.keys()) - set(output_data.keys()):
                self.connection.try_send(
                    Message(
                        MessageType.PROCESSING_ERROR,
                        {
                            "error": f"Missing output fields: {missing}. "
                            f"Make sure that the process method returns a dict with one entry per output slot."
                        },
                    )
                )

            # check that the output data doesn't contain extra fields
            if extra_fields := list(set(output_data.keys()) - set(self.output_slots.keys())):
                self.connection.try_send(
                    Message(
                        MessageType.PROCESSING_ERROR,
                        {
                            "error": f"Extra output fields: {extra_fields}. "
                            f"The process method should only return those fields that were specified in the output slots."
                        },
                    )
                )

            # send output data
            for name in self.output_slots.keys():
                data = output_data[name]
                if data is None:
                    # skip sending None data
                    continue

                try:
                    data = Data(self.output_slots[name].dtype, data[0], data[1])
                except Exception:
                    error_message = traceback.format_exc()
                    self.connection.try_send(Message(MessageType.PROCESSING_ERROR, {"error": error_message}))
                    continue

                # send the data to all connected nodes
                for target_slot, conn, self_conn in self.output_slots[name].connections:
                    try:
                        msg = Message(MessageType.DATA, {"data": data, "slot_name": target_slot})
                    except Exception:
                        error_message = traceback.format_exc()
                        self.connection.try_send(Message(MessageType.PROCESSING_ERROR, {"error": error_message}))
                        continue

                    if conn._id in self.pending_connections:
                        # filter out dead threads
                        self.pending_connections[conn._id] = [
                            (thread, timestamp)
                            for thread, timestamp in self.pending_connections[conn._id]
                            if thread.is_alive()
                        ]
                        # check if the connection has timed out
                        timeout_occurred = False
                        for _, creation in self.pending_connections[conn._id]:
                            if time.time() - creation > self.MESSAGE_TIMEOUT / 1000:
                                # the connection has timed out, remove it
                                # TODO: figure out what's going on, if we remove a timed out connection the GUI will sometimes lose contact with the node
                                # self.output_slots[name].connections.remove((target_slot, conn, self_conn))
                                timeout_occurred = True
                                continue

                        if timeout_occurred:
                            # skip sending the message if the connection has timed out
                            continue
                    else:
                        self.pending_connections[conn._id] = []

                    # send the message (in a separate thread because connections may time out and block)
                    t = Thread(
                        target=conn.send, name=f"{self.__class__.__name__}-send-{conn._id}", args=(msg,), daemon=True
                    )
                    t.start()
                    self.pending_connections[conn._id].append((t, time.time()))

    @staticmethod
    def _configure(cls) -> Tuple[Dict[str, InputSlot], Dict[str, OutputSlot], NodeParams]:
        """Retrieves the node's configuration of input slots, output slots, and parameters."""
        in_slots = cls.config_input_slots()
        out_slots = cls.config_output_slots()
        params = cls.config_params()

        return (
            {name: slot if isinstance(slot, InputSlot) else InputSlot(slot) for name, slot in in_slots.items()},
            {name: slot if isinstance(slot, OutputSlot) else OutputSlot(slot) for name, slot in out_slots.items()},
            NodeParams(params),
        )

    def common_autotrigger_changed(self, value):
        """If the new value of the parameter common.autotrigger is True, trigger the processing loop."""
        if value:
            self.process_flag.set()

    @require_init
    def clear_error(self):
        """Clear the error message."""
        self.connection.try_send(Message(MessageType.PROCESSING_ERROR, {"error": None}))

    @classmethod
    def create(cls, initial_params: Optional[Dict[str, Dict[str, Any]]] = None, retries: int = 3) -> NodeRef:
        """
        Create a new node instance in a separate process and return a reference to the node.

        ### Parameters
        `initial_params` : Optional[Dict[str, Dict[str, Any]]]
            A dictionary of parameter groups, where each group is a dictionary of parameter names and values.
            Defaults to None.

        ### Returns
        `NodeRef`
            A reference to the node.
        """
        if cls.NO_MULTIPROCESSING:
            raise MultiprocessingForbiddenError("Multiprocessing is forbidden for this node.")

        if initial_params is not None and "process_group" in initial_params["common"]:
            pg_name = initial_params["common"]["process_group"]
            if len(pg_name) > 0:
                pg = NodeProcessRegistry().get(pg_name)
                conn1, conn2 = Connection.create()
                # spawn the node in the given process group
                pg.conn.send((cls, (conn1, conn2), initial_params))
                ref = pg.conn.recv()
                ref.initialize()
                return ref

        # generate arguments for the node
        in_slots, out_slots, params = cls._configure(cls)
        # integrate initial parameters if they are provided
        if initial_params is not None:
            try:
                params.update(initial_params)
            except InvalidParamError:
                print(
                    "\n====================== ERROR ======================\n"
                    f"Setting parameters failed for {cls.__name__}. This is likely due to updates"
                    " in the node's code. Make sure to reconfigure the node."
                    "\n===================================================\n"
                )

        tries = 0
        while True:
            try:
                conn1, conn2 = Connection.create()

                # instantiate the node in a separate process
                proc = Process(
                    target=cls,
                    name=cls.__name__,
                    args=(conn2, in_slots, out_slots, params, NodeEnv.MULTIPROCESSING),
                    daemon=True,
                )
                proc.start()
                break
            except Exception as e:
                tries += 1
                if tries >= retries:
                    raise e
                time.sleep(0.1)

        # create the node reference
        ref = NodeRef(
            conn1,
            {name: slot.dtype for name, slot in in_slots.items()},
            {name: slot.dtype for name, slot in out_slots.items()},
            params,
            cls,
            process=proc,
        )
        return ref

    @classmethod
    def create_local(
        cls,
        initial_params: Optional[Dict[str, Dict[str, Any]]] = None,
        conns: Optional[Tuple[Connection, Connection]] = None,
        init_ref: bool = True,
    ) -> Tuple[NodeRef, "Node"]:
        """
        Create a new node instance in the current process and return a reference to the node,
        as well as the node itself.

        ### Parameters
        `initial_params` : Optional[Dict[str, Dict[str, Any]]]
            A dictionary of parameter groups, where each group is a dictionary of parameter names and values.
            Defaults to None.

        ### Returns
        `Tuple[NodeRef, Node]`
            A tuple containing the node reference and the node itself.
        """
        # generate arguments for the node
        in_slots, out_slots, params = cls._configure(cls)

        # integrate initial parameters if they are provided
        if initial_params is not None:
            try:
                params.update(initial_params)
            except InvalidParamError:
                print(
                    "\n====================== ERROR ======================\n"
                    f"Setting parameters failed for {cls.__name__}. This is likely due to updates"
                    " in the node's code. Make sure to reconfigure the node."
                    "\n===================================================\n"
                )

        conn1, conn2 = Connection.create() if conns is None else conns
        # instantiate the node in the current process
        node = cls(conn2, in_slots, out_slots, params, NodeEnv.LOCAL)
        # create the node reference
        ref = NodeRef(
            conn1,
            {name: slot.dtype for name, slot in in_slots.items()},
            {name: slot.dtype for name, slot in out_slots.items()},
            deepcopy(params),
            cls,
            create_initialized=init_ref,
        )
        return ref, node

    @classmethod
    def create_standalone(cls) -> "Node":
        """
        Create a new node instance in the current process without a connection to the manager.

        ### Returns
        `Node`
            The node instance.
        """
        # generate arguments for the node
        in_slots, out_slots, params = cls._configure(cls)
        # instantiate the node in the current process
        return cls(None, in_slots, out_slots, params, NodeEnv.STANDALONE)

    @classmethod
    def category(cls) -> str:
        """
        Returns the category of the node, i.e. the name of the node's module.

        ### Returns
        `str`
            The category of the node.
        """
        return cls.__module__.split(".")[-2]

    @classmethod
    def docstring(cls) -> str:
        """
        Returns the cleaned docstring of the class (removes leading and trailing whitespace per line).

        ### Returns
        `str`
            The cleaned docstring of the class, or an empty string if no docstring is present.
        """

        if hasattr(cls, "__doc__") and cls.__doc__:
            return "\n".join([line.strip() for line in cls.__doc__.split("\n")])
        return ""

    @property
    def assets_path(self) -> PosixPath:
        """
        Returns the absolute path to the assets folder of goofi-pipe.

        ### Returns
        `PosixPath`
            The path to the assets folder of the node.
        """
        return pkg_resources.files(assets)

    @property
    def data_path(self) -> str:
        """
        Returns the absolute path to the data folder of goofi-pipe.

        ### Returns
        `str`
            The path to the data folder of the node.
        """
        return join(dirname(dirname(dirname(__file__))), "data")

    @property
    @require_init
    def alive(self) -> bool:
        return self._alive

    @property
    @require_init
    def params(self) -> NodeParams:
        return self._params

    @property
    @require_init
    def input_slots(self) -> Dict[str, InputSlot]:
        return self._input_slots

    @property
    @require_init
    def output_slots(self) -> Dict[str, OutputSlot]:
        return self._output_slots

    @staticmethod
    def config_input_slots() -> Dict[str, Union[InputSlot, DataType]]:
        """
        This method is called when the node is instantiated. It should return a dict containing the input slots
        of the node. The keys are the names of the input slots, and the values are either InputSlot instances or
        DataType instances.

        ### Returns
        `Dict[str, Union[InputSlot, DataType]]`
            A dict containing the input slots of the node.
        """
        return {}

    @staticmethod
    def config_output_slots() -> Dict[str, Union[OutputSlot, DataType]]:
        """
        This method is called when the node is instantiated. It should return a dict containing the output slots
        of the node. The keys are the names of the output slots, and the values are either OutputSlot instances or
        DataType instances.

        ### Returns
        `Dict[str, Union[OutputSlot, DataType]]`
            A dict containing the output slots of the node.
        """
        return {}

    @staticmethod
    def config_params() -> Dict[str, Dict[str, Any]]:
        """
        This method is called when the node is instantiated. It should return a dict containing the parameters
        of the node. The keys are the names of the parameter groups, and the values are dicts containing the
        parameters of each group.

        ### Returns
        `Dict[str, Dict[str, Any]]`
            A dict containing the parameters of the node. The values may be any type that is supported by the
            `Param` class.
        """
        return {}

    @require_init
    def setup(self) -> None:
        """
        This method is called after the node is instantiated. It can be used to set up the node.
        """
        pass

    @abstractmethod
    def process(self, **kwargs) -> Dict[str, Tuple[Any, Dict[str, Any]]]:
        """
        This method is called when the node is triggered. It should process the input data and return a dict
        containing the output data.

        ### Parameters
        `**kwargs` : Any
            The input data.

        ### Returns
        `Dict[str, Tuple[Any, Dict[str, Any]]]`
            A dict containing the output data. The keys are the names of the output slots, and the values are
            tuples containing the data and metadata of the output data.
        """
        pass

    def terminate(self) -> None:
        """
        This method is called when the node is terminated. It can be used to clean up any resources used by the
        node.
        """
        pass
