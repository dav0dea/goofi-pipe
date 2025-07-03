import pickle
import platform
import queue
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from multiprocessing import Pipe
from multiprocessing.connection import _ConnectionBase
from multiprocessing.managers import BaseManager
from typing import Dict, Tuple, Type

import zmq


class Connection(ABC):
    _CONNECTION_IDS = None
    _BACKEND = None

    def __init__(self) -> None:
        if Connection._CONNECTION_IDS is None:
            raise RuntimeError("Connection._CONNECTION_IDS is None. Call Connection.set_backend() first.")

        # register a unique id for the connection
        self._id = 0
        while self._id in Connection._CONNECTION_IDS:
            self._id += 1
        Connection._CONNECTION_IDS.append(self._id)

    @staticmethod
    def get_backends() -> Dict[str, Type["Connection"]]:
        """
        List all available connection backends.
        """
        return {
            "zmq-tcp": TCPZeroMQConnection,
            "zmq-ipc": IPCZeroMQConnection,
            "mp": MultiprocessingConnection,
        }

    @staticmethod
    def set_backend(backend: str, mp_manager: BaseManager) -> None:
        """
        Set the backend to use for creating connections and initialize a shared set of connection ids.

        ### Parameters
        `backend` : str
            The backend to use. Choose from "auto", "zmq-tcp", "zmq-ipc" or "mp".
        `mp_manager` : multiprocessing.Manager
            The multiprocessing manager to use for creating shared objects.
        """
        # initialize a shared set of connection ids
        assert Connection._CONNECTION_IDS is None, "Connection._CONNECTION_IDS is not None."
        Connection._CONNECTION_IDS = mp_manager.list()

        if backend == "auto":
            if platform.system() == "Windows":
                backend = "mp"
            else:
                backend = "zmq-tcp"

        # set the backend if discovering connection IDs was successful (prevents overwriting the backend)
        assert (
            backend in Connection.get_backends().keys()
        ), f"Invalid backend: {backend}. Choose from {list(Connection.get_backends().keys())}"
        Connection._BACKEND = backend

    @staticmethod
    def create() -> Tuple["Connection", "Connection"]:
        """
        Create two instances of the connection class and return them as a tuple. Both instances should be
        connected to each other using the underlying connection mechanism of the deriving class.
        """
        assert Connection._BACKEND is not None, "No backend set. Call Connection.set_backend() first."
        return Connection.get_backends()[Connection._BACKEND]._create()

    @staticmethod
    @abstractmethod
    def _create() -> Tuple["Connection", "Connection"]:
        """
        Create two instances of the connection class and return them as a tuple. Both instances should be
        connected to each other using the underlying connection mechanism of the deriving class.
        """
        raise TypeError("This is an abstract method.")

    @abstractmethod
    def send(self, obj: object) -> None:
        """
        Send an object through the connection (requires the object to be picklable).

        This method should raise a ConnectionError if either it, or the receiving end of the connection is closed.

        ### Parameters
        `obj` : object
            The object to send through the connection.
        """
        pass

    @abstractmethod
    def recv(self) -> object:
        """
        Receive an object through the connection.

        This method should raise a ConnectionError if either it, or the sending end of the connection is closed.

        ### Returns
        `object`
            The object received through the connection.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Close the connection and free any resources associated with it.
        """
        pass

    def try_send(self, obj: object) -> bool:
        """
        Try to send an object through the connection (i.e. catch any ConnectionError that might be raised).

        ### Parameters
        `obj` : object
            The object to send through the connection.

        ### Returns
        `bool`
            True if the object was sent, False otherwise.
        """
        try:
            self.send(obj)
            return True
        except ConnectionError:
            return False

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Connection):
            return False
        return self._id == other._id


class MultiprocessingConnection(Connection):
    def __init__(self, conn: _ConnectionBase) -> None:
        super().__init__()
        self.conn = conn

    @staticmethod
    def _create() -> Tuple[Connection, Connection]:
        conn1, conn2 = Pipe()
        return MultiprocessingConnection(conn1), MultiprocessingConnection(conn2)

    def send(self, obj: object) -> None:
        try:
            self.conn.send(obj)
        except (OSError, BrokenPipeError):
            raise ConnectionError("Connection closed")

    def recv(self) -> object:
        try:
            return self.conn.recv()
        except (OSError, EOFError, TypeError):
            raise ConnectionError("Connection closed")

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def __del__(self) -> None:
        """Destructor to close the connection and free any resources associated with it."""
        try:
            self.close()
            self.conn._handle = None
        except Exception:
            pass


class ZeroMQConnection(Connection, ABC):
    def __init__(self, push_endpoint: str, pull_endpoint: str = None) -> None:
        super().__init__()
        self.context = zmq.Context()
        self.push_endpoint = push_endpoint
        self.pull_endpoint = pull_endpoint
        self.push_queue = queue.Queue()
        self.pull_queue = queue.Queue()

        self.alive = True
        self.push_thread = threading.Thread(target=self.run_push, daemon=True)
        self.pull_thread = threading.Thread(target=self.run_pull, daemon=True)

    @classmethod
    def _create(cls) -> Tuple[Connection, Connection]:
        if cls.protocol() == "tcp":
            endpoint1 = TCPZeroMQConnection.unique_tcp_endpoint()
            endpoint2 = TCPZeroMQConnection.unique_tcp_endpoint()
        elif cls.protocol() == "ipc":
            endpoint1 = IPCZeroMQConnection.unique_ipc_endpoint()
            endpoint2 = IPCZeroMQConnection.unique_ipc_endpoint()
        else:
            raise ValueError(f"Invalid protocol: {cls.protocol}")
        return cls(endpoint1, endpoint2), cls(endpoint2, endpoint1)

    def run_push(self):
        push_socket = None
        if self.push_endpoint is not None:
            push_socket = self.context.socket(zmq.PUSH)
            push_socket.connect(self.push_endpoint)

        while self.alive:
            try:
                obj = self.push_queue.get(block=False)
                push_socket.send(pickle.dumps(obj))
            except queue.Empty:
                pass
            time.sleep(0.01)

        if push_socket:
            push_socket.close()

    def run_pull(self):
        if self.pull_endpoint is None:
            # no pull endpoint, stop the thread
            return

        pull_socket = self.context.socket(zmq.PULL)
        pull_socket.bind(self.pull_endpoint)

        while self.alive:
            try:
                obj = pickle.loads(pull_socket.recv(zmq.NOBLOCK))
                self.pull_queue.put(obj)
            except zmq.Again:
                pass
            time.sleep(0.01)

        if pull_socket:
            pull_socket.close()

    def send(self, obj: object) -> None:
        if not self.alive:
            raise ConnectionError("Connection closed")
        if not self.push_thread.is_alive():
            try:
                self.push_thread.start()
            except RuntimeError:
                pass

        self.push_queue.put(obj)

    def recv(self) -> object:
        if not self.alive:
            raise ConnectionError("Connection closed")
        if not self.pull_thread.is_alive():
            try:
                self.pull_thread.start()
            except RuntimeError:
                pass

        return self.pull_queue.get()

    def close(self) -> None:
        self.alive = False
        if self.push_thread.is_alive():
            self.push_thread.join()
        if self.pull_thread.is_alive():
            self.pull_thread.join()

    def __reduce__(self):
        return (self.__class__, (self.push_endpoint, None), self.__getstate__())

    def __getstate__(self):
        return {"_id": self._id}

    def __setstate__(self, state):
        self._id = state["_id"]

    def __del__(self) -> None:
        """Destructor to close the connection and free any resources associated with it."""
        self.close()

    @staticmethod
    @abstractmethod
    def protocol() -> str:
        """Return the protocol used by the connection"""
        pass

    @staticmethod
    def unique_ipc_endpoint() -> str:
        """Generate a unique temporary file path. The file is deleted immediately after creation to avoid clashes"""
        temp_file = tempfile.NamedTemporaryFile(delete=True)
        endpoint = f"ipc://{temp_file.name}"
        # delete the file
        temp_file.close()
        return endpoint

    @staticmethod
    def unique_tcp_endpoint() -> str:
        """Generate a unique TCP endpoint. The endpoint is guaranteed to be unique for the current machine"""
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://127.0.0.1:0")
        port = socket.getsockopt(zmq.LAST_ENDPOINT).decode("ascii").split(":")[-1].rstrip("]")
        socket.close()
        context.term()
        return f"tcp://127.0.0.1:{port}"


class TCPZeroMQConnection(ZeroMQConnection):
    @staticmethod
    def protocol() -> str:
        return "tcp"


class IPCZeroMQConnection(ZeroMQConnection):
    @staticmethod
    def protocol() -> str:
        return "ipc"
