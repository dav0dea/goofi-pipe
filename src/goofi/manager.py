import importlib
import time
from copy import deepcopy
from multiprocessing import Manager as MPManager
from os import path
from pathlib import Path
from threading import Thread
from typing import Any, Dict, Optional

import numpy as np
import yaml

from goofi.connection import Connection
from goofi.gui.window import Window
from goofi.message import Message, MessageType
from goofi.node import MultiprocessingForbiddenError
from goofi.node_helpers import NodeRef, list_nodes


def mark_unsaved_changes(func):
    """
    Decorator that marks the manager as having unsaved changes after the function is called.
    """

    def wrapper(self, *args, **kwargs):
        res = func(self, *args, **kwargs)
        self.unsaved_changes = True
        return res

    return wrapper


class NodeContainer:
    """
    The node container keeps track of all nodes in the manager. It provides methods to add and remove nodes,
    and to access them by name.
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, NodeRef] = {}

    def add_node(self, name: str, node: NodeRef, force_name: bool = False) -> str:
        """
        Adds a node to the container with a unique name.

        ### Parameters
        `name` : str
            The name of the node.
        `node` : NodeRef
            The node to add.
        `force_name`: bool
            If True, raise an error if the name is already taken. Otherwise makes the name unique.
        """
        if not isinstance(name, str):
            raise ValueError(f"Expected string, got {type(name)}.")
        if not isinstance(node, NodeRef):
            raise ValueError(f"Expected NodeRef, got {type(node)}.")

        if force_name:
            # check if the name is already taken
            if name in self._nodes:
                raise KeyError(f"Node {name} already in container.")
            # register the node under the given name
            self._nodes[name] = node
            return name

        # generate a unique name for the node
        idx = 0
        while f"{name}{idx}" in self._nodes:
            idx += 1
        # register the node under the generated name
        self._nodes[f"{name}{idx}"] = node
        return f"{name}{idx}"

    def remove_node(self, name: str) -> None:
        """
        Terminates the node and removes it from the container.

        ### Parameters
        `name` : str
            The name of the node.
        """
        if name in self._nodes:
            self._nodes[name].terminate()
            del self._nodes[name]
            return True
        raise KeyError(f"Node {name} not in container")

    def __getitem__(self, name: str) -> NodeRef:
        return self._nodes[name]

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self):
        return iter(self._nodes.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._nodes


class Manager:
    """
    The manager keeps track of all nodes, and provides methods to add and remove nodes, and links between them.
    It also interfaces with the GUI to display the nodes and links, and to handle user interaction.

    ### Parameters
    `filepath` : Optional[str]
        The path to the file to load from. If `None`, does not load a file.
    `headless` : bool
        Whether to run in headless mode. If `True`, the GUI will not be started.
    `use_multiprocessing` : bool
        Whether to use multiprocessing for nodes that support it. If `False`, all nodes will be created in the
        same process as the manager.
    `duration` : float
        The duration to run the manager for. If `0`, runs indefinitely.
    `communication_backend` : str
        The communication backend to use for node communication. Default is "auto" (platform-dependent).
    """

    def __init__(
        self,
        filepath: Optional[str] = None,
        headless: bool = True,
        use_multiprocessing: bool = True,
        duration: float = 0,
        communication_backend: str = "auto",
    ) -> None:
        # create a multiprocessing manager
        self._mp_manager = MPManager()

        try:
            # set up communication backend
            Connection.set_backend(communication_backend, self._mp_manager)
        except AssertionError:
            print("Connection backend already set. Skipping.")

        if communication_backend == "auto":
            # make sure a backend was selected
            assert Connection._BACKEND is not None
        else:
            # make sure the connection backend is set correctly
            assert Connection._BACKEND == communication_backend

        # TODO: add proper logging
        print("Starting goofi-pipe...")
        # preload all nodes to avoid delays
        list_nodes(verbose=True)

        # TODO: add proper logging
        mp_state = "enabled" if use_multiprocessing else "disabled"
        print(f"Initializing goofi-pipe manager (multiprocessing {mp_state}).")
        print(f"Using communication backend: {Connection._BACKEND}.")

        self._headless = headless
        self._use_multiprocessing = use_multiprocessing
        self._running = True
        self.nodes = NodeContainer()

        # store attributes related to loading and saving
        self._save_path = None
        self._unsaved_changes = False

        if self.headless:
            # there is no GUI, so we can run everything in the main thread
            self.post_init(filepath, duration)
        else:
            # start the blocking non-daemon post-initialization thread to leave the main thread free for the GUI (limitation of MacOS)
            Thread(target=self.post_init, args=(filepath, duration), daemon=False).start()

            # initialize the GUI
            # NOTE: this is a blocking call, so it must be the last thing we do
            Window(self)

    def post_init(self, filepath: Optional[str] = None, duration: float = 0) -> None:
        """
        Wait until everything is initialized and run post-initialization tasks (e.g. loading a .gfi file) since the GUI
        potentially blocks the main thread.

        This function is called in a separate thread and will block until the manager is terminated. This is to avoid blocking
        the main thread. We leave the main thread to the GUI, which is necessary on MacOS.

        ### Parameters
        `filepath` : Optional[str]
            The path to the file to load from. If `None`, does not load a file.
        `duration` : float
            The duration to run the manager for. If `0`, runs indefinitely.
        """
        # wait for the GUI to initialize
        if not self.headless:
            win = None
            while win is None:
                # try to get the window instance from the main thread
                try:
                    win = Window()
                except RuntimeError:
                    # the window is not initialized yet, wait a bit
                    time.sleep(0.01)

            # make sure the GUI has finished initializing
            while not win._initialized:
                time.sleep(0.01)

        # load the manager state from a file
        if filepath is not None:
            self.load(filepath, load_on_init=True)

        if duration > 0:
            # run for a fixed duration
            time.sleep(duration)
            self.terminate()

        try:
            # run indefinitely
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.terminate()

    def load(self, filepath: str, load_on_init: bool = False) -> None:
        """
        Loads the state of the manager from a file.

        ### Parameters
        `filepath` : str
            The path to the file to load from.
        `load_on_init` : bool
            Whether the file is being loaded during initialization of the manager.
        """
        if len(self.nodes) > 0:
            # make sure the manager is empty
            raise RuntimeError("This goofi-pipe already contains nodes.")

        if not path.exists(filepath):
            if load_on_init:
                # missing file during initialization, terminate the application
                self.terminate()
            raise FileNotFoundError(f"File '{filepath}' does not exist.")

        # TODO: add proper logging
        print(f"Loading manager state from '{filepath}'...")

        # load the yaml file
        with open(filepath, "r") as f:
            manager_yaml = yaml.load(f, Loader=yaml.FullLoader)

        # create all nodes
        for name, node in manager_yaml["nodes"].items():
            # In rare cases node positions can be corrupted, which shows up as a position of (min int32, min int32).
            # This causes the dearpygui to segfault when trying to create a link to the node. Workaround this by
            # resetting the position to (0, 0) if it is corrupted.
            xpos, ypos = node["gui_kwargs"]["pos"]
            if xpos == np.iinfo(np.int32).min or ypos == np.iinfo(np.int32).min:
                print(f"WARNING: Node '{name}' has a corrupted position ({xpos}, {ypos}). Resetting to (0, 0).")
                node["gui_kwargs"]["pos"] = (0, 0)

            # add the node to the manager
            self.add_node(node["_type"], node["category"], name=name, params=node["params"], **node["gui_kwargs"])

        # add links
        for link in manager_yaml["links"]:
            self.add_link(link["node_out"], link["node_in"], link["slot_out"], link["slot_in"])

        # store the save path
        self.save_path = filepath
        self.unsaved_changes = False

        # TODO: add proper logging
        print("Finished loading manager state.")

    @mark_unsaved_changes
    def add_node(
        self,
        node_type: str,
        category: str,
        notify_gui: bool = True,
        name: Optional[str] = None,
        params: Optional[Dict[str, Dict[str, Any]]] = None,
        **gui_kwargs,
    ) -> str:
        """
        Adds a node to the container.

        ### Parameters
        `node_type` : str
            The name of the node type (the node's class name).
        `category` : str
            The category of the node.
        `notify_gui` : bool
            Whether to notify the gui to add the node.
        `name` : Optional[str]
            Raises an error if the name is already taken. If `None`, a unique name is generated.
        `params` : Optional[Dict[str, Dict[str, Any]]]
            The parameters of the node. If `None`, the default parameters are used.
        `gui_kwargs` : dict
            Additional keyword arguments to pass to the gui.

        ### Returns
        `name` : str
            The name of the node.
        """
        # TODO: add proper logging
        print(f"Adding node '{node_type}' from category '{category}'.")

        # import the node
        mod = importlib.import_module(f"goofi.nodes.{category}.{node_type.lower()}")
        node = getattr(mod, node_type)

        # instantiate the node
        ref = None
        if self._use_multiprocessing:
            # try to spawn the node in a separate process
            try:
                ref = node.create(initial_params=params)
            except MultiprocessingForbiddenError:
                # the node doesn't support multiprocessing, create it in the local process
                pass
        if ref is None:
            # spawn the node in the local process
            ref = node.create_local(initial_params=params)[0]

        # add the node to the container
        if name is None:
            # default name is the node type
            name = self.nodes.add_node(node_type.lower(), ref)
        else:
            # force the given name
            name = self.nodes.add_node(name, ref, force_name=True)

        # add the node to the gui
        if not self.headless and notify_gui:
            Window().add_node(name, ref, **gui_kwargs)
        return name

    @mark_unsaved_changes
    def remove_node(self, name: str, notify_gui: bool = True, **gui_kwargs) -> None:
        """
        Removes a node from the container.

        ### Parameters
        `name` : str
            The name of the node.
        `notify_gui` : bool
            Whether to notify the gui to remove the node.
        `gui_kwargs` : dict
            Additional keyword arguments to pass to the gui.
        """
        # TODO: add proper logging
        print(f"Removing node '{name}'.")

        self.nodes.remove_node(name)
        if not self.headless and notify_gui:
            Window().remove_node(name, **gui_kwargs)

    @mark_unsaved_changes
    def add_link(
        self, node_out: str, node_in: str, slot_out: str, slot_in: str, notify_gui: bool = True, **gui_kwargs
    ) -> None:
        """
        Adds a link between two nodes.

        ### Parameters
        `node_out` : str
            The name of the output node.
        `node_in` : str
            The name of the input node.
        `slot_out` : str
            The output slot name of `node_out`.
        `slot_in` : str
            The input slot name of `node_in`.
        `notify_gui` : bool
            Whether to notify the gui to add the link.
        `gui_kwargs` : dict
            Additional keyword arguments to pass to the gui.
        """
        # TODO: Prevent multiple links to the same input slot. The GUI already prevents this, but the manager should too.
        self.nodes[node_out].connection.send(
            Message(
                MessageType.ADD_OUTPUT_PIPE,
                {"slot_name_out": slot_out, "slot_name_in": slot_in, "node_connection": self.nodes[node_in].connection},
            )
        )

        if not self.headless and notify_gui:
            Window().add_link(node_out, node_in, slot_out, slot_in, **gui_kwargs)

    @mark_unsaved_changes
    def remove_link(
        self, node_out: str, node_in: str, slot_out: str, slot_in: str, notify_gui: bool = True, **gui_kwargs
    ) -> None:
        """
        Removes a link between two nodes.

        ### Parameters
        `node_out` : str
            The name of the output node.
        `node_in` : str
            The name of the input node.
        `slot_out` : str
            The output slot name of `node_out`.
        `slot_in` : str
            The input slot name of `node_in`.
        `notify_gui` : bool
            Whether to notify the gui to remove the link.
        `gui_kwargs` : dict
            Additional keyword arguments to pass to the gui.
        """
        self.nodes[node_out].connection.send(
            Message(
                MessageType.REMOVE_OUTPUT_PIPE,
                {"slot_name_out": slot_out, "slot_name_in": slot_in, "node_connection": self.nodes[node_in].connection},
            )
        )

        if not self.headless and notify_gui:
            Window().remove_link(node_out, node_in, slot_out, slot_in, **gui_kwargs)

    def terminate(self, notify_gui: bool = True) -> None:
        """
        Terminates the manager and all nodes.

        ### Parameters
        `notify_gui` : bool
            Whether to notify the gui to terminate.
        """
        if not self.headless and notify_gui:
            try:
                # terminate the gui, which calls manager.terminate() with notify_gui=False once it is closed
                Window().terminate()
                return
            except Exception:
                # TODO: add proper logging
                print("Closing the GUI failed.")

        # TODO: add proper logging
        print("Shutting down goofi-pipe manager.")
        # terminate the manager
        self._running = False
        for node in self.nodes:
            self.nodes[node].terminate()

        # close the communication backend
        self._mp_manager.shutdown()

    def save(self, filepath: Optional[str] = None, overwrite: bool = False, timeout: float = 3.0) -> None:
        """
        Saves the state of the manager to a file.

        ### Parameters
        `filepath` : Optional[str]
            The path to the file to save to. If `None`, a default filename is generated in the current directory.
        `overwrite` : bool
            Whether to overwrite an existing file.
        `timeout` : float
            The timeout in seconds for waiting for a response from each node.
        """
        # if no filepath was given, use a default filename in the current directory
        if not filepath and self._save_path:
            filepath = self._save_path
        elif not filepath:
            filepath = "."

        # make sure we get a string
        if not isinstance(filepath, str):
            raise ValueError(f"Expected string, got {type(filepath)}.")

        if path.exists(filepath) and path.isdir(filepath):
            # directory was given, create a default, non-conflicting filename
            idx = 0
            while path.exists(path.join(filepath, f"untitled{idx}.gfi")):
                idx += 1
            filepath = path.join(filepath, f"untitled{idx}.gfi")

        # add the file extension if it is missing
        if not filepath.endswith(".gfi"):
            filepath += ".gfi"

        # check if the file already exists
        if path.exists(filepath) and not overwrite:
            raise FileExistsError(f"File {filepath} already exists.")

        # TODO: add proper logging
        print("Saving manager state...")

        # wait for all nodes to respond, if their serialization_pending flag is set
        start = time.time()
        serialized_nodes = {}
        for name in self.nodes:
            while self.nodes[name].serialization_pending and time.time() - start < timeout:
                # wait for the node to respond or for the timeout to be reached
                time.sleep(0.01)

            if self.nodes[name].serialization_pending:
                # TODO: add proper logging
                print(f"WARNING: Node {name} timed out while waiting for serialization. Node state is possibly outdated.")

            # check if we got a response in time
            if self.nodes[name].serialized_state is None:
                raise RuntimeError(f"Node {name} does not have a serialized state. Recreate the node and try again.")

            if not self.headless:
                # retrieve the GUI state
                gui_kwargs = Window().get_node_state(name)
                if gui_kwargs is not None:
                    self.nodes[name].gui_kwargs = gui_kwargs

            # insert GUI state into the serialized node
            state = deepcopy(self.nodes[name].serialized_state)
            state["gui_kwargs"] = self.nodes[name].gui_kwargs

            # store the serialized state
            serialized_nodes[name] = state

        # generate a list of links from the serialized nodes
        links = []
        for node_name_out, node in serialized_nodes.items():
            # iterate over all output slots of the current node
            for slot_name_out, conns in node["out_conns"].items():
                # filter out self-connections
                conns = [(s, c) for s, c, self_conn in conns if not self_conn]

                # iterate over all connections of the current slot
                for slot_name_in, conn in conns:
                    # find the node that matches the output connection of the current slot
                    for node_name_in in serialized_nodes.keys():
                        # check if the connection matches the current node
                        if conn == self.nodes[node_name_in].connection:
                            # verify that the input slot exists
                            if slot_name_in not in self.nodes[node_name_in].input_slots:
                                continue
                            # verify that the output slot exists
                            if slot_name_out not in self.nodes[node_name_out].output_slots:
                                continue

                            # found the node, add the link
                            links.append(
                                {
                                    "node_out": node_name_out,
                                    "node_in": node_name_in,
                                    "slot_out": slot_name_out,
                                    "slot_in": slot_name_in,
                                }
                            )
                            break
                    # NOTE: it's okay if we didn't find the node, it could be some external connection (e.g. GUI)

        # remove the output connections from the serialized_nodes dict so we can convert it to yaml
        for node in serialized_nodes.values():
            node.pop("out_conns")

        # make sure the number of serialized nodes is correct
        if len(serialized_nodes) != len(self.nodes):
            filepath = f"{filepath}.incomplete"
            # TODO: add proper logging
            print(
                f"WARNING: Mismatch between serialized nodes and actual nodes, saving may be incomplete. Saving to '{filepath}'."
            )

        # convert the manager instance into yaml format
        manager_yaml = yaml.dump({"nodes": serialized_nodes, "links": links}, sort_keys=False)

        # write the yaml to the file
        with open(filepath, "w") as f:
            f.write(manager_yaml)

        # TODO: add proper logging
        print(f"Successfuly saved manager state to '{filepath}'.")

        # store the save path
        self.save_path = filepath
        self.unsaved_changes = False

    @property
    def save_path(self) -> Optional[str]:
        return self._save_path

    @save_path.setter
    def save_path(self, filepath: str) -> None:
        self._save_path = filepath

        # update the window title
        if not self.headless:
            Window().update_title()

    @property
    def unsaved_changes(self) -> bool:
        return self._unsaved_changes

    @unsaved_changes.setter
    def unsaved_changes(self, value: bool) -> None:
        self._unsaved_changes = value

        # update the window title
        if not self.headless:
            Window().update_title()

    @property
    def running(self) -> bool:
        return self._running

    @property
    def headless(self) -> bool:
        return self._headless


def main(duration: float = 0, args=None):
    """
    This is the main entry point for goofi-pipe. It parses command line arguments, creates a manager
    instance and runs all nodes until the manager is terminated.

    ### Parameters
    `duration` : float
        The duration to run the manager for. If `0`, runs indefinitely.
    `args` : list
        A list of arguments to pass to the manager. If `None`, uses `sys.argv[1:]`.
    """
    import argparse

    comm_choices = list(Connection.get_backends().keys())

    # parse arguments
    parser = argparse.ArgumentParser(description="goofi-pipe")
    parser.add_argument("filepath", nargs="?", help="path to the file to load from")
    parser.add_argument("--headless", action="store_true", help="run in headless mode")
    parser.add_argument("--no-multiprocessing", action="store_true", help="disable multiprocessing")
    parser.add_argument("--comm", choices=["auto"] + comm_choices, default="auto", help="node communication backend")
    parser.add_argument("--build-docs", action="store_true", help="update the node list in the README")
    parser.add_argument("--example", nargs="?", const="", help="run example files instead of starting the manager")
    args = parser.parse_args(args)

    if args.build_docs:
        # just update the docs and exit
        docs()
        return

    if args.example is not None:
        if len(args.example) == 0:
            # list example files and exit
            example_dir = Path(__file__).parents[2] / "examples"
            example_files = sorted(example_dir.glob("*.gfi"))
            if not example_files:
                print("No example files found.")
                return
            print("Available example files:")
            for example in example_files:
                print(f" - {example.name}")
            print("Use `--example <filename>` to run an example file.")
            return
        else:
            assert args.filepath is None, "Please specify either a direct filepath or an example, not both."
            args.filepath = str(Path(__file__).parents[2] / "examples" / args.example)

    # create and run the manager (this blocks until the manager is terminated)
    Manager(
        filepath=args.filepath,
        headless=args.headless,
        use_multiprocessing=not args.no_multiprocessing,
        duration=duration,
        communication_backend=args.comm,
    )


def docs():
    """
    Updates the documentation by updating the list of nodes in the README.
    """
    from os import path

    from tqdm import tqdm

    CATEGORY_DESCRIPTIONs = {
        "inputs": "Nodes that provide data to the pipeline.",
        "outputs": "Nodes that send data to external systems.",
        "analysis": "Nodes that perform analysis on the data.",
        "array": "Nodes implementing array operations.",
        "signal": "Nodes implementing signal processing operations.",
        "misc": "Miscellaneous nodes that do not fit into other categories.",
    }

    nodes_cls = list_nodes(verbose=True)

    nodes = dict()
    for node in tqdm(nodes_cls, desc="Collecting node information"):
        if node.category() not in nodes:
            nodes[node.category()] = []

        # collect the node information
        nodes[node.category()].append(
            {
                "name": node.__name__,
                "input_slots": node.config_input_slots(),
                "output_slots": node.config_output_slots(),
            }
        )

    # find the README file
    print("Loading README file...", end="")
    readme_path = path.join(path.dirname(__file__), "..", "..", "README.md")
    readme_path = path.abspath(readme_path)
    assert path.exists(readme_path), f"README file not found: {readme_path}"

    # read the README file
    with open(readme_path, "r") as f:
        readme = f.read()
    print("done")

    # find the start and end of the node list
    start_tag = "<!-- !!GOOFI_PIPE_NODE_LIST_START!! -->"
    end_tag = "<!-- !!GOOFI_PIPE_NODE_LIST_END!! -->"
    start = readme.find(start_tag)
    end = readme.find(end_tag)

    # generate the new node list
    new_nodes = []
    for category, nodes_list in tqdm(nodes.items(), desc="Generating new node list"):
        new_nodes.append(f"## {category.capitalize()}\n")
        new_nodes.append(f"{CATEGORY_DESCRIPTIONs[category]}\n")
        new_nodes.append("<details><summary>View Nodes</summary>\n")
        for node in nodes_list:
            new_nodes.append(f"<details><summary>&emsp;{node['name']}</summary>\n")
            new_nodes.append("  - **Inputs:**")
            for slot, slot_type in node["input_slots"].items():
                new_nodes.append(f"    - {slot}: {slot_type}")
            new_nodes.append("  - **Outputs:**")
            for slot, slot_type in node["output_slots"].items():
                new_nodes.append(f"    - {slot}: {slot_type}")
            new_nodes.append("  </details>\n")
        new_nodes.append("</details>\n")

    # insert the new node list into the README
    print("Updating README file...", end="")
    new_readme = readme[: start + len(start_tag)] + "\n" + "\n".join(new_nodes) + readme[end:]

    # write the updated README
    with open(readme_path, "w") as f:
        f.write(new_readme)
    print("done")


if __name__ == "__main__":
    main()
