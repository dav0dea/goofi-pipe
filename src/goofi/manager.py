"""Manager / orchestrator for goofi-pipe.

After the iceoryx2 transport refactor, the manager:

- assigns a process-wide iceoryx2 instance id and propagates it to spawned
  nodes (so they all agree on service names);
- spawns each node in its own OS process by default (one-node-per-group),
  or hosts all nodes in the manager process when multiprocessing is
  disabled (single group);
- wires links by sending `REGISTER_SUBSCRIBER` on the producer's ctrl
  channel and `SUBSCRIBE_INPUT` on the consumer's ctrl channel — no
  Connection objects cross process boundaries any more;
- reads each node's most recent serialized state directly from the
  corresponding `NodeRef` (push-based, dirty/clean) when saving;
- registers an `atexit` hook to drop iceoryx2 shared-memory entries
  belonging to this instance id.
"""
from __future__ import annotations

import atexit
import glob
import importlib
import os
import subprocess  # noqa: F401  (used by _cleanup_iceoryx2_shm)
import time
import uuid
from copy import deepcopy
from os import path
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from goofi.message import MessageType
from goofi.node import MultiprocessingForbiddenError, Node
from goofi.node_helpers import NodeProcessRegistry, NodeRef, list_nodes
from goofi.transport import data_service_name, get_instance_id, set_instance_id


def mark_unsaved_changes(func):
    def wrapper(self, *args, **kwargs):
        res = func(self, *args, **kwargs)
        self.unsaved_changes = True
        return res

    return wrapper


class NodeContainer:
    """Bookkeeping dict of NodeRefs keyed by unique name."""

    def __init__(self) -> None:
        self._nodes: Dict[str, NodeRef] = {}

    def add_node(self, name: str, node: NodeRef, force_name: bool = False) -> str:
        if not isinstance(name, str):
            raise ValueError(f"Expected string, got {type(name)}.")
        if not isinstance(node, NodeRef):
            raise ValueError(f"Expected NodeRef, got {type(node)}.")

        if force_name:
            if name in self._nodes:
                raise KeyError(f"Node {name} already in container.")
            self._nodes[name] = node
            return name

        idx = 0
        while f"{name}{idx}" in self._nodes:
            idx += 1
        self._nodes[f"{name}{idx}"] = node
        return f"{name}{idx}"

    def remove_node(self, name: str) -> None:
        if name in self._nodes:
            self._nodes[name].terminate()
            del self._nodes[name]
            return
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
    """Goofi-pipe orchestrator."""

    def __init__(
        self,
        filepath: Optional[str] = None,
        headless: bool = True,
        use_multiprocessing: bool = True,
        duration: float = 0,
    ) -> None:
        # Single transport instance id per Manager. Embedded in every
        # iceoryx2 service name; lets multiple goofi-pipe instances coexist
        # on one host without /dev/shm collisions.
        instance_id = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
        set_instance_id(instance_id)
        # Sweep any orphan iceoryx2 nodes from previously-crashed runs.
        _cleanup_iceoryx2_shm(instance_id)
        atexit.register(_cleanup_iceoryx2_shm, instance_id)

        print("Starting goofi-pipe...")
        list_nodes(verbose=True)

        mp_state = "enabled" if use_multiprocessing else "disabled"
        print(f"Initializing goofi-pipe manager (multiprocessing {mp_state}). instance_id={instance_id}")

        self._instance_id = instance_id
        self._headless = headless
        self._use_multiprocessing = use_multiprocessing
        self._running = True
        self.nodes = NodeContainer()
        # node_name -> process_group id. When two nodes share a group, links
        # between them use thread transport instead of iceoryx2.
        self._node_groups: Dict[str, str] = {}
        # explicit link table — manager-owned, replaces the per-node
        # out_conns list from the old code.
        self._links: List[Dict[str, str]] = []

        NodeProcessRegistry().headless = headless

        self._save_path: Optional[str] = None
        self._unsaved_changes = False

        if self.headless:
            self.post_init(filepath, duration)
        else:
            # Window() blocks main thread; everything else runs in a daemon.
            from goofi.gui.window import Window

            Thread(target=self.post_init, args=(filepath, duration), daemon=False).start()
            Window(self)

    def post_init(self, filepath: Optional[str] = None, duration: float = 0) -> None:
        if not self.headless:
            from goofi.gui.window import Window

            win = None
            while win is None:
                try:
                    win = Window()
                except RuntimeError:
                    time.sleep(0.01)
            while not win._initialized:
                time.sleep(0.01)

        if filepath is not None:
            self.load(filepath, load_on_init=True)

        if duration > 0:
            time.sleep(duration)
            self.terminate()

        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.terminate()

    # ------------------------------------------------------------------
    # Node / link CRUD
    # ------------------------------------------------------------------

    def _resolve_group(self, name: str, params: Optional[Dict[str, Dict[str, Any]]]) -> str:
        """Determine the process group for a new node."""
        if not self._use_multiprocessing:
            return "default"
        if params and "common" in params and isinstance(params["common"].get("process_group"), str):
            grp = params["common"]["process_group"]
            if grp:
                return grp
        # Default: one group per node = one process per node.
        return name

    def _same_group(self, node_a: str, node_b: str) -> bool:
        return self._node_groups.get(node_a) == self._node_groups.get(node_b)

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
        print(f"Adding node '{node_type}' from category '{category}'.")

        mod = importlib.import_module(f"goofi.nodes.{category}.{node_type.lower()}")
        node_cls: Node = getattr(mod, node_type)

        # Determine the name *before* spawning so the spawned node uses it
        # as its node_id (which feeds into every service name).
        if name is None:
            base = node_type.lower()
            idx = 0
            while f"{base}{idx}" in self.nodes:
                idx += 1
            assigned_name = f"{base}{idx}"
        else:
            assigned_name = name

        group = self._resolve_group(assigned_name, params)

        ref: Optional[NodeRef] = None
        if self._use_multiprocessing:
            try:
                ref = node_cls.create(node_id=assigned_name, initial_params=params)
            except MultiprocessingForbiddenError:
                ref = None
        if ref is None:
            ref, _ = node_cls.create_local(node_id=assigned_name, initial_params=params)

        ref.set_message_handler(MessageType.SHUTDOWN, lambda *args: self.terminate())

        registered = self.nodes.add_node(assigned_name, ref, force_name=True)
        self._node_groups[registered] = group

        # Best-effort: block briefly for the initial STATE_UPDATE so the
        # rest of the system (save / GUI) has node state to read.
        ref.wait_for_state(timeout=2.0)

        if not self.headless and notify_gui:
            from goofi.gui.window import Window

            Window().add_node(registered, ref, **gui_kwargs)
        return registered

    @mark_unsaved_changes
    def remove_node(self, name: str, notify_gui: bool = True, **gui_kwargs) -> None:
        print(f"Removing node '{name}'.")
        # Drop any links touching this node.
        for link in list(self._links):
            if link["node_out"] == name or link["node_in"] == name:
                self._teardown_link(link, notify_gui=False)
                self._links.remove(link)

        self.nodes.remove_node(name)
        self._node_groups.pop(name, None)
        if not self.headless and notify_gui:
            from goofi.gui.window import Window

            Window().remove_node(name, **gui_kwargs)

    @mark_unsaved_changes
    def add_link(
        self,
        node_out: str,
        node_in: str,
        slot_out: str,
        slot_in: str,
        notify_gui: bool = True,
        **gui_kwargs,
    ) -> None:
        if node_out not in self.nodes:
            raise KeyError(f"No such node: {node_out}")
        if node_in not in self.nodes:
            raise KeyError(f"No such node: {node_in}")

        # Idempotent: if the same link already exists, no-op.
        for link in self._links:
            if (
                link["node_out"] == node_out
                and link["node_in"] == node_in
                and link["slot_out"] == slot_out
                and link["slot_in"] == slot_in
            ):
                return

        src_ref = self.nodes[node_out]
        dst_ref = self.nodes[node_in]
        in_process = self._same_group(node_out, node_in)
        service = data_service_name(node_out, slot_out)

        # Order matters: register on the source first so it knows to
        # publish, then subscribe on the destination.
        src_ref.register_subscriber(slot_out)
        dst_ref.subscribe_input(slot_in, service, in_process)

        self._links.append(
            {"node_out": node_out, "node_in": node_in, "slot_out": slot_out, "slot_in": slot_in}
        )

        if not self.headless and notify_gui:
            from goofi.gui.window import Window

            Window().add_link(node_out, node_in, slot_out, slot_in, **gui_kwargs)

    @mark_unsaved_changes
    def remove_link(
        self,
        node_out: str,
        node_in: str,
        slot_out: str,
        slot_in: str,
        notify_gui: bool = True,
        **gui_kwargs,
    ) -> None:
        link = {"node_out": node_out, "node_in": node_in, "slot_out": slot_out, "slot_in": slot_in}
        if link not in self._links:
            return
        self._teardown_link(link, notify_gui=False)
        self._links.remove(link)
        if not self.headless and notify_gui:
            from goofi.gui.window import Window

            Window().remove_link(node_out, node_in, slot_out, slot_in, **gui_kwargs)

    def _teardown_link(self, link: Dict[str, str], notify_gui: bool) -> None:
        try:
            self.nodes[link["node_out"]].unregister_subscriber(link["slot_out"])
        except KeyError:
            pass
        try:
            self.nodes[link["node_in"]].unsubscribe_input(link["slot_in"])
        except KeyError:
            pass

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self, filepath: str, load_on_init: bool = False) -> None:
        if len(self.nodes) > 0:
            raise RuntimeError("This goofi-pipe already contains nodes.")
        if not path.exists(filepath):
            if load_on_init:
                self.terminate()
            raise FileNotFoundError(f"File '{filepath}' does not exist.")

        print(f"Loading manager state from '{filepath}'...")

        with open(filepath, "r") as f:
            manager_yaml = yaml.load(f, Loader=yaml.FullLoader)

        for name, node in manager_yaml["nodes"].items():
            xpos, ypos = node["gui_kwargs"]["pos"]
            if xpos == np.iinfo(np.int32).min or ypos == np.iinfo(np.int32).min:
                print(f"WARNING: Node '{name}' has a corrupted position. Resetting to (0, 0).")
                node["gui_kwargs"]["pos"] = (0, 0)

            self.add_node(node["_type"], node["category"], name=name, params=node["params"], **node["gui_kwargs"])

        for link in manager_yaml["links"]:
            self.add_link(link["node_out"], link["node_in"], link["slot_out"], link["slot_in"])

        self.save_path = filepath
        self.unsaved_changes = False
        print("Finished loading manager state.")

    def save(self, filepath: Optional[str] = None, overwrite: bool = False, timeout: float = 3.0) -> None:
        """Persist the current graph to a `.gfi` YAML file.

        Reads each node's pushed `serialized_state` directly — no
        request/response round-trip. If a node hasn't pushed yet, waits
        briefly with a small per-node timeout.
        """
        if not filepath and self._save_path:
            filepath = self._save_path
        elif not filepath:
            filepath = "."

        if not isinstance(filepath, str):
            raise ValueError(f"Expected string, got {type(filepath)}.")

        if path.exists(filepath) and path.isdir(filepath):
            idx = 0
            while path.exists(path.join(filepath, f"untitled{idx}.gfi")):
                idx += 1
            filepath = path.join(filepath, f"untitled{idx}.gfi")

        if not filepath.endswith(".gfi"):
            filepath += ".gfi"

        if path.exists(filepath) and not overwrite:
            raise FileExistsError(f"File {filepath} already exists.")

        print("Saving manager state...")

        serialized_nodes: Dict[str, Any] = {}
        for name in self.nodes:
            ref = self.nodes[name]
            ref.wait_for_state(timeout=timeout)
            if ref.serialized_state is None:
                raise RuntimeError(
                    f"Node {name} does not have a serialized state. Recreate the node and try again."
                )
            state = deepcopy(ref.serialized_state)

            if not self.headless:
                from goofi.gui.window import Window

                gui_kwargs = Window().get_node_state(name)
                if gui_kwargs is not None:
                    ref.gui_kwargs = gui_kwargs
            state["gui_kwargs"] = ref.gui_kwargs
            # Drop output-subscriber bookkeeping — it's transient runtime
            # state, not part of the persisted graph definition.
            state.pop("output_subscribers", None)
            serialized_nodes[name] = state

        links = list(self._links)
        manager_yaml = yaml.dump({"nodes": serialized_nodes, "links": links}, sort_keys=False)

        with open(filepath, "w") as f:
            f.write(manager_yaml)

        print(f"Successfully saved manager state to '{filepath}'.")
        self.save_path = filepath
        self.unsaved_changes = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def terminate(self, notify_gui: bool = True) -> None:
        print("Shutting down goofi-pipe manager.")
        self._running = False
        NodeProcessRegistry().terminate()
        for node in list(self.nodes):
            try:
                self.nodes[node].terminate()
            except Exception:
                pass

        if not self.headless and notify_gui:
            try:
                from goofi.gui.window import Window

                Window().terminate()
                return
            except Exception:
                print("Closing the GUI failed.")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def save_path(self) -> Optional[str]:
        return self._save_path

    @save_path.setter
    def save_path(self, filepath: str) -> None:
        self._save_path = filepath
        if not self.headless:
            from goofi.gui.window import Window

            Window().update_title()

    @property
    def unsaved_changes(self) -> bool:
        return self._unsaved_changes

    @unsaved_changes.setter
    def unsaved_changes(self, value: bool) -> None:
        self._unsaved_changes = value
        if not self.headless:
            from goofi.gui.window import Window

            Window().update_title()

    @property
    def running(self) -> bool:
        return self._running

    @property
    def headless(self) -> bool:
        return self._headless

    @property
    def instance_id(self) -> str:
        return self._instance_id

    @property
    def links(self) -> Tuple[Dict[str, str], ...]:
        return tuple(dict(link) for link in self._links)


# ---------------------------------------------------------------------------
# /dev/shm cleanup
# ---------------------------------------------------------------------------


def _cleanup_iceoryx2_shm(instance_id: str) -> None:
    """Best-effort sweep of `/dev/shm/iox2_*` orphans from dead instances.

    iceoryx2 cleans up its own services on graceful Node drop. On crashes
    or SIGKILL the entries linger. We try two cleanups, in order:

    1. `Node.try_cleanup_dead_nodes` — clean reaper for nodes that
       registered themselves and then died. Returns 0 if the registry was
       lost (e.g., after manager hot-restart on the same /dev/shm).
    2. Filesystem sweep — for every `/dev/shm/iox2_*` entry that doesn't
       belong to a currently-running process, delete it. We identify
       liveness by reading file mtime + PID-presence heuristics; this is a
       best-effort fallback because iceoryx2's on-disk naming uses an
       opaque hash rather than the owning PID.
    """
    try:
        import iceoryx2 as iox2

        iox2.Node.try_cleanup_dead_nodes(iox2.ServiceType.Ipc, iox2.config.global_config())
    except Exception:
        pass

    # Filesystem-level sweep — delete `/dev/shm/iox2_*` entries older than
    # 60s that no live goofi process is holding open. We use `lsof` if
    # available; otherwise fall back to age-based deletion. Run only on
    # startup (when nothing of ours is yet attached); the atexit path
    # would race with our own still-attached endpoints.
    if instance_id in _cleanup_iceoryx2_shm._swept:
        return
    _cleanup_iceoryx2_shm._swept.add(instance_id)

    try:
        import subprocess

        entries = glob.glob("/dev/shm/iox2_*")
        if not entries:
            return
        # Find which entries are currently held open by any process.
        held: set[str] = set()
        try:
            out = subprocess.run(
                ["lsof", "+D", "/dev/shm"], capture_output=True, text=True, timeout=2.0
            )
            for line in out.stdout.splitlines():
                if "iox2_" in line:
                    parts = line.split()
                    held.add(parts[-1])
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        for entry in entries:
            if entry in held:
                continue
            try:
                # Only sweep if older than 60 seconds — gives concurrent
                # starts a chance to attach before we wipe them.
                if time.time() - os.path.getmtime(entry) > 60:
                    os.remove(entry)
            except OSError:
                pass
    except Exception:
        pass


_cleanup_iceoryx2_shm._swept = set()


def get_example_patch(args) -> bool:
    if len(args.example) == 0:
        example_dir = Path(__file__).parents[2] / "examples"
        example_files = sorted(example_dir.glob("*.gfi"))
        if not example_files:
            print("No example files found.")
            return False
        print("Available example files:")
        for example_arg in example_files:
            print(f" - {example_arg.name}")
        print("Use `--example <filename>` to run an example file.")
        return False
    assert args.filepath is None, "Please specify either a direct filepath or an example, not both."
    args.filepath = str(Path(__file__).parents[2] / "examples" / args.example)
    return True


def main(duration: Optional[float] = None, args=None):
    import argparse

    parser = argparse.ArgumentParser(description="goofi-pipe")
    parser.add_argument("filepath", nargs="?", help="path to the file to load from")
    parser.add_argument("--headless", action="store_true", help="run in headless mode")
    parser.add_argument("--no-multiprocessing", action="store_true", help="disable multiprocessing")
    parser.add_argument(
        "--duration",
        default=0,
        type=float,
        help="Duration (in seconds) after which goofi-pipe automatically shuts down (0 to run indefinitely)",
    )
    parser.add_argument("--update-readme-docs", action="store_true", help="update the node list in the README")
    parser.add_argument(
        "--gen-node-docs", action="store_true", help="generate missing node docstrings using the openai API"
    )
    parser.add_argument("--example", nargs="?", const="", help="run example files instead of starting the manager")
    args = parser.parse_args(args)

    if args.update_readme_docs:
        from goofi.doc_utils import update_docs

        update_docs()
        return

    if args.gen_node_docs:
        from goofi.doc_utils import gen_node_docs

        gen_node_docs()
        return

    if args.example is not None:
        if not get_example_patch(args):
            return

    if duration is not None and args.duration != 0:
        raise ValueError(
            "Manager duration should be given either as a parameter or as a command line argument, not both."
        )
    elif duration is None:
        duration = args.duration

    Manager(
        filepath=args.filepath,
        headless=args.headless,
        use_multiprocessing=not args.no_multiprocessing,
        duration=duration,
    )


if __name__ == "__main__":
    main()
