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
import contextlib
import importlib
import logging
import os
import re
import threading
import time
import uuid
from copy import deepcopy
from os import path
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import yaml

from goofi.message import MessageType
from goofi.node import MultiprocessingForbiddenError, Node
from goofi.node_helpers import NodeProcessRegistry, NodeRef, list_nodes
from goofi.transport import ensure_iox2_runtime_dirs, set_instance_id

if TYPE_CHECKING:
    from goofi.bridge.server import BridgeServer


# Reserved separator for namespaced sub-patch member names (e.g. "sub0::osc0").
# A user/file node name may never contain it, so the grouping runtime can mint
# collision-free qualified names and round-trip them unambiguously.
SUBPATCH_SEP = "::"

logger = logging.getLogger(__name__)

# Matches a string-literal `nd('name')` / `nd("name")` reference (up to the name's
# closing quote), capturing the quote and the exact name. Whitespace after `nd(`
# is tolerated and normalized away on rewrite; the closing paren is left untouched.
_ND_REF = re.compile(r"nd\(\s*(?P<q>['\"])(?P<name>[^'\"]*)(?P=q)")


def _rewrite_nd_literal(expr: Optional[str], rename_map: Dict[str, str]) -> Optional[str]:
    """Best-effort rewrite of string-literal `nd('name')` references whose name is a
    key of ``rename_map`` to the mapped name. References to names not in the map
    (external producers) are left untouched. Non-string-literal / dynamic nd() args
    are out of scope (they can't be statically rewritten)."""
    if not expr or "nd(" not in expr:
        return expr

    def _repl(m: "re.Match[str]") -> str:
        new = rename_map.get(m.group("name"))
        if new is None:
            return m.group(0)
        q = m.group("q")
        return f"nd({q}{new}{q}"

    return _ND_REF.sub(_repl, expr)


def _reject_reserved_name(name: str) -> None:
    if SUBPATCH_SEP in name:
        raise ValueError(
            f"node name {name!r} may not contain the reserved separator '{SUBPATCH_SEP}'"
        )


class SubPatchTooDeep(ValueError):
    """A namespaced member name would overflow iceoryx2's 255-byte ServiceName."""


def mark_unsaved_changes(func):
    def wrapper(self, *args, **kwargs):
        res = func(self, *args, **kwargs)
        self.unsaved_changes = True
        return res

    return wrapper


class NodeContainer:
    """Bookkeeping dict of NodeRefs keyed by stable `uid` (insertion-ordered).

    The key is the node's universal identity, never its display name — so a node
    can be renamed without re-keying anything here. Display-name generation and
    uniqueness live in `Manager.add_node` (a display concern), not here."""

    def __init__(self) -> None:
        self._nodes: Dict[str, NodeRef] = {}  # uid -> NodeRef

    def add_node(self, uid: str, node: NodeRef) -> str:
        if not isinstance(uid, str):
            raise ValueError(f"Expected string uid, got {type(uid)}.")
        if not isinstance(node, NodeRef):
            raise ValueError(f"Expected NodeRef, got {type(node)}.")
        if uid in self._nodes:
            raise KeyError(f"Node {uid} already in container.")
        self._nodes[uid] = node
        return uid

    def remove_node(self, uid: str) -> None:
        if uid in self._nodes:
            self._nodes[uid].terminate()
            del self._nodes[uid]
            return
        raise KeyError(f"Node {uid} not in container")

    def replace(self, uid: str, node: NodeRef) -> None:
        """Swap the NodeRef behind a uid in place (restart), preserving insertion
        order so the editor's node ordering is stable."""
        if uid not in self._nodes:
            raise KeyError(f"Node {uid} not in container")
        self._nodes[uid] = node

    def __getitem__(self, uid: str) -> NodeRef:
        return self._nodes[uid]

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self):
        return iter(self._nodes.keys())

    def __contains__(self, uid: str) -> bool:
        return uid in self._nodes


class Manager:
    """Goofi-pipe orchestrator."""

    def __init__(
        self,
        filepath: Optional[str] = None,
        headless: bool = True,
        use_multiprocessing: bool = True,
        duration: float = 0,
        bridge_host: str = "127.0.0.1",
        bridge_port: int = 8000,
    ) -> None:
        # Single transport instance id per Manager. Embedded in every
        # iceoryx2 service name; lets multiple goofi-pipe instances coexist
        # on one host without /dev/shm collisions.
        instance_id = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
        set_instance_id(instance_id)
        # Create iceoryx2's runtime directory layout before any transport use.
        # Must precede the cleanup sweep below, which scans those directories.
        ensure_iox2_runtime_dirs()
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
        # explicit link table — manager-owned, replaces the per-node out_conns
        # list from the old code. Endpoints are node UIDs (node_out/node_in).
        self._links: List[Dict[str, str]] = []
        # NB: there is no separate uid index — `self.nodes` (NodeContainer) IS
        # keyed by uid, so it is the single uid->NodeRef map.
        # Sub-patch state (flatten-at-runtime). The live graph stays flat; these
        # maps are the first-class record of grouping (NOT re-derived from name
        # prefixes). `_membership` maps a member's uid -> instance id;
        # `_instances` holds per-instance {kind, interface, pos, members} where
        # members maps member uid -> local name; `_definitions` holds shared
        # sub-patch graphs (populated in the sharing phase).
        self._membership: Dict[str, str] = {}
        self._instances: Dict[str, Dict[str, Any]] = {}
        self._definitions: Dict[str, Dict[str, Any]] = {}

        NodeProcessRegistry().headless = headless

        self._save_path: Optional[str] = None
        self._unsaved_changes = False
        # Opaque frontend workspace-layout blob (panel/tab arrangement). The
        # backend never interprets it — it round-trips through the .gfi file so
        # a saved patch carries its UI layout. Set by the bridge on save, read
        # back on load, and surfaced to the browser via the graph snapshot.
        self._layout: Optional[Any] = None

        # Bridge (browser frontend). Started in non-headless mode; the
        # process keeps running until terminate() — main thread serves
        # KeyboardInterrupt either way.
        self._bridge: Optional["BridgeServer"] = None
        self._bridge_host = bridge_host
        self._bridge_port = bridge_port

        if not self.headless:
            from goofi.bridge.server import start_bridge

            self._bridge = start_bridge(self, host=bridge_host, port=bridge_port)
            if self._bridge.url:
                print(f"  goofi-pipe is running. Open {self._bridge.url} in your browser.\n")

        # Watch node processes for crashes and auto-restart them (works headless;
        # no-op when multiprocessing is disabled).
        self._start_supervisor()

        self.post_init(filepath, duration)

    def post_init(self, filepath: Optional[str] = None, duration: float = 0) -> None:
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

    def _resolve_group(self, node_id: str, params: Optional[Dict[str, Dict[str, Any]]]) -> str:
        """Determine the process group for a new node.

        The default group is the node's unique id, so each default node lands
        in its own group (= its own process) and `_spawn_node`'s `group !=
        node_id` test stays a reliable "explicit group?" check.
        """
        if not self._use_multiprocessing:
            return "default"
        if params and "common" in params and isinstance(params["common"].get("process_group"), str):
            grp = params["common"]["process_group"]
            if grp:
                return grp
        # Default: one group per node = one process per node.
        return node_id

    def _same_group(self, node_a: str, node_b: str) -> bool:
        return self._node_groups.get(node_a) == self._node_groups.get(node_b)

    def _spawn_node(
        self,
        node_cls: type,
        node_id: str,
        params: Optional[Dict[str, Dict[str, Any]]],
        group: str,
    ) -> NodeRef:
        """Pick a spawn strategy and return the resulting `NodeRef`.

        Three paths:
        - `--no-multiprocessing`: host the node in the manager's process
          (thread-based ctrl/status, fastest startup).
        - explicit `process_group` set: host the node inside a shared
          subprocess managed by `NodeProcessRegistry` so peers in the same
          group can communicate via thread transport.
        - default: spawn the node in its own subprocess via `Node.create()`.

        Multiprocessing-forbidden nodes always fall back to `create_local()`.
        """
        capture_logs = not self._headless
        if not self._use_multiprocessing:
            ref, _ = node_cls.create_local(
                node_id=node_id, initial_params=params, capture_logs=capture_logs
            )
            return ref
        if group != node_id:  # explicit group → registry host process
            pg = NodeProcessRegistry().get(group, self._instance_id)
            pg.spawn(node_cls, node_id, params)
            # Build the manager-side ref. _build_ref configures params from
            # cls._configure() and applies the dict overrides; the actual
            # node lives in the group host process.
            _, _, params_obj = node_cls._configure()
            if params is not None:
                try:
                    params_obj.update(params)
                except Exception:
                    pass
            return node_cls._build_ref(node_id, params_obj, in_process=False, process=None)
        try:
            return node_cls.create(
                node_id=node_id, initial_params=params, capture_logs=capture_logs
            )
        except MultiprocessingForbiddenError:
            ref, _ = node_cls.create_local(
                node_id=node_id, initial_params=params, capture_logs=capture_logs
            )
            return ref

    def _mint_uid(self) -> str:
        """A fresh node uid not currently in use (48 bits + dedup pass)."""
        while True:
            uid = uuid.uuid4().hex[:12]
            if uid not in self.nodes:
                return uid

    def _service_budget_ok(self, name: str, slots=None) -> bool:
        """Whether `name` fits iceoryx2's 255-byte ServiceName once embedded.

        Service names are `goofi.{instance_id}.{kind}.{node_id}.{slot}` where
        `node_id = f"{name}-{uuid8}"` (manager.py mints an 8-hex suffix). When the
        node's real slot names are known, check the LONGEST one; otherwise fall back
        to a generous 48-char slot allowance. Either way deep sub-patch nesting fails
        early with a clear error rather than a late iceoryx2 crash mid-spawn.
        """
        slot = max((s for s in (slots or ())), key=len, default="s" * 48)
        worst = f"goofi.{self._instance_id}.status.{name}-deadbeef.{slot}"
        return len(worst.encode()) <= 255

    @mark_unsaved_changes
    def add_node(
        self,
        node_type: str,
        category: str,
        notify_gui: bool = True,
        name: Optional[str] = None,
        params: Optional[Dict[str, Dict[str, Any]]] = None,
        member_uid: Optional[str] = None,
        membership: Optional[Dict[str, Any]] = None,
        allow_reserved: bool = False,
        **gui_kwargs,
    ) -> str:
        print(f"Adding node '{node_type}' from category '{category}'.")
        # User/file top-level names may not contain the reserved separator. The
        # sub-patch expander sets allow_reserved=True for namespaced member names.
        if name is not None and not allow_reserved:
            _reject_reserved_name(name)
        if name is not None and not self._service_budget_ok(name):
            raise SubPatchTooDeep(
                f"node name {name!r} is too long for an iceoryx2 service name "
                "(deep sub-patch nesting). Flatten or shorten names."
            )

        mod = importlib.import_module(f"goofi.nodes.{category}.{node_type.lower()}")
        node_cls: type = getattr(mod, node_type)

        # Two independent identities:
        #  - `uid` (universal): the key everything references the node by; minted
        #    once, never reused, stable across rename/restart/reload.
        #  - display `name`: the user-facing label, auto-numbered per type and
        #    reused once freed (delete oscillator0 -> the next Oscillator is
        #    oscillator0 again). It keys NOTHING — purely display + `nd()`.
        #  - `node_id` (transport): feeds every iceoryx2 service name; carries a
        #    name prefix for debug + a fresh uuid suffix so a quick re-add can't
        #    race the old, still-terminating node for a per-service slot.
        if name is None:
            base = node_type.lower()
            existing = {self.nodes[u].name for u in self.nodes}
            idx = 0
            while f"{base}{idx}" in existing:
                idx += 1
            assigned_name = f"{base}{idx}"
        else:
            assigned_name = name

        # Mint (or restore, on load) the universal uid — the container key.
        if member_uid is None or member_uid in self.nodes:
            member_uid = self._mint_uid()

        node_id = f"{assigned_name}-{uuid.uuid4().hex[:8]}"
        group = self._resolve_group(node_id, params)

        ref = self._spawn_node(node_cls, node_id, params, group)
        ref.uid = member_uid
        ref.name = assigned_name
        ref.membership = membership
        ref.set_message_handler(MessageType.SHUTDOWN, lambda *args: self.terminate())

        # Preserve gui_kwargs (notably 'pos') for the bridge / patch save.
        if gui_kwargs:
            ref.gui_kwargs = dict(gui_kwargs)

        self.nodes.add_node(member_uid, ref)
        self._node_groups[member_uid] = group

        # Best-effort: block briefly for the initial STATE_UPDATE so the
        # rest of the system (save / bridge) has node state to read.
        ref.wait_for_state(timeout=2.0)

        # Refresh every node's name->id directory so `nd('name')` expression
        # references (including the new node's own) resolve to current ids.
        self._broadcast_node_directory()

        if self._bridge is not None and notify_gui:
            self._bridge.control.on_node_added(member_uid)
        return member_uid

    @mark_unsaved_changes
    def remove_node(self, uid: str, notify_gui: bool = True, **gui_kwargs) -> None:
        print(f"Removing node '{uid}'.")
        # A node that's still a member of a sub-patch (remove_instance pops
        # membership BEFORE calling this, so teardown skips the whole block):
        _inst_id = self._membership.get(uid)
        if _inst_id is not None:
            _inst = self._instances.get(_inst_id)
            if _inst is not None:
                # Deleting a single member of a SHARED sub-patch would desync it from
                # its definition/siblings (topology isn't mirrored) and leave dangling
                # ports — block it; the user must make the instance unique first.
                if _inst.get("def_id"):
                    raise ValueError(
                        f"cannot delete member {uid!r} of a shared sub-patch; make it unique first"
                    )
                # Unique: unwire any boundary pointing at this member (its spliced
                # external links drop below with the node's links) and drop the member
                # from the instance so save/iteration never references a gone node.
                _local = _inst["members"].get(uid)
                for _bid, _e in list(_inst["interface"].items()):
                    if _e.get("inner_node") == _local:
                        _inst["interface"][_bid] = {**_e, "inner_node": None, "inner_slot": None}
                _inst["members"].pop(uid, None)
                self._membership.pop(uid, None)
        # Drop any links touching this node.
        for link in list(self._links):
            if link["node_out"] == uid or link["node_in"] == uid:
                self._teardown_link(link, notify_gui=False)
                self._links.remove(link)

        # NodeContainer is the uid index — removing from it drops the node.
        self.nodes.remove_node(uid)
        self._node_groups.pop(uid, None)
        # Refresh the directory so surviving nodes stop resolving `nd('name')`
        # to the node that just left.
        self._broadcast_node_directory()
        if self._bridge is not None and notify_gui:
            self._bridge.control.on_node_removed(uid)

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

        # Each input slot accepts exactly one wire. If `slot_in` on
        # `node_in` is already occupied by a different source, tear down
        # the old wire first — keeping the data plane consistent with the
        # graph definition.
        for existing in list(self._links):
            if existing["node_in"] == node_in and existing["slot_in"] == slot_in:
                self.remove_link(
                    existing["node_out"],
                    existing["node_in"],
                    existing["slot_out"],
                    existing["slot_in"],
                    notify_gui=notify_gui,
                )

        link = {"node_out": node_out, "node_in": node_in, "slot_out": slot_out, "slot_in": slot_in}
        self._wire_link(link)
        self._links.append(link)

        if self._bridge is not None and notify_gui:
            self._bridge.control.on_link_added(link)

    def _wire_link(self, link: Dict[str, str]) -> None:
        """Establish the data-plane wire for one link from the CURRENT refs.

        Derives the service from the source node's stable transport id (not its
        reusable display name), so it is also correct after a restart re-mints that
        id. Order matters: register on the source first so it knows to publish,
        then subscribe on the destination."""
        src_ref = self.nodes[link["node_out"]]
        dst_ref = self.nodes[link["node_in"]]
        in_process = self._same_group(link["node_out"], link["node_in"])
        src_ref.register_subscriber(link["slot_out"])
        dst_ref.subscribe_input(link["slot_in"], src_ref.data_service_for(link["slot_out"]), in_process)

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
        if self._bridge is not None and notify_gui:
            self._bridge.control.on_link_removed(link)

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
    # Liveness supervision + auto-restart of crashed node processes
    # ------------------------------------------------------------------

    @staticmethod
    def _node_is_dead(ref: NodeRef) -> bool:
        """True when a node's OWN OS process has exited. Only one-node-per-process
        nodes carry a process; LOCAL (in-manager) and shared process-group members
        have ``ref.process is None`` and are not supervised here."""
        proc = ref.process
        return proc is not None and not proc.is_alive()

    def restart_node(self, uid: str) -> NodeRef:
        """Respawn a node whose process died, preserving identity + links.

        Mints a FRESH transport id (the old one must never be reused — see
        add_node), re-applies the last-known params, and re-wires every link
        touching the node in BOTH directions (a new id changes the node's service
        names, so a downstream consumer must re-subscribe to the new service).
        Display name, member_uid, sub-patch membership and gui position are kept;
        the link table is untouched. Restarts are unlimited; the count rides on the
        new ref for observability."""
        with (getattr(self, "_supervisor_lock", None) or contextlib.nullcontext()):
            old = self.nodes[uid]
            node_cls = old.node_class
            try:
                params = old.params.serialize()
            except Exception:
                params = None
            display, membership = old.name, old.membership
            gui_kwargs = dict(old.gui_kwargs)
            count = old.restart_count + 1

            # Drop the dead manager-side ref (stops its messaging loop / endpoints).
            try:
                old.terminate()
            except Exception:
                pass

            # Fresh transport id, keeping the display name as the debug prefix.
            new_id = f"{display}-{uuid.uuid4().hex[:8]}"
            # Recompute the group from the NEW id: a default node's group IS its
            # node_id, so it must track the new id to stay "own process" (else
            # _spawn_node's `group != node_id` test would misroute it to the
            # shared-group registry). An explicit process_group is preserved.
            group = self._resolve_group(new_id, params)
            new_ref = self._spawn_node(node_cls, new_id, params, group)
            new_ref.set_message_handler(MessageType.SHUTDOWN, lambda *args: self.terminate())
            new_ref.gui_kwargs = gui_kwargs
            new_ref.membership = membership
            new_ref.uid = uid
            new_ref.name = display
            new_ref.restart_count = count

            # The container IS the uid index, so replace() repoints it in place.
            self.nodes.replace(uid, new_ref)
            self._node_groups[uid] = group

            # Re-wire the bridge status to the NEW ref BEFORE waiting for state, so
            # the respawned node's first push (and its healthy error-clear) reaches
            # the browser and lifts the crash chip.
            if self._bridge is not None:
                try:
                    self._bridge.control.rewire_node_status(uid)
                except Exception:
                    logger.exception("restart: bridge re-wire failed for %s", uid)

            new_ref.wait_for_state(timeout=2.0)

            # Re-establish every link touching this node from the current refs.
            for link in self._links:
                if link["node_out"] == uid or link["node_in"] == uid:
                    try:
                        self._wire_link(link)
                    except Exception as exc:
                        logger.warning("restart: failed to rewire link %s: %s", link, exc)

            self._broadcast_node_directory()
            return new_ref

    def _supervise_once(self) -> None:
        """One liveness sweep: respawn any node whose process has died, announcing
        the crash to the browser first so the node visibly flips to an error."""
        for uid in list(self.nodes):
            try:
                ref = self.nodes[uid]
            except KeyError:
                continue
            if not self._node_is_dead(ref):
                continue
            exitcode = ref.process.exitcode if ref.process is not None else None
            count = ref.restart_count + 1
            print(f"supervisor: node '{ref.name}' process died (exit {exitcode}) — restarting (#{count})")
            if self._bridge is not None:
                try:
                    self._bridge.control.on_node_crashed(uid, exitcode, count)
                except Exception:
                    logger.exception("supervisor: crash broadcast failed for %s", uid)
            try:
                self.restart_node(uid)
            except Exception:
                logger.exception("supervisor: restart of %s failed", uid)

    def _supervisor_loop(self) -> None:
        while self._running and not self._supervisor_stop.is_set():
            self._supervisor_stop.wait(0.5)
            if not self._running or self._supervisor_stop.is_set():
                break
            try:
                with self._supervisor_lock:
                    self._supervise_once()
            except Exception:
                logger.exception("supervisor sweep failed")

    def _start_supervisor(self) -> None:
        """Spin up the liveness supervisor daemon. No-op without multiprocessing —
        LOCAL nodes share the manager process and can't crash independently."""
        if not self._use_multiprocessing:
            return
        self._supervisor_lock = threading.RLock()
        self._supervisor_stop = threading.Event()
        self._supervisor_thread = threading.Thread(
            target=self._supervisor_loop, name="goofi-supervisor", daemon=True
        )
        self._supervisor_thread.start()

    def _stop_supervisor(self) -> None:
        stop = getattr(self, "_supervisor_stop", None)
        if stop is not None:
            stop.set()
        thread = getattr(self, "_supervisor_thread", None)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)

    # ------------------------------------------------------------------
    # Sub-patches (flatten-at-runtime)
    # ------------------------------------------------------------------

    @mark_unsaved_changes
    def rename_node(self, uid: str, name: str) -> None:
        """Set a node's mutable DISPLAY name. Safe BY CONSTRUCTION: nothing keys on
        the name (the graph is uid-keyed), so no reference moves — we just write the
        label, refresh the `nd()` directory, and notify the browser. The node's
        transport id and uid are untouched, so data subscriptions keep flowing."""
        if uid not in self.nodes:
            raise KeyError(f"No such node: {uid}")
        self.nodes[uid].name = name
        self._broadcast_node_directory()
        if self._bridge is not None:
            self._bridge.control.on_node_renamed(uid, name)

    def _fresh_instance_id(self) -> str:
        idx = 0
        while True:
            cand = f"subpatch{idx}"
            if cand not in self._instances and cand not in self.nodes:
                return cand
            idx += 1

    def _slot_dtype(self, display: str, slot: str, dir: str) -> str:
        """Name of a node slot's DataType ('ARRAY'/'STRING'/'TABLE'), default ARRAY."""
        ref = self.nodes[display]
        slots = ref.input_slots if dir == "in" else ref.output_slots
        dt = slots.get(slot)
        return dt.name if dt is not None else "ARRAY"

    def _beside_member_pos(self, display: str, dir: str) -> list:
        """A default In/Out pill position beside its inner member (In left, Out right)."""
        pos = (self.nodes[display].gui_kwargs or {}).get("pos") or [0, 0]
        dx = -280 if dir == "in" else 320
        return [int(pos[0]) + dx, int(pos[1])]

    def _derive_interface(self, members: Dict[str, str]) -> Dict[str, Any]:
        """Auto-derive the boundary interface from links crossing the member set.

        A member input slot fed from outside is a boundary input; a member output
        feeding outside is a boundary output. Entries are first-class boundary
        records (dir/dtype/inner_node/inner_slot/pos) identical in shape to
        authored In/Out nodes — the live graph stays flat, so this does not affect
        runtime or save/load round-tripping. One boundary per inner (node, slot).
        """
        iface: Dict[str, Any] = {}
        mset = set(members)
        for link in self._links:
            out_m = link["node_out"] in mset
            in_m = link["node_in"] in mset
            if in_m and not out_m:
                disp, local, slot, dir = link["node_in"], members[link["node_in"]], link["slot_in"], "in"
            elif out_m and not in_m:
                disp, local, slot, dir = link["node_out"], members[link["node_out"]], link["slot_out"], "out"
            else:
                continue
            key = f"{local}.{slot}"
            if key in iface:
                continue  # one boundary per inner slot (a 2nd external consumer fans out on the port)
            iface[key] = {
                "dir": dir,
                "dtype": self._slot_dtype(disp, slot, dir),
                "inner_node": local,
                "inner_slot": slot,
                "pos": self._beside_member_pos(disp, dir),
            }
        return iface

    @contextlib.contextmanager
    def _transaction(self):
        """Atomic rollback for multi-node sub-patch mutations (spec §2.10, backlog #2).

        Snapshots the in-memory graph-state maps and the live node set on entry. On an
        exception inside the block, tears down any nodes spawned during the block (so a
        partial splice leaves no orphan process) and restores the maps in place — the
        graph is byte-identical to before. On success the changes commit. (Live transport
        re-wiring of an externally-displaced link is not replayed here; the restored
        _links list re-wires on the next load.)"""
        snap_links = deepcopy(self._links)
        snap_groups = deepcopy(self._node_groups)
        snap_membership = deepcopy(self._membership)
        snap_instances = deepcopy(self._instances)
        snap_definitions = deepcopy(self._definitions)
        before_nodes = set(self.nodes)
        try:
            yield
        except Exception:
            for n in set(self.nodes) - before_nodes:
                try:
                    self.remove_node(n, notify_gui=False)
                except Exception:
                    pass
            self._links[:] = snap_links
            self._node_groups.clear()
            self._node_groups.update(snap_groups)
            self._membership.clear()
            self._membership.update(snap_membership)
            self._instances.clear()
            self._instances.update(snap_instances)
            self._definitions.clear()
            self._definitions.update(snap_definitions)
            raise

    def _surface_mirror_failure(self, node: str, what: str, exc: Exception) -> None:
        """Report a strict-mirror propagation failure for a shared-sibling member —
        logged, and pushed to the UI as an error event when a bridge is attached — so a
        silently-diverging shared family is visible instead of swallowed (backlog #8)."""
        logger.warning("strict-mirror: failed to propagate %s to sibling %s: %s", what, node, exc)
        if self._bridge is not None:
            try:
                self._bridge.control.broadcast_threadsafe(
                    {"event": "error", "payload": {"node": node, "error": f"shared-mirror failed ({what}): {exc}"}}
                )
            except Exception:
                pass

    def _rewrite_member_expressions(self, member_names, rename_map: Dict[str, str]) -> None:
        """Rewrite string-literal nd() refs in each member's param expressions to the
        renamed fellow members (group: bare->qualified; expand: qualified->bare),
        preserving the enable/trigger/autoeval flags. Best-effort: members without a
        live ref/params are skipped."""
        for name in member_names:
            if name not in self.nodes:
                continue
            ref = self.nodes[name]
            for grp in list(ref.params.keys()):
                for pname, p in ref.params[grp].items():
                    expr = getattr(p, "expression", None)
                    new_expr = _rewrite_nd_literal(expr, rename_map)
                    if new_expr != expr:
                        ref.set_expression(
                            grp,
                            pname,
                            new_expr,
                            enabled=bool(getattr(p, "expression_enabled", False)),
                            triggers_process=bool(getattr(p, "expression_triggers_process", False)),
                            autoeval=bool(getattr(p, "expression_autoeval", False)),
                        )

    @mark_unsaved_changes
    def group_nodes(
        self,
        member_names,
        interface: Optional[Dict[str, Any]] = None,
        pos=(0, 0),
        notify_gui: bool = True,
    ) -> str:
        """Group existing nodes into a unique (inline) sub-patch instance.

        Members keep their uids (no respawn — node_id and data subscriptions
        survive); only their DISPLAY name is qualified to `inst_id::local`.
        Membership/interface are recorded as first-class state, keyed by uid.
        Returns the new instance id. `member_names` is a list of member UIDs.
        """
        member_uids = list(member_names)
        if not member_uids:
            raise ValueError("no members to group")
        for u in member_uids:
            if u not in self.nodes:
                raise KeyError(f"No such node: {u}")
            if u in self._membership:
                raise ValueError(f"node {u} is already in a sub-patch")
            _reject_reserved_name(self.nodes[u].name)

        inst_id = self._fresh_instance_id()
        members: Dict[str, str] = {}  # uid -> local name
        rename_map: Dict[str, str] = {}  # old display -> qualified display (for nd())
        done: list = []  # (uid, local) renamed so far, for rollback
        try:
            for u in member_uids:
                ref = self.nodes[u]
                local = ref.name
                new_name = f"{inst_id}{SUBPATCH_SEP}{local}"
                slots = list(ref.output_slots or ()) + list(ref.input_slots or ())
                if not self._service_budget_ok(new_name, slots=slots):
                    raise SubPatchTooDeep(f"grouping would overflow service-name budget for {new_name!r}")
                ref.name = new_name  # display only — uid (the key) is unchanged
                done.append((u, local))
                members[u] = local
                rename_map[local] = new_name
        except Exception:
            # Roll back any display renames so a failure leaves the graph intact.
            for u, local in reversed(done):
                self.nodes[u].name = local
            raise

        if interface is None:
            interface = self._derive_interface(members)
        self._instances[inst_id] = {
            "kind": "unique",
            "def_id": None,
            "interface": interface,
            "pos": list(pos),
            "members": members,
        }
        for u, local in members.items():
            self._membership[u] = inst_id
            self.nodes[u].membership = {"instance": inst_id, "local_name": local}

        # Rewrite intra-group nd('name') references to the qualified member names so
        # cross-references survive the rename (spec §2.6, backlog #1).
        self._rewrite_member_expressions(members.keys(), rename_map)
        self._broadcast_node_directory()

        if self._bridge is not None and notify_gui:
            self._bridge.control.on_subpatch_changed()
        return inst_id

    def _fresh_member_local(self, inst_id: str, node_type: str, requested: Optional[str]) -> str:
        """Pick a local member name unique within the sub-patch. For a shared
        instance the local namespace is the family's (def + every sibling carry
        the same locals by strict mirror), so the instance's own members suffice
        — but we also fold in the def's locals defensively."""
        existing = set(self._instances[inst_id]["members"].values())
        def_id = self._instances[inst_id].get("def_id")
        if def_id and def_id in self._definitions:
            existing |= set(self._definitions[def_id]["members"].keys())
        if requested is not None:
            _reject_reserved_name(requested)
            if requested in existing:
                raise ValueError(f"a member named {requested!r} already exists in {inst_id}")
            return requested
        base = node_type.lower()
        idx = 0
        while f"{base}{idx}" in existing:
            idx += 1
        return f"{base}{idx}"

    @mark_unsaved_changes
    def add_member_node(
        self,
        inst_id: str,
        node_type: str,
        category: str,
        name: Optional[str] = None,
        params: Optional[Dict[str, Dict[str, Any]]] = None,
        pos=(0, 0),
        notify_gui: bool = True,
    ) -> str:
        """Create a node directly inside an existing sub-patch instance.

        The node is spawned with a namespaced display name (`inst_id::local`) and
        recorded in the instance's membership/members maps. For a SHARED instance
        the new member is mirrored into the definition and every sibling instance
        (strict mirror), matching `update_param`/`set_node_pos`. Returns the new
        member's display name.
        """
        if inst_id not in self._instances:
            raise KeyError(f"No such sub-patch: {inst_id}")
        inst = self._instances[inst_id]
        local = self._fresh_member_local(inst_id, node_type, name)
        disp = f"{inst_id}{SUBPATCH_SEP}{local}"
        if not self._service_budget_ok(disp):
            raise SubPatchTooDeep(f"adding {disp!r} would overflow the service-name budget")

        # Spawn silently; the single on_subpatch_changed below re-syncs clients
        # atomically (node + updated members map) without a top-level flash.
        uid = self.add_node(
            node_type,
            category,
            notify_gui=False,
            name=disp,
            params=params,
            pos=tuple(pos),
            allow_reserved=True,
            membership={"instance": inst_id, "local_name": local},
        )
        self._membership[uid] = inst_id
        inst["members"][uid] = local

        def_id = inst.get("def_id")
        if def_id:
            rec = self._node_record(uid)
            rec.pop("uid", None)  # per-instance identity is never shared
            rec.pop("name", None)  # display name is per-instance, not part of the template
            self._definitions[def_id]["members"][local] = rec
            for sib in self._shared_siblings(inst_id):
                sib_disp = f"{sib}{SUBPATCH_SEP}{local}"
                sib_uid = self._add_node_from_record(
                    sib_disp, dict(rec), allow_reserved=True, notify_gui=False,
                    membership={"instance": sib, "local_name": local},
                )
                self._membership[sib_uid] = sib
                self._instances[sib]["members"][sib_uid] = local

        if self._bridge is not None and notify_gui:
            self._bridge.control.on_subpatch_changed()
        return uid

    @mark_unsaved_changes
    def expand_instance(self, inst_id: str, notify_gui: bool = True) -> List[str]:
        """Dissolve a sub-patch: restore members' bare DISPLAY names, drop state.
        Returns the member UIDs (now top-level)."""
        if inst_id not in self._instances:
            raise KeyError(f"No such sub-patch: {inst_id}")
        inst = self._instances[inst_id]
        restored: List[str] = []  # uids
        rename_map: Dict[str, str] = {}  # old display -> bare display (for nd())
        for uid, local in list(inst["members"].items()):
            ref = self.nodes[uid]
            old_disp = ref.name
            # Keep bare display names distinct for nd() readability (not required —
            # uid is the key — but avoids surprising shadowing on expand).
            existing = {self.nodes[u].name for u in self.nodes if u != uid}
            target = local
            if target in existing:
                base, idx = local, 0
                while f"{base}{idx}" in existing:
                    idx += 1
                target = f"{base}{idx}"
            ref.name = target
            self._membership.pop(uid, None)
            ref.membership = None
            restored.append(uid)
            rename_map[old_disp] = target
        # Reverse the grouping rewrite: qualified nd('inst::name') -> bare nd('name').
        self._rewrite_member_expressions(restored, rename_map)
        self._broadcast_node_directory()
        del self._instances[inst_id]
        if self._bridge is not None and notify_gui:
            self._bridge.control.on_subpatch_changed()
        return restored

    # `ungroup` reads better at call sites that just want the group dissolved.
    ungroup = expand_instance

    @mark_unsaved_changes
    def remove_instance(self, inst_id: str, notify_gui: bool = True) -> None:
        """Delete a whole sub-patch: its member nodes (and their links) and the
        instance record. A virtual sub-patch node responds to Delete like any
        node. GCs an orphaned shared definition, like `make_unique`."""
        if inst_id not in self._instances:
            raise KeyError(f"No such sub-patch: {inst_id}")
        inst = self._instances[inst_id]
        def_id = inst.get("def_id")
        for member in list(inst["members"].keys()):
            # Pop membership BEFORE remove_node so its boundary defensive-unwire
            # (which keys on membership) no-ops during this teardown.
            self._membership.pop(member, None)
            self.remove_node(member, notify_gui=False)
        del self._instances[inst_id]
        if def_id and not any(i.get("def_id") == def_id for i in self._instances.values()):
            self._definitions.pop(def_id, None)
        if self._bridge is not None and notify_gui:
            self._bridge.control.on_subpatch_changed()

    # ------------------------------------------------------------------
    # Shared sub-patches (strict mirror)
    # ------------------------------------------------------------------

    def _fresh_def_id(self) -> str:
        idx = 0
        while True:
            cand = f"def{idx}"
            if cand not in self._definitions:
                return cand
            idx += 1

    def _definition_from_instance(self, inst_id: str) -> Dict[str, Any]:
        """Snapshot a (unique) instance's topology+params as a reusable definition."""
        inst = self._instances[inst_id]
        members = {local: self._node_record(uid) for uid, local in inst["members"].items()}
        # Strip per-instance identity (uid + display name) from the shared definition
        # records — the template is keyed by local name.
        for rec in members.values():
            rec.pop("uid", None)
            rec.pop("name", None)
        links = []
        for link in self._links:
            if self._membership.get(link["node_out"]) == inst_id and self._membership.get(link["node_in"]) == inst_id:
                links.append(self._local_link(link, inst_id))
        # Deep-copy the interface: entries are mutable port dicts that boundary
        # edits rewrite, and the def must not alias the source instance's dicts.
        return {"members": members, "links": links, "interface": deepcopy(inst["interface"])}

    @mark_unsaved_changes
    def share_instance(self, inst_id: str, notify_gui: bool = True) -> str:
        """Promote a unique instance to a shared one backed by a new definition.

        Returns the definition id. Other instances can then be spawned from it
        with `instantiate_definition`; param edits mirror across all of them.
        """
        if inst_id not in self._instances:
            raise KeyError(f"No such sub-patch: {inst_id}")
        inst = self._instances[inst_id]
        if inst.get("def_id"):
            return inst["def_id"]
        def_id = self._fresh_def_id()
        self._definitions[def_id] = self._definition_from_instance(inst_id)
        inst["kind"] = "shared"
        inst["def_id"] = def_id
        if self._bridge is not None and notify_gui:
            self._bridge.control.on_subpatch_changed()
        return def_id

    @mark_unsaved_changes
    def instantiate_definition(self, def_id: str, pos=(0, 0), notify_gui: bool = True) -> str:
        """Spawn a fresh shared instance of a definition (strict-mirror sibling)."""
        if def_id not in self._definitions:
            raise KeyError(f"No such definition: {def_id}")
        d = self._definitions[def_id]
        inst_id = self._fresh_instance_id()
        members: Dict[str, str] = {}  # uid -> local
        local_to_uid: Dict[str, str] = {}
        # Atomic: a failure mid-splice tears down spawned members + restores the maps.
        with self._transaction():
            for local, rec in d["members"].items():
                new_name = f"{inst_id}{SUBPATCH_SEP}{local}"
                if not self._service_budget_ok(new_name):
                    raise SubPatchTooDeep(f"instance name {new_name!r} overflows the service-name budget")
                uid = self._add_node_from_record(
                    new_name, dict(rec), allow_reserved=True,
                    membership={"instance": inst_id, "local_name": local},
                )
                members[uid] = local
                local_to_uid[local] = uid
            for link in d["links"]:
                self.add_link(
                    local_to_uid[link["node_out"]],
                    local_to_uid[link["node_in"]],
                    link["slot_out"], link["slot_in"], notify_gui=False,
                )
            self._instances[inst_id] = {
                "kind": "shared",
                "def_id": def_id,
                # Deep-copy so this sibling's boundary edits never cross-mutate the def
                # or other siblings (entries are mutable port dicts).
                "interface": deepcopy(d["interface"]),
                "pos": list(pos),
                "members": members,
            }
            for uid in members:
                self._membership[uid] = inst_id
        if self._bridge is not None and notify_gui:
            self._bridge.control.on_subpatch_changed()
        return inst_id

    @mark_unsaved_changes
    def make_unique(self, inst_id: str, notify_gui: bool = True) -> None:
        """Detach a shared instance into a private (unique) copy; GC an orphan def."""
        if inst_id not in self._instances:
            raise KeyError(f"No such sub-patch: {inst_id}")
        inst = self._instances[inst_id]
        def_id = inst.get("def_id")
        inst["kind"] = "unique"
        inst["def_id"] = None
        # Detach the interface from the def/siblings so later boundary edits on this
        # now-unique instance can't cross-mutate the family it just left.
        inst["interface"] = deepcopy(inst["interface"])
        if def_id and not any(i.get("def_id") == def_id for i in self._instances.values()):
            self._definitions.pop(def_id, None)
        if self._bridge is not None and notify_gui:
            self._bridge.control.on_subpatch_changed()

    # ------------------------------------------------------------------
    # In/Out boundary authoring (virtual nodes — interface entries only)
    # ------------------------------------------------------------------

    def _member_uid(self, inst_id: str, local: str) -> Optional[str]:
        """The live uid of the member with local name `local` in `inst_id`, or None.
        Members map uid -> local, so this is the reverse lookup. The uid is the key
        links / membership / the data route use, never the qualified display name."""
        for uid, l in self._instances[inst_id]["members"].items():
            if l == local:
                return uid
        return None

    def _fresh_boundary_id(self, inst_id: str, dir: str) -> str:
        """Lowest unused `in0`/`out0`… among the instance's current interface keys."""
        iface = self._instances[inst_id]["interface"]
        idx = 0
        while f"{dir}{idx}" in iface:
            idx += 1
        return f"{dir}{idx}"

    def _boundary_external_links(self, inst_id: str, dir: str, local: str, slot: str) -> List[dict]:
        """This instance's external flat links for the boundary mapping (local, slot):
        the member-side endpoint matches and the other end is NOT a member of this
        instance (so nested sibling-instance members count as external — correct)."""
        uid = self._member_uid(inst_id, local)
        out: List[dict] = []
        for link in self._links:
            if dir == "in" and link["node_in"] == uid and link["slot_in"] == slot:
                if self._membership.get(link["node_out"]) != inst_id:
                    out.append(link)
            elif dir == "out" and link["node_out"] == uid and link["slot_out"] == slot:
                if self._membership.get(link["node_in"]) != inst_id:
                    out.append(link)
        return out

    def _unsplice_instance(self, inst_id: str, dir: str, local: str, slot: str, notify_gui: bool) -> None:
        for link in self._boundary_external_links(inst_id, dir, local, slot):
            self.remove_link(
                link["node_out"], link["node_in"], link["slot_out"], link["slot_in"], notify_gui=notify_gui
            )

    def _resplice_instance(self, inst_id, dir, old_local, old_slot, new_local, new_slot, notify_gui) -> None:
        new_uid = self._member_uid(inst_id, new_local)
        for link in self._boundary_external_links(inst_id, dir, old_local, old_slot):
            self.remove_link(
                link["node_out"], link["node_in"], link["slot_out"], link["slot_in"], notify_gui=notify_gui
            )
            if dir == "in":
                self.add_link(link["node_out"], new_uid, link["slot_out"], new_slot, notify_gui=notify_gui)
            else:
                self.add_link(new_uid, link["node_in"], new_slot, link["slot_in"], notify_gui=notify_gui)

    def _shared_siblings(self, inst_id: str) -> List[str]:
        def_id = self._instances[inst_id].get("def_id")
        if not def_id:
            return []
        return [i for i, inst in self._instances.items() if i != inst_id and inst.get("def_id") == def_id]

    def _mirror_boundary_entry(self, inst_id: str, bnd_id: str, entry: Optional[dict]) -> None:
        """Mirror a boundary's TOPOLOGY (dir/dtype/inner/pos) to the definition and
        every shared sibling (entry=None removes it). External wires stay per-instance."""
        def_id = self._instances[inst_id].get("def_id")
        if not def_id:
            return
        if entry is None:
            self._definitions[def_id]["interface"].pop(bnd_id, None)
        else:
            self._definitions[def_id]["interface"][bnd_id] = deepcopy(entry)
        for sib in self._shared_siblings(inst_id):
            if entry is None:
                self._instances[sib]["interface"].pop(bnd_id, None)
            else:
                self._instances[sib]["interface"][bnd_id] = deepcopy(entry)

    @mark_unsaved_changes
    def add_boundary(self, inst_id: str, dir: str, dtype: str, pos=(0, 0), notify_gui: bool = True) -> str:
        """Add a virtual In/Out node to a sub-patch (unwired). Returns its boundary id."""
        if inst_id not in self._instances:
            raise KeyError(f"No such sub-patch: {inst_id}")
        if dir not in ("in", "out"):
            raise ValueError(f"dir must be in/out, got {dir!r}")
        bnd_id = self._fresh_boundary_id(inst_id, dir)
        entry = {"dir": dir, "dtype": dtype, "inner_node": None, "inner_slot": None, "pos": list(pos)}
        self._instances[inst_id]["interface"][bnd_id] = entry
        self._mirror_boundary_entry(inst_id, bnd_id, entry)
        if self._bridge is not None and notify_gui:
            self._bridge.control.on_subpatch_changed()
        return bnd_id

    @mark_unsaved_changes
    def wire_boundary(self, inst_id, bnd_id, inner_node, inner_slot, notify_gui: bool = True) -> None:
        """Set (or clear, with inner_node=None) a boundary's single inner target.

        Wiring exposes the port on the collapsed node; unwiring tears down the
        boundary's external wires. Enforces single-target, a dtype match, and one
        boundary per inner slot. For shared instances the inner mapping mirrors to
        every sibling, and each sibling re-splices its OWN external links.
        """
        inst = self._instances[inst_id]
        iface = inst["interface"]
        if bnd_id not in iface:
            raise KeyError(f"No such boundary {bnd_id} on {inst_id}")
        entry = iface[bnd_id]
        dir = entry["dir"]
        old_local, old_slot = entry["inner_node"], entry["inner_slot"]
        siblings = self._shared_siblings(inst_id)

        if inner_node is None:
            if old_local is not None:
                for iid in [inst_id, *siblings]:
                    self._unsplice_instance(iid, dir, old_local, old_slot, notify_gui)
            new_entry = {**entry, "inner_node": None, "inner_slot": None}
        else:
            uid = self._member_uid(inst_id, inner_node)
            if uid is None or uid not in self.nodes or self._membership.get(uid) != inst_id:
                raise ValueError(f"{inner_node} is not a member of {inst_id}")
            slots = self.nodes[uid].input_slots if dir == "in" else self.nodes[uid].output_slots
            dt = slots.get(inner_slot)
            if dt is None:
                raise ValueError(f"no {dir} slot {inner_slot!r} on {inner_node}")
            # `dtype` is absent on legacy (pre-dtype) entries — tolerate it and heal
            # below by storing the real slot dtype.
            expected = entry.get("dtype")
            if expected is not None and dt.name != expected:
                raise ValueError(
                    f"dtype mismatch: {inner_node}.{inner_slot} is {dt.name}, boundary is {expected}"
                )
            for k, e in iface.items():
                if k != bnd_id and e["dir"] == dir and e["inner_node"] == inner_node and e["inner_slot"] == inner_slot:
                    raise ValueError(f"inner slot {inner_node}.{inner_slot} already exposed by {k}")
            # An In port must own its inner input alone: refuse an inner input slot
            # already fed by an INTERNAL member→member link, else the external splice
            # would silently evict that internal wire (add_link is single-source).
            if dir == "in":
                for link in self._links:
                    if (
                        link["node_in"] == uid
                        and link["slot_in"] == inner_slot
                        and self._membership.get(link["node_out"]) == inst_id
                    ):
                        raise ValueError(
                            f"{inner_node}.{inner_slot} is already fed inside the sub-patch; "
                            f"an In node can't expose an already-connected input"
                        )
            if old_local is not None and (old_local, old_slot) != (inner_node, inner_slot):
                for iid in [inst_id, *siblings]:
                    self._resplice_instance(iid, dir, old_local, old_slot, inner_node, inner_slot, notify_gui)
            new_entry = {**entry, "dtype": dt.name, "inner_node": inner_node, "inner_slot": inner_slot}

        iface[bnd_id] = new_entry
        self._mirror_boundary_entry(inst_id, bnd_id, new_entry)
        if self._bridge is not None and notify_gui:
            self._bridge.control.on_subpatch_changed()

    @mark_unsaved_changes
    def remove_boundary(self, inst_id: str, bnd_id: str, notify_gui: bool = True) -> None:
        """Delete an In/Out node, tearing down its external wires across siblings."""
        inst = self._instances[inst_id]
        iface = inst["interface"]
        if bnd_id not in iface:
            raise KeyError(f"No such boundary {bnd_id} on {inst_id}")
        entry = iface[bnd_id]
        if entry["inner_node"] is not None:
            for iid in [inst_id, *self._shared_siblings(inst_id)]:
                self._unsplice_instance(iid, entry["dir"], entry["inner_node"], entry["inner_slot"], notify_gui)
        del iface[bnd_id]
        self._mirror_boundary_entry(inst_id, bnd_id, None)
        if self._bridge is not None and notify_gui:
            self._bridge.control.on_subpatch_changed()

    @mark_unsaved_changes
    def set_boundary_pos(self, inst_id: str, bnd_id: str, pos) -> List[tuple]:
        """Move an In/Out pill, mirroring the pos across shared siblings (like member
        pos). Returns the (inst_id, bnd_id) pairs changed so the bridge can broadcast."""
        pos = list(pos)
        iface = self._instances[inst_id]["interface"]
        if bnd_id not in iface:
            raise KeyError(f"No such boundary {bnd_id} on {inst_id}")
        changed = [(inst_id, bnd_id)]
        iface[bnd_id] = {**iface[bnd_id], "pos": pos}
        def_id = self._instances[inst_id].get("def_id")
        if def_id:
            if bnd_id in self._definitions[def_id]["interface"]:
                self._definitions[def_id]["interface"][bnd_id]["pos"] = pos
            for sib in self._shared_siblings(inst_id):
                if bnd_id in self._instances[sib]["interface"]:
                    self._instances[sib]["interface"][bnd_id] = {**self._instances[sib]["interface"][bnd_id], "pos": pos}
                    changed.append((sib, bnd_id))
        return changed

    def resolve_boundary(self, inst_id: str, bnd_id: str) -> tuple:
        """Translate a (sub-patch, boundary) port to the real inner (member uid,
        slot) for the external-wire splice. Raises if unwired/unknown."""
        inst = self._instances.get(inst_id)
        if inst is None or bnd_id not in inst["interface"]:
            raise KeyError(f"No such boundary {inst_id}:{bnd_id}")
        entry = inst["interface"][bnd_id]
        if entry["inner_node"] is None:
            raise ValueError(f"boundary {inst_id}:{bnd_id} is not wired yet")
        uid = self._member_uid(inst_id, entry["inner_node"])
        if uid is None:
            raise ValueError(f"boundary {inst_id}:{bnd_id} inner member is gone")
        return uid, entry["inner_slot"]

    @mark_unsaved_changes
    def update_param(self, node: str, group: str, name: str, value: Any) -> None:
        """Update a node param, mirroring across siblings if it's a shared member.

        Strict mirror: editing a shared sub-patch member updates the definition
        and every sibling instance's corresponding member in lockstep.
        """
        self.nodes[node].update_param(group, name, value)

        inst_id = self._membership.get(node)
        if not inst_id:
            return
        inst = self._instances.get(inst_id) or {}
        def_id = inst.get("def_id")
        if not def_id:
            return
        local = inst["members"][node]
        # Update the definition's stored value (the save source of truth).
        rec = self._definitions[def_id]["members"].get(local)
        if rec is not None:
            rec.setdefault("params", {}).setdefault(group, {})[name] = value
        # Propagate to every sibling instance's corresponding member.
        for other_id, other in self._instances.items():
            if other_id == inst_id or other.get("def_id") != def_id:
                continue
            for onode, olocal in other["members"].items():
                if olocal == local:
                    try:
                        self.nodes[onode].update_param(group, name, value)
                    except Exception as exc:
                        # Surface, don't swallow: a sibling that fails to mirror would
                        # silently drift from the family (and a later save would persist
                        # whichever sibling is live). Report it so the divergence is visible.
                        self._surface_mirror_failure(onode, f"{group}.{name}", exc)

    @mark_unsaved_changes
    def set_node_pos(self, name: str, pos) -> List[str]:
        """Set a node's editor position, mirroring across shared siblings.

        Strict mirror of layout: moving a member of a *shared* sub-patch updates
        the definition's stored member position and every sibling instance's
        corresponding member in lockstep (same as `update_param` does for values).
        Returns every node name whose position changed, so the bridge can emit a
        `node_moved` for each.
        """
        pos = list(pos)
        changed = [name]
        ref = self.nodes[name]
        ref.gui_kwargs = {**(ref.gui_kwargs or {}), "pos": pos}

        inst_id = self._membership.get(name)
        if not inst_id:
            return changed
        inst = self._instances.get(inst_id) or {}
        def_id = inst.get("def_id")
        if not def_id:
            return changed
        local = inst["members"][name]
        # Record on the definition (the save source of truth). Assign a fresh dict
        # rather than mutating in place — `_node_record` can leave the def's member
        # gui_kwargs aliased to a live node's dict, and we must not touch that.
        rec = self._definitions[def_id]["members"].get(local)
        if rec is not None:
            rec["gui_kwargs"] = {**(rec.get("gui_kwargs") or {}), "pos": pos}
        # Propagate to every sibling instance's corresponding member. Tolerate a
        # stale members entry whose node was already removed (remove_node leaves
        # _membership/members untouched) — same defensive stance as update_param.
        for other_id, other in self._instances.items():
            if other_id == inst_id or other.get("def_id") != def_id:
                continue
            for onode, olocal in other["members"].items():
                if olocal == local and onode in self.nodes:
                    oref = self.nodes[onode]
                    oref.gui_kwargs = {**(oref.gui_kwargs or {}), "pos": pos}
                    changed.append(onode)
        return changed

    def _node_record(self, uid: str) -> Dict[str, Any]:
        ref = self.nodes[uid]
        if ref.serialized_state is None:
            raise RuntimeError(f"Node {uid} does not have a serialized state. Recreate the node and try again.")
        state = deepcopy(ref.serialized_state)
        state["gui_kwargs"] = ref.gui_kwargs
        state.pop("output_subscribers", None)
        if ref.uid is not None:
            state["uid"] = ref.uid
        if ref.name is not None:
            state["name"] = ref.name
        return state

    def _local_link(self, link: Dict[str, str], inst_id: str) -> Dict[str, str]:
        m = self._instances[inst_id]["members"]
        return {
            "node_out": m[link["node_out"]],
            "node_in": m[link["node_in"]],
            "slot_out": link["slot_out"],
            "slot_in": link["slot_in"],
        }

    def build_v2_tree(self):
        """Collapse the live flat graph into (root_nodes, root_links, definitions,
        instances) for the v2 envelope, reading first-class membership state.

        UNIQUE instances inline their members/links/interface. SHARED instances
        reference a definition (emitted once) and carry only per-member uid+pos.
        """
        # root_nodes / links / instance members are all keyed by uid; the readable
        # display name rides inside each node record (see _node_record).
        member_set = set(self._membership)
        root_nodes = {uid: self._node_record(uid) for uid in self.nodes if uid not in member_set}

        internal: Dict[str, list] = {iid: [] for iid in self._instances}
        root_links: list = []
        for link in self._links:
            oi = self._membership.get(link["node_out"])
            ii = self._membership.get(link["node_in"])
            if oi is not None and oi == ii:
                internal[oi].append(self._local_link(link, oi))
            else:
                root_links.append(dict(link))

        instances: Dict[str, Any] = {}
        definitions: Dict[str, Any] = {}
        for iid, inst in self._instances.items():
            if inst.get("def_id"):
                def_id = inst["def_id"]
                if def_id not in definitions:
                    definitions[def_id] = deepcopy(self._definitions[def_id])
                instances[iid] = {
                    "kind": "shared",
                    "def": def_id,
                    "pos": inst["pos"],
                    # Only per-instance state — topology+params live in the definition.
                    "members": {
                        local: {
                            "uid": self.nodes[nn].uid,
                            "pos": (self.nodes[nn].gui_kwargs or {}).get("pos"),
                        }
                        for nn, local in inst["members"].items()
                    },
                }
            else:
                members = {inst["members"][nn]: self._node_record(nn) for nn in inst["members"]}
                instances[iid] = {
                    "kind": "unique",
                    "pos": inst["pos"],
                    "interface": inst["interface"],
                    "members": members,
                    "links": internal.get(iid, []),
                }
            # Per-instance viewer state (collapsed sub-patch slots) rides on the record
            # so a reload keeps the kind/settings the user chose (backlog #17).
            if inst.get("viewers"):
                instances[iid]["viewers"] = deepcopy(inst["viewers"])
        return root_nodes, root_links, definitions, instances

    def _add_node_from_record(
        self, name: str, node: Dict[str, Any], allow_reserved: bool = False, membership=None,
        notify_gui: bool = True,
    ) -> str:
        gk = node.get("gui_kwargs") or {}
        pos = gk.get("pos")
        if pos is not None:
            xpos, ypos = pos
            if xpos == np.iinfo(np.int32).min or ypos == np.iinfo(np.int32).min:
                print(f"WARNING: Node '{name}' has a corrupted position. Resetting to (0, 0).")
                gk["pos"] = (0, 0)
        return self.add_node(
            node["_type"],
            node["category"],
            notify_gui=notify_gui,
            name=name,
            params=node["params"],
            member_uid=node.get("uid"),
            membership=membership if membership is not None else node.get("membership"),
            allow_reserved=allow_reserved,
            **gk,
        )

    def _expand_doc(self, root_nodes, root_links, instances, definitions=None) -> None:
        """Splice a v2 document's root graph + sub-patch instances into the live
        flat graph. Add all nodes first, then all links (so add_link never races
        a not-yet-spawned endpoint). Handles both unique (inline) and shared
        (definition-backed) instances."""
        self._definitions.update(definitions or {})

        for key, node in root_nodes.items():
            # v2 records carry their display name; v1 used the dict key as the name.
            self._add_node_from_record(node.get("name", key), node)

        # (inst_id, local_link, local_to_uid): resolve template-local link endpoints
        # to the freshly-minted member uids once all members of the instance exist.
        internal_links: List[tuple] = []
        for inst_id, inst in instances.items():
            kind = inst.get("kind", "unique")
            members_map: Dict[str, str] = {}  # uid -> local
            local_to_uid: Dict[str, str] = {}
            if kind == "shared":
                def_id = inst["def"]
                d = self._definitions[def_id]
                per = inst.get("members") or {}
                for local, rec in d["members"].items():
                    new_name = f"{inst_id}{SUBPATCH_SEP}{local}"
                    pm = per.get(local) or {}
                    node_rec = dict(rec)
                    if pm.get("uid"):
                        node_rec["uid"] = pm["uid"]
                    if pm.get("pos") is not None:
                        node_rec["gui_kwargs"] = {**(rec.get("gui_kwargs") or {}), "pos": pm["pos"]}
                    uid = self._add_node_from_record(
                        new_name, node_rec, allow_reserved=True,
                        membership={"instance": inst_id, "local_name": local},
                    )
                    members_map[uid] = local
                    local_to_uid[local] = uid
                # Deep-copy: each loaded shared instance gets its own port dicts so
                # later boundary edits don't alias the definition / siblings.
                interface = deepcopy(d["interface"])
                for link in d["links"]:
                    internal_links.append((local_to_uid, link))
            else:
                for local, rec in (inst.get("members") or {}).items():
                    new_name = f"{inst_id}{SUBPATCH_SEP}{local}"
                    uid = self._add_node_from_record(
                        new_name, rec, allow_reserved=True,
                        membership={"instance": inst_id, "local_name": local},
                    )
                    members_map[uid] = local
                    local_to_uid[local] = uid
                interface = inst.get("interface", {})
                for link in inst.get("links", []):
                    internal_links.append((local_to_uid, link))

            self._instances[inst_id] = {
                "kind": kind,
                "def_id": inst.get("def") if kind == "shared" else None,
                "interface": interface,
                "pos": inst.get("pos", [0, 0]),
                "members": members_map,
            }
            if inst.get("viewers"):
                self._instances[inst_id]["viewers"] = inst["viewers"]
            for uid in members_map:
                self._membership[uid] = inst_id

        for local_to_uid, link in internal_links:
            self.add_link(
                local_to_uid[link["node_out"]],
                local_to_uid[link["node_in"]],
                link["slot_out"], link["slot_in"],
            )
        for link in root_links:
            self.add_link(link["node_out"], link["node_in"], link["slot_out"], link["slot_in"])

    def _node_directory(self) -> Dict[str, str]:
        """Map each live DISPLAY name to its node's stable transport id, for
        `nd('name')` resolution. Names aren't guaranteed unique anymore (the graph
        keys on uid); on a collision the last node wins — an acceptable edge case
        for an expression-convenience lookup."""
        return {self.nodes[uid].name: self.nodes[uid].node_id for uid in self.nodes}

    def _broadcast_node_directory(self) -> None:
        """Push the current name->id directory to every live node so their
        expression engines resolve `nd('name')` to the producing node's id."""
        directory = self._node_directory()
        for name in list(self.nodes):
            try:
                self.nodes[name].send_directory(directory)
            except Exception:
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

        from goofi.patch_format import normalize_loaded, read_graph

        with open(filepath, "r") as f:
            raw = yaml.load(f, Loader=yaml.FullLoader)

        root_nodes, root_links, instances, defs, layout = read_graph(normalize_loaded(raw))
        self._expand_doc(root_nodes, root_links, instances, defs)

        # Restore the frontend layout if the patch carries one (optional key —
        # older patches without it leave layout None, and the browser falls
        # back to its localStorage / default layout). Broadcast it: the bridge
        # is up before this initial load completes, so a client that connected
        # during load got a layout-less `hello` and needs this to catch up
        # (nodes already stream via node_added events; layout has no other
        # delivery path). The frontend ignores a null payload.
        self._layout = layout
        if self._bridge is not None:
            self._bridge.control.broadcast_threadsafe(
                {"event": "layout", "payload": {"layout": self._layout}}
            )

        self.save_path = filepath
        self.unsaved_changes = False
        print("Finished loading manager state.")

    def serialize_patch(self, timeout: float = 3.0) -> str:
        """Serialize the current graph to `.gfi` YAML text, without writing.

        Reads each node's pushed `serialized_state` directly (waiting briefly
        per node if it hasn't pushed yet) and merges its gui_kwargs. Shared by
        `save()` (writes to disk) and the bridge `serialize` op ("Save in
        browser" download).
        """
        # Snapshot the names first: a patch may still be spawning nodes on
        # another thread, and iterating the live container would raise
        # "dictionary changed size during iteration" (matches terminate()).
        for name in list(self.nodes):
            ref = self.nodes[name]
            ref.wait_for_state(timeout=timeout)
            if ref.serialized_state is None:
                raise RuntimeError(f"Node {name} does not have a serialized state. Recreate the node and try again.")

        # Collapse the flat live graph into the recursive v2 envelope, reading
        # first-class sub-patch membership (not name prefixes).
        from goofi.patch_format import build_v2

        root_nodes, root_links, definitions, instances = self.build_v2_tree()
        doc = build_v2(
            nodes=root_nodes,
            links=root_links,
            layout=self.layout,
            definitions=definitions,
            instances=instances,
        )
        return yaml.dump(doc, sort_keys=False)

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
        manager_yaml = self.serialize_patch(timeout=timeout)

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
        # Stop the liveness supervisor first so it can't race a teardown by
        # "restarting" nodes we're about to terminate.
        self._stop_supervisor()
        NodeProcessRegistry().terminate()
        for node in list(self.nodes):
            try:
                self.nodes[node].terminate()
            except Exception:
                pass

        if self._bridge is not None:
            try:
                self._bridge.shutdown()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def save_path(self) -> Optional[str]:
        return self._save_path

    @save_path.setter
    def save_path(self, filepath: str) -> None:
        self._save_path = filepath
        if self._bridge is not None:
            self._bridge.control.broadcast_threadsafe(
                {"event": "save_path_changed", "payload": {"save_path": filepath}}
            )

    @property
    def layout(self) -> Optional[Any]:
        # getattr default keeps managers built via __new__ (test fixtures) safe
        # even though they don't run __init__.
        return getattr(self, "_layout", None)

    @layout.setter
    def layout(self, value: Optional[Any]) -> None:
        self._layout = value

    @property
    def unsaved_changes(self) -> bool:
        return self._unsaved_changes

    @unsaved_changes.setter
    def unsaved_changes(self, value: bool) -> None:
        self._unsaved_changes = value
        if self._bridge is not None:
            self._bridge.control.broadcast_threadsafe(
                {"event": "unsaved_changes", "payload": {"unsaved_changes": value}}
            )

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
    """Reap orphan iceoryx2 nodes via the native reaper.

    iceoryx2 cleans up its own services on graceful Node drop. On crashes
    or SIGKILL the entries linger; `try_cleanup_dead_nodes` finds nodes
    whose owning process is gone and releases their resources — a single
    cross-platform call that works on Linux, macOS, and Windows. We call
    it on Manager startup and again at process exit so successive sessions
    don't accumulate shared-memory entries.
    """
    try:
        import iceoryx2 as iox2

        iox2.Node.try_cleanup_dead_nodes(iox2.ServiceType.Ipc, iox2.config.global_config())
    except Exception:
        # Cleanup is best-effort; never let it fail the startup or exit path.
        pass


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


def build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="goofi-pipe")
    parser.add_argument("filepath", nargs="?", help="path to the file to load from")
    parser.add_argument("--headless", action="store_true", help="run in headless mode (no browser bridge)")
    parser.add_argument("--no-multiprocessing", action="store_true", help="disable multiprocessing")
    parser.add_argument("--port", type=int, default=8000, help="port to serve the browser UI on (default 8000)")
    parser.add_argument(
        "--bind",
        type=str,
        default="127.0.0.1",
        help=(
            "host/interface to bind the bridge to (default: 127.0.0.1 — loopback only). "
            "Pass 0.0.0.0 to expose on the network, but note the bridge has NO auth and "
            "expression nodes execute arbitrary Python."
        ),
    )
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
    return parser


def main(duration: Optional[float] = None, args=None):
    parser = build_arg_parser()
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
        bridge_host=args.bind,
        bridge_port=args.port,
    )


if __name__ == "__main__":
    main()
