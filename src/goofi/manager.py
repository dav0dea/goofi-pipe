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
from dataclasses import asdict, dataclass, field, replace
from os import path
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import yaml

from goofi.expression import rewrite_nd_refs
from goofi.message import MessageType
from goofi.node import MultiprocessingForbiddenError, Node
from goofi.node_helpers import NodeProcessRegistry, NodeRef, list_nodes
from goofi.transport import ensure_iox2_runtime_dirs, set_instance_id

if TYPE_CHECKING:
    from goofi.bridge.server import BridgeServer


logger = logging.getLogger(__name__)

# Marker prefixing a shared DEFINITION's INTERNAL (intra-sub-patch) nd() member refs,
# so the instantiate-time local->display rewrite can never be hijacked by an EXTERNAL
# ref whose name happens to equal a member's local template key. A definition stores
# internal refs as `nd('\x1f<local>')` and external refs verbatim as `nd('<display>')`;
# the two namespaces can't collide because the marker is a control char no display name
# can contain (untypeable, never minted) yet is still valid in a Python expression
# source (unlike NUL, which ast.parse rejects) and round-trips through .gfi YAML.
_DEF_INTERNAL_REF = "\x1f"

# The reserved id of the ROOT scope — the top-level graph materialized as a first-class
# SubPatchInstance(parent=None) so a top-level node is a member of ROOT exactly like a
# sub-patch member (one add/remove/membership path; root = the scope with no parent).
# Can never collide with a minted entity uid (always 12 hex chars) or a display name
# (subpatchN / node-type names). Mirrored on the frontend (must match) — see
# lib/editor/subpatchScene.ts.
ROOT_ID = "__root__"


class SubPatchTooDeep(ValueError):
    """A namespaced member name would overflow iceoryx2's 255-byte ServiceName."""


@dataclass
class Boundary:
    """One In/Out port on a sub-patch's interface. `inner_node` is the LOCAL name of
    the member the port maps to (Phase 3: or a nested instance); `inner_slot` its
    slot. Unwired ports carry inner_node=None. Identical in shape whether auto-derived
    from a crossing link or authored as a virtual In/Out node."""

    dir: str  # 'in' | 'out'
    dtype: Optional[str]
    inner_node: Optional[str]
    inner_slot: Optional[str]
    pos: List[float]


@dataclass
class SubPatchInstance:
    """Runtime grouping record for one sub-patch instance. The live graph stays flat;
    this is the first-class record of the grouping (never re-derived from name
    prefixes). `members` maps member uid -> local name — the uid is the key that
    links / membership / the data route use, the local name is the per-template
    handle keyed into the definition."""

    uid: str  # stable instance identity (also the _instances key)
    name: str  # display label, e.g. "subpatch0"
    kind: str  # 'unique' | 'shared'
    def_id: Optional[str]  # set iff kind == 'shared'
    members: Dict[str, str]  # member uid -> local name
    interface: Dict[str, Boundary]  # boundary id -> port
    pos: List[float]
    viewers: Dict[str, Any] = field(default_factory=dict)
    parent: Optional[str] = None  # parent instance uid (nesting); None at single level


@dataclass
class SubPatchDef:
    """A reusable shared-sub-patch definition that every instance strict-mirrors:
    template members (local name -> node record), internal links (local-name
    endpoints), the boundary interface, and nested-instance members by reference.

    `instances` maps a nested-instance member's local -> {"def": child_def_id, "pos"}:
    a definition is identity-free and shared by every instance, so it can only name a
    nested sub-patch as a TEMPLATE (another def_id), never as an inline unique child —
    this is the "independent nested defs" model (the nested sub-patch is its own
    first-class definition; editing it propagates to every instance everywhere)."""

    members: Dict[str, Dict[str, Any]]  # local name -> node record
    links: List[Dict[str, str]]
    interface: Dict[str, Boundary]
    instances: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # local -> {def, pos}


def _iface_to_dict(interface: Dict[str, Boundary]) -> Dict[str, Dict[str, Any]]:
    """Serialize an interface (boundary id -> Boundary) to plain dicts for the JSON
    snapshot and the saved .gfi document."""
    return {bid: asdict(b) for bid, b in interface.items()}


def _rewrite_record_nd(rec: Dict[str, Any], name_map: Dict[str, str]) -> None:
    """Rewrite nd() refs in a serialized node record's stashed param expressions
    (the {value, expression, ...} form) per `name_map`, in place. Used to translate a
    member record between live display names and a definition's template locals."""
    for group in (rec.get("params") or {}).values():
        if not isinstance(group, dict):
            continue
        for pval in group.values():
            if isinstance(pval, dict) and pval.get("expression"):
                pval["expression"] = rewrite_nd_refs(pval["expression"], name_map)


def _iface_from_dict(raw: Dict[str, Dict[str, Any]]) -> Dict[str, Boundary]:
    """Build runtime Boundary ports from plain dicts (.gfi load / wire payloads),
    tolerating legacy pre-dtype entries (dtype defaults to None)."""
    return {
        bid: Boundary(
            dir=e["dir"],
            dtype=e.get("dtype"),
            inner_node=e.get("inner_node"),
            inner_slot=e.get("inner_slot"),
            pos=list(e.get("pos") or [0, 0]),
        )
        for bid, e in raw.items()
    }


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
        # maps are the first-class, typed record of grouping (NOT re-derived from
        # name prefixes). `_membership` maps a member's uid -> instance id (the
        # reverse index of every instance's `members`); `_instances` holds the
        # SubPatchInstance records; `_definitions` holds the shared SubPatchDef
        # templates (populated in the sharing phase). The three views (an
        # instance's `members`, `_membership`, and each node's `.membership`
        # marker) are kept in lockstep by `_attach_member` / `_detach_member`.
        self._membership: Dict[str, str] = {}
        self._instances: Dict[str, SubPatchInstance] = {}
        self._definitions: Dict[str, SubPatchDef] = {}
        self._install_root_scope()

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
        """A fresh entity uid not currently in use (48 bits + dedup pass). Spans BOTH
        nodes and sub-patch instances — instances are first-class, uid-keyed entities,
        so the two never collide."""
        while True:
            uid = uuid.uuid4().hex[:12]
            if uid not in self.nodes and uid not in self._instances:
                return uid

    def _display_names_in_use(self, exclude_uid: Optional[str] = None) -> set:
        """Every display name currently taken — nodes AND sub-patch instances. Both are
        first-class entities rendered in one flat canvas, and nd() resolves by the bare
        name, so they share ONE global namespace. Any fresh/unique-name allocation must
        consult both or a node could shadow an instance label (or vice versa)."""
        return {self.nodes[u].name for u in self.nodes if u != exclude_uid} | {
            i.name for i in self._instances.values() if i.uid != exclude_uid
        }

    def _fresh_display_name(self, base: str) -> str:
        """Lowest free flat display name `base0`, `base1`, … (globally unique). Names
        are flat at every nesting depth — no `inst::local` qualification — so nd()
        resolves by the bare name and grouping never has to rename a member."""
        existing = self._display_names_in_use()
        idx = 0
        while f"{base}{idx}" in existing:
            idx += 1
        return f"{base}{idx}"

    def _restore_member_name(self, saved_name: Optional[str], node_type: str) -> str:
        """Flat display name for a member being loaded: the saved flat name
        (disambiguated against the live graph, e.g. when splicing into a populated
        graph), or a fresh type-based name when absent."""
        return self._unique_display_name(saved_name) if saved_name else self._fresh_display_name(node_type.lower())

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
        scope: str = ROOT_ID,
        **gui_kwargs,
    ) -> str:
        # ONE public add path, parameterized by scope (root = the scope with no parent).
        # Adding into a real sub-patch needs scope-specific orchestration (a template
        # local key + strict-mirror across shared siblings), so dispatch to that core;
        # `membership` is the LOW-LEVEL form used internally (load / sibling-mirror) and
        # bypasses the dispatch. A top-level (ROOT) add falls through to the spawn below.
        if scope != ROOT_ID and membership is None:
            return self.add_member_node(
                scope, node_type, category, name=name, params=params,
                pos=tuple(gui_kwargs.get("pos", (0, 0))), notify_gui=notify_gui,
                member_uid=member_uid,
            )
        print(f"Adding node '{node_type}' from category '{category}'.")
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
            assigned_name = self._fresh_display_name(node_type.lower())
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
        # A top-level node is a member of the ROOT scope (root = the scope with no
        # parent), so membership has ONE funnel. A sub-patch member instead carries an
        # explicit membership dict and is attached by its caller (add_member_node). The
        # root local is the (deduped) display name — same convention as group/expand;
        # it's never serialized (ROOT dissolves at save) and never seen by nd().
        if membership is None:
            self._attach_member(ROOT_ID, member_uid, self._unique_local_in(ROOT_ID, assigned_name))

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
        print(f"Removing node '{self.nodes[uid].name if uid in self.nodes else '?'}' ({uid}).")
        # A node that's still a member of a sub-patch (remove_instance pops
        # membership BEFORE calling this, so teardown skips the whole block):
        # Every node is a member of SOME scope (ROOT for a top-level node), so detach
        # always runs through the one funnel; the event then diverges only on whether
        # the scope was a real sub-patch vs ROOT (a top-level remove). (remove_instance
        # pops membership BEFORE calling this, so its teardown skips the whole block.)
        _inst_id = self._membership.get(uid)
        _membership_payload: Optional[Dict[str, Any]] = None
        unwired_boundary = False
        if _inst_id is not None:
            _inst = self._instances.get(_inst_id)
            if _inst is not None:
                # Deleting a single member of a SHARED sub-patch would desync it from
                # its definition/siblings (topology isn't mirrored) and leave dangling
                # ports — block it; the user must make the instance unique first.
                if _inst.def_id:
                    raise ValueError(
                        f"cannot delete member {uid!r} of a shared sub-patch; make it unique first"
                    )
                # Capture the scope membership for the incremental node_removed BEFORE
                # detaching (so the frontend can drop the uid from the owning scope's
                # members map). Then unwire any boundary pointing at this member (ROOT has
                # none): if it fed ≥1 boundary the instance's computed slots/interface
                # change — a structural edit that resyncs the whole record below.
                _local = _inst.members.get(uid)
                _membership_payload = {"instance": _inst_id, "local_name": _local}
                for _bid, _e in list(_inst.interface.items()):
                    if _e.inner_node == _local:
                        _inst.interface[_bid] = replace(_e, inner_node=None, inner_slot=None)
                        unwired_boundary = True
                self._detach_member(uid)
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
            # Root ≡ unique sub-patch: a member's removal is an incremental node_removed
            # carrying its membership (so the frontend drops it from the owning scope's
            # members map), EXACTLY like a top-level remove — UNLESS it unwired a boundary,
            # which restructures the instance (computed slots/interface). That case resyncs
            # the whole record via subpatch_changed (whose snapshot also does the per-node
            # cleanup on_node_removed did and re-sweeps status wiring).
            if unwired_boundary:
                self._bridge.control.on_subpatch_changed()
            else:
                self._bridge.control.on_node_removed(uid, membership=_membership_payload)

    @mark_unsaved_changes
    def add_link(
        self,
        node_out: str,
        node_in: str,
        slot_out: str,
        slot_in: str,
        notify_gui: bool = True,
        mirror: bool = True,
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
        # graph definition. Carry `mirror` so a sibling-mirror call doesn't
        # re-mirror its own eviction.
        for existing in list(self._links):
            if existing["node_in"] == node_in and existing["slot_in"] == slot_in:
                self.remove_link(
                    existing["node_out"],
                    existing["node_in"],
                    existing["slot_out"],
                    existing["slot_in"],
                    notify_gui=notify_gui,
                    mirror=mirror,
                )

        link = {"node_out": node_out, "node_in": node_in, "slot_out": slot_out, "slot_in": slot_in}
        self._wire_link(link)
        self._links.append(link)
        if mirror:
            self._mirror_internal_link(link, add=True, notify_gui=notify_gui)

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
        mirror: bool = True,
        **gui_kwargs,
    ) -> None:
        link = {"node_out": node_out, "node_in": node_in, "slot_out": slot_out, "slot_in": slot_in}
        if link not in self._links:
            return
        self._teardown_link(link, notify_gui=False)
        self._links.remove(link)
        if mirror:
            self._mirror_internal_link(link, add=False, notify_gui=notify_gui)
        if self._bridge is not None and notify_gui:
            self._bridge.control.on_link_removed(link)

    def _mirror_internal_link(self, link: Dict[str, str], add: bool, notify_gui: bool) -> None:
        """Mirror an internal member->member link of a SHARED instance into the
        definition (local-keyed) + every sibling's corresponding members. A no-op
        for top-level links, external boundary wires (endpoints in different
        instances), and unique instances (which inline their links on save).

        Sibling calls pass mirror=False so they don't recurse; def-link writes are
        idempotent so a restart/replay re-wiring an existing link is harmless."""
        inst_id = self._membership.get(link["node_out"])
        if inst_id is None or inst_id != self._membership.get(link["node_in"]):
            return
        inst = self._instances.get(inst_id)
        if inst is None or not inst.def_id:
            return
        def_id = inst.def_id
        members = inst.members
        lout, lin = members.get(link["node_out"]), members.get(link["node_in"])
        if lout is None or lin is None:
            return
        local_link = {"node_out": lout, "node_in": lin, "slot_out": link["slot_out"], "slot_in": link["slot_in"]}
        deflinks = self._definitions[def_id].links
        if add and local_link not in deflinks:
            deflinks.append(local_link)
        elif not add and local_link in deflinks:
            deflinks.remove(local_link)
        for sib in self._shared_siblings(inst_id):
            sout, sin = self._member_uid(sib, lout), self._member_uid(sib, lin)
            if sout is None or sin is None:
                continue
            if add:
                self.add_link(sout, sin, link["slot_out"], link["slot_in"], notify_gui=notify_gui, mirror=False)
            else:
                self.remove_link(sout, sin, link["slot_out"], link["slot_in"], notify_gui=notify_gui, mirror=False)

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
        Display name, uid, sub-patch membership and gui position are kept;
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

    def _unique_display_name(self, requested: str, exclude_uid: Optional[str] = None) -> str:
        """Return `requested` if no OTHER live node holds it, else the smallest
        integer-suffixed variant that's free. Display names must stay unique
        because `nd('name')` resolves a cross-reference by display name — a
        collision would make the lookup ambiguous (the directory is name->id)."""
        taken = self._display_names_in_use(exclude_uid)
        if requested not in taken:
            return requested
        idx = 1
        while f"{requested}{idx}" in taken:
            idx += 1
        return f"{requested}{idx}"

    @mark_unsaved_changes
    def rename_node(self, uid: str, name: str) -> None:
        """Set a node's mutable DISPLAY name — flat and globally unique at every
        nesting depth (no `inst::local` qualification). The graph is uid-keyed so no
        reference *key* moves, but nd() refs resolve by name, so the rename rewrites
        every nd('old') -> nd('new') across the graph in the same call (the caller
        records it as one undo entry). A member's `local` template key is decoupled
        from its display, so renaming a member doesn't disturb its boundaries; a
        shared member can't be renamed yet (its display is per-instance but lifting
        the guard is a separate follow-up)."""
        if uid not in self.nodes:
            raise KeyError(f"No such node: {uid}")
        inst_id = self._membership.get(uid)
        # "Member" here means a real sub-patch member: a top-level (ROOT-scoped) node is
        # renamed like any node (on_node_renamed), not as a structural sub-patch change.
        is_member = inst_id is not None and inst_id != ROOT_ID
        if is_member and self._instances[inst_id].def_id:
            raise ValueError("Renaming a member of a shared sub-patch isn't supported yet.")
        old_name = self.nodes[uid].name
        new_name = self._unique_display_name(name, exclude_uid=uid)
        if new_name == old_name:
            return
        if not self._service_budget_ok(new_name):
            raise SubPatchTooDeep(f"renaming would overflow the service-name budget for {new_name!r}")
        self.nodes[uid].name = new_name
        # Fellow members AND external referrers point at this node by its display name
        # in nd() — rewrite across the whole graph (names are globally unique, so the
        # rewrite is unambiguous and reversed exactly by the inverse rename on undo).
        self._rewrite_member_expressions(list(self.nodes), {old_name: new_name})
        # Shared definitions store EXTERNAL refs verbatim (by display name); a rename of
        # an external producer must follow into the def so freshly-instantiated siblings
        # and the save/load round-trip stay current. Internal refs are stored
        # `\x1f`-prefixed, so this plain {old: new} map can never touch them.
        for d in self._definitions.values():
            for rec in d.members.values():
                _rewrite_record_nd(rec, {old_name: new_name})
        self._broadcast_node_directory()
        if self._bridge is not None:
            if is_member:
                self._bridge.control.on_subpatch_changed()
            else:
                self._bridge.control.on_node_renamed(uid, new_name)

    def _fresh_instance_id(self) -> str:
        """A fresh stable uid for an instance — its universal key, exactly like a
        node's uid (never the reused display label)."""
        return self._mint_uid()

    def _fresh_instance_name(self) -> str:
        """Lowest free `subpatch0`, `subpatch1`, … display label, unique among nodes
        AND instances so a collapsed group node's label never shadows another."""
        existing = self._display_names_in_use()
        idx = 0
        while f"subpatch{idx}" in existing:
            idx += 1
        return f"subpatch{idx}"

    def _restore_instance_name(self, saved_name: Optional[str]) -> str:
        """The saved instance display name if free, else a fresh one (e.g. splicing a
        sub-patch into a graph that already has that label)."""
        if not saved_name:
            return self._fresh_instance_name()
        existing = self._display_names_in_use()
        return saved_name if saved_name not in existing else self._fresh_instance_name()

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
            iface[key] = Boundary(
                dir=dir,
                dtype=self._slot_dtype(disp, slot, dir),
                inner_node=local,
                inner_slot=slot,
                pos=self._beside_member_pos(disp, dir),
            )
        return iface

    def _entity_name(self, uid: str) -> str:
        """Display name of any sub-patch entity — a real node OR a nested instance.
        Both share one globally-unique flat namespace, so this is the single resolver
        for seeding a member's local template key or reporting an entity by name."""
        if uid in self.nodes:
            return self.nodes[uid].name
        return self._instances[uid].name

    def _install_root_scope(self) -> None:
        """Materialize the root graph as a first-class scope: a unique, parent-less,
        boundary-less instance under the reserved id. Every top-level node is its member,
        so add/remove/membership have ONE code path (root = the scope with no parent). It
        owns no process; `self.nodes` (real processes) is untouched. Idempotent."""
        if ROOT_ID not in self._instances:
            self._instances[ROOT_ID] = SubPatchInstance(
                uid=ROOT_ID, name="root", kind="unique", def_id=None,
                members={}, interface={}, pos=[0.0, 0.0],
            )

    def _attach_member(self, inst_id: str, uid: str, local: str) -> None:
        """Record `uid` as a member of `inst_id` under local name `local`, keeping the
        views in lockstep: the instance's `members` map, the uid->instance reverse
        index, and the member's own parent marker. A member is either a real NODE
        (marker on `ref.membership`) or — once nested (Phase 3a) — another sub-patch
        INSTANCE (marker on `SubPatchInstance.parent`); the same funnel maintains both,
        so the parent edge can't drift from the index.

        A MOVE, not just an add: any prior parent is dropped first, so an entity is never
        listed under two scopes (idempotent; a no-op for a fresh entity). Now that every
        entity has a parent — ROOT for a top-level one — callers that re-home an entity no
        longer need an explicit detach-first to avoid a stale old-parent entry."""
        if uid in self._membership:
            self._detach_member(uid)
        self._instances[inst_id].members[uid] = local
        self._membership[uid] = inst_id
        if uid in self.nodes:
            self.nodes[uid].membership = {"instance": inst_id, "local_name": local}
        elif uid in self._instances:
            self._instances[uid].parent = inst_id

    def _detach_member(self, uid: str) -> Optional[str]:
        """Remove `uid` from whatever instance owns it, clearing all views.
        Returns the instance id it was detached from (or None if it had no parent).
        Tolerates a node whose live ref is already gone (teardown order)."""
        inst_id = self._membership.pop(uid, None)
        if inst_id is not None and inst_id in self._instances:
            self._instances[inst_id].members.pop(uid, None)
        if uid in self.nodes:
            self.nodes[uid].membership = None
        elif uid in self._instances:
            self._instances[uid].parent = None
        return inst_id

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
        # The node-side membership MARKER (ref.membership) is a second view of an entity's
        # parent, mutated by _attach_member/_detach_member alongside _membership. It lives
        # on the live NodeRef (NOT inside the deepcopied _instances), so snapshot it too or
        # rollback leaves a surviving node's marker pointing at a rolled-back instance.
        snap_markers = {uid: deepcopy(self.nodes[uid].membership) for uid in self.nodes}
        before_nodes = set(self.nodes)
        try:
            yield
        except Exception:
            # Restore the state maps FIRST, then tear down the orphan nodes: with the
            # maps back to their pre-block state a spawned member is no longer recorded
            # as a shared-sub-patch member, so remove_node's shared-member-delete guard
            # (and any other map-reading guard) won't trip and strand it.
            self._links[:] = snap_links
            self._node_groups.clear()
            self._node_groups.update(snap_groups)
            self._membership.clear()
            self._membership.update(snap_membership)
            self._instances.clear()
            self._instances.update(snap_instances)
            self._definitions.clear()
            self._definitions.update(snap_definitions)
            # Restore the node-side markers for every surviving node (spawned nodes are
            # torn down below, so only pre-block nodes need their marker put back).
            for uid, marker in snap_markers.items():
                if uid in self.nodes:
                    self.nodes[uid].membership = marker
            for n in set(self.nodes) - before_nodes:
                try:
                    self.remove_node(n, notify_gui=False)
                except Exception:
                    pass
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

    def _rewrite_member_expressions(self, node_uids, name_map: Dict[str, str]) -> None:
        """Rewrite string-literal nd() refs across the given live nodes' param
        expressions per `name_map` (old display/local name -> new), preserving the
        enable/trigger/autoeval flags. The single live-ref nd() rewriter — used by
        rename (display->new display) and instantiate (template local->fresh display).
        Pass the whole node set so EXTERNAL referrers are rewritten too. Nodes without
        a live ref/params are skipped."""
        for name in node_uids:
            if name not in self.nodes:
                continue
            ref = self.nodes[name]
            for grp in list(ref.params.keys()):
                for pname, p in ref.params[grp].items():
                    expr = getattr(p, "expression", None)
                    new_expr = rewrite_nd_refs(expr, name_map)
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

        Flat naming: members keep their uids AND their (already globally-unique)
        DISPLAY names — grouping is purely organizational, never a rename, so no
        nd() cross-reference has to be rewritten. Each member's `local` is a
        per-instance template key, seeded here from its display name.
        Membership/interface are recorded as first-class state, keyed by uid.
        Returns the new instance id. `member_names` is a list of member UIDs.
        """
        member_uids = list(member_names)
        if not member_uids:
            raise ValueError("no members to group")
        for u in member_uids:
            # ROOT is a real instance id, so it would pass the existence check below and
            # then get re-homed under the new group (ROOT.parent set) — corrupting the
            # canvas. The root scope can never be a member of a sub-patch.
            self._reject_root(u, "group")
            if u not in self.nodes and u not in self._instances:
                raise KeyError(f"No such entity: {u}")
        # Every member must share ONE parent scope (same nesting level); the new
        # instance is created at that level (None = top-level). A mixed-scope set is
        # ambiguous — there is no single place to put the group.
        parents = {self._membership.get(u) for u in member_uids}
        if len(parents) != 1:
            raise ValueError("group members must share one parent scope")
        shared_parent = parents.pop()
        # Nesting a child inside a SHARED parent would have to mirror the new structure
        # into the definition and every sibling family — that recursion is Phase 3d.
        if shared_parent is not None and self._instances[shared_parent].def_id:
            raise ValueError("cannot group inside a shared sub-patch (deferred to 3d)")

        inst_id = self._fresh_instance_id()
        members: Dict[str, str] = {}  # uid -> local name (template key)
        # A member's local seeds from its globally-unique display name (a node's name
        # or an instance's label); since display names are unique across the unified
        # node+instance namespace, locals can't collide and need no extra dedup.
        used_locals: set = set()
        for u in member_uids:
            base = self._entity_name(u)
            local, idx = base, 1
            while local in used_locals:
                local = f"{base}{idx}"
                idx += 1
            used_locals.add(local)
            members[u] = local

        if interface is None:
            interface = self._derive_interface(members)
        else:
            interface = _iface_from_dict(interface)
        # Reparenting is a multi-scope mutation (detach members from their current
        # parent, re-home them under the new instance, nest the new instance under the
        # shared parent). Wrap it so a mid-way failure restores every index byte-clean.
        with self._transaction():
            # Lift the members out of their current parent (a no-op at top level) before
            # re-homing them, so the old parent's `members` map no longer lists them.
            for u in member_uids:
                self._detach_member(u)
            self._instances[inst_id] = SubPatchInstance(
                uid=inst_id,
                name=self._fresh_instance_name(),
                kind="unique",
                def_id=None,
                members={},
                interface=interface,
                pos=list(pos),
                parent=shared_parent,
            )
            for u, local in members.items():
                self._attach_member(inst_id, u, local)
            # Nest the new instance under the shared parent (its local is the instance's
            # own globally-unique display label).
            if shared_parent is not None:
                self._attach_member(shared_parent, inst_id, self._instances[inst_id].name)
        self._broadcast_node_directory()

        if self._bridge is not None and notify_gui:
            self._bridge.control.on_subpatch_changed()
        return inst_id

    def _fresh_member_local(self, inst_id: str, node_type: str, requested: Optional[str]) -> str:
        """Pick a local member name unique within the sub-patch. For a shared
        instance the local namespace is the family's (def + every sibling carry
        the same locals by strict mirror), so the instance's own members suffice
        — but we also fold in the def's locals defensively."""
        existing = set(self._instances[inst_id].members.values())
        def_id = self._instances[inst_id].def_id
        if def_id and def_id in self._definitions:
            existing |= set(self._definitions[def_id].members.keys())
        if requested is not None:
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
        member_uid: Optional[str] = None,
    ) -> str:
        """Create a node directly inside an existing sub-patch instance.

        The node is spawned with a fresh flat (globally-unique) display name and
        recorded under a per-instance `local` template key. For a SHARED instance
        the new member is mirrored into the definition and every sibling instance
        (strict mirror), matching `update_param`/`set_node_pos`. Returns the new
        member's uid.
        """
        if inst_id not in self._instances:
            raise KeyError(f"No such sub-patch: {inst_id}")
        inst = self._instances[inst_id]
        local = self._fresh_member_local(inst_id, node_type, name)

        # Spawn silently; the single on_subpatch_changed below re-syncs clients
        # atomically (node + updated members map) without a top-level flash. Atomic:
        # a sibling-mirror spawn failure must not leave the family half-mirrored.
        with self._transaction():
            uid = self.add_node(
                node_type,
                category,
                notify_gui=False,
                name=self._fresh_display_name(node_type.lower()),
                params=params,
                pos=tuple(pos),
                membership={"instance": inst_id, "local_name": local},
                # Restore the original uid on redo-of-add (else captured links orphan).
                member_uid=member_uid,
            )
            self._attach_member(inst_id, uid, local)

            def_id = inst.def_id
            if def_id:
                rec = self._node_record(uid)
                rec.pop("uid", None)  # per-instance identity is never shared
                rec.pop("name", None)  # display name is per-instance, not part of the template
                self._definitions[def_id].members[local] = rec
                for sib in self._shared_siblings(inst_id):
                    sib_uid = self._add_node_from_record(
                        self._fresh_display_name(node_type.lower()), dict(rec), notify_gui=False,
                        membership={"instance": sib, "local_name": local},
                    )
                    self._attach_member(sib, sib_uid, local)

        if self._bridge is not None and notify_gui:
            # Root ≡ unique sub-patch: adding a member to a UNIQUE instance is an
            # incremental node_added (its membership rides the payload, so the frontend
            # adds it to the owning scope's members map) — exactly like a top-level add. A
            # SHARED add fans out across the mirror siblings (structural), so resync the
            # whole record via subpatch_changed.
            if def_id:
                self._bridge.control.on_subpatch_changed()
            else:
                self._bridge.control.on_node_added(uid)
        return uid

    def _parent_is_shared(self, inst_id: str) -> bool:
        """True if `inst_id`'s parent is a SHARED instance. Mutating the structure of a
        member of a shared sub-patch (expand/remove/make_unique) would diverge that
        instance from its definition + siblings, so it's rejected — make the parent
        unique first (symmetric to group_nodes' 'cannot group inside a shared' guard)."""
        parent = self._instances[inst_id].parent
        return parent is not None and self._instances[parent].def_id is not None

    def _reject_if_in_shared_parent(self, inst_id: str, verb: str) -> None:
        if self._parent_is_shared(inst_id):
            raise ValueError(
                f"cannot {verb} a member of a shared sub-patch; make the parent unique first"
            )

    def _reject_root(self, inst_id: str, verb: str) -> None:
        """Guard the structural sub-patch mutators against the ROOT scope. ROOT is a real
        scope (every node is its member) but NOT a sub-patch — it has no parent, no
        boundaries, and is the canvas itself, so it cannot be dissolved/shared/expanded.
        ROOT now ships on the wire, so the bridge's remove_node op could route ROOT_ID to
        remove_instance; reject it here rather than corrupting the graph."""
        if inst_id == ROOT_ID:
            raise ValueError(f"cannot {verb} the root scope; it is the canvas, not a sub-patch")

    @mark_unsaved_changes
    def expand_instance(self, inst_id: str, notify_gui: bool = True) -> List[str]:
        """Dissolve a sub-patch, lifting its members ONE level up — into this instance's
        parent, or to top-level when it is a root. A member that is itself a nested
        instance has its whole subtree reparented (not flattened). Flat naming means
        members already carry globally-unique display names, so this is purely
        organizational: no rename, no nd() rewrite. Returns the member UIDs."""
        self._reject_root(inst_id, "expand")
        if inst_id not in self._instances:
            raise KeyError(f"No such sub-patch: {inst_id}")
        self._reject_if_in_shared_parent(inst_id, "expand")
        inst = self._instances[inst_id]
        def_id = inst.def_id
        parent = inst.parent
        # Defensive: a parent boundary that forwards INTO this instance has its inner
        # target dissolved — unwire it (and its external links), like remove_node does.
        self._unwire_parent_boundaries_to(inst_id, notify_gui=False)
        restored: List[str] = list(inst.members)
        for uid in restored:
            self._detach_member(uid)
            if parent is not None:
                # Re-home one level up. A member's local is DECOUPLED from its (mutable,
                # reusable) display name, so dedup against the receiving parent's existing
                # locals to avoid a duplicate-local collision.
                self._attach_member(parent, uid, self._unique_local_in(parent, self._entity_name(uid)))
        self._detach_member(inst_id)  # remove the now-empty instance from its own parent
        del self._instances[inst_id]
        # GC an orphaned shared definition (the expanded instance may have been its
        # last reference), like remove_instance / make_unique.
        if def_id and not any(i.def_id == def_id for i in self._instances.values()):
            self._definitions.pop(def_id, None)
        if self._bridge is not None and notify_gui:
            self._bridge.control.on_subpatch_changed()
        return restored

    # `ungroup` reads better at call sites that just want the group dissolved.
    ungroup = expand_instance

    def _unique_local_in(self, parent: str, base: str) -> str:
        """Lowest free `base`, `base1`, `base2`, … local name within `parent`'s members."""
        existing = set(self._instances[parent].members.values())
        if base not in existing:
            return base
        idx = 1
        while f"{base}{idx}" in existing:
            idx += 1
        return f"{base}{idx}"

    def _unwire_parent_boundaries_to(self, inst_id: str, notify_gui: bool) -> None:
        """Unwire any boundary on `inst_id`'s parent that forwards into `inst_id` (its
        inner_node is this instance's local), tearing down the boundary's external links.
        Called before an instance is dissolved/removed so the parent keeps no boundary
        dangling at a gone member. The parent is unique here (shared parents are guarded)."""
        parent = self._instances[inst_id].parent
        if parent is None:
            return
        local = self._instances[parent].members.get(inst_id)
        if local is None:
            return
        for bid, b in list(self._instances[parent].interface.items()):
            if b.inner_node == local:
                self.wire_boundary(parent, bid, None, None, notify_gui=notify_gui)

    @mark_unsaved_changes
    def remove_instance(self, inst_id: str, notify_gui: bool = True) -> None:
        """Delete a whole sub-patch and its entire subtree: member nodes (and their
        links), and any nested instance members recursively. A virtual sub-patch node
        responds to Delete like any node. GCs an orphaned shared definition, like
        `make_unique`."""
        self._reject_root(inst_id, "delete")
        if inst_id not in self._instances:
            raise KeyError(f"No such sub-patch: {inst_id}")
        self._reject_if_in_shared_parent(inst_id, "delete")
        self._remove_instance_core(inst_id)
        if self._bridge is not None and notify_gui:
            self._bridge.control.on_subpatch_changed()

    def _remove_instance_core(self, inst_id: str) -> None:
        """Recursive teardown (no guard — the public entry checks it once at the root,
        then the whole subtree comes down regardless of its internal sharing)."""
        inst = self._instances[inst_id]
        def_id = inst.def_id
        # Defensive: drop any parent boundary that forwarded into this instance.
        self._unwire_parent_boundaries_to(inst_id, notify_gui=False)
        for member in list(inst.members.keys()):
            if member in self._instances:
                self._remove_instance_core(member)  # recurse into the nested subtree
            else:
                # Pop membership BEFORE remove_node so its boundary defensive-unwire
                # (which keys on membership) no-ops during this teardown.
                self._membership.pop(member, None)
                self.remove_node(member, notify_gui=False)
        self._detach_member(inst_id)  # drop self from its parent (a nested instance, or ROOT)
        del self._instances[inst_id]
        if def_id and not any(i.def_id == def_id for i in self._instances.values()):
            self._definitions.pop(def_id, None)

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

    def _definition_from_instance(self, inst_id: str) -> SubPatchDef:
        """Snapshot an instance's topology+params as a reusable definition.

        Node members serialize as node records under `members`; a NESTED-INSTANCE
        member serializes by reference under `instances` ({local -> {def, pos}}) — so
        the child must already carry its own def_id (share_instance promotes it first).
        The snapshot is read-only."""
        inst = self._instances[inst_id]
        # nd() cross-refs between NODE members are stored in TEMPLATE form (against the
        # local key) so each future instance re-points them at its own fresh member
        # names. Translate live display names -> `\x1f`-marked locals; the marker keeps
        # an INTERNAL ref distinct from a verbatim EXTERNAL ref that happens to share a
        # member's local key. `_entity_name` resolves both nodes and nested instances.
        display_to_local = {
            self._entity_name(uid): _DEF_INTERNAL_REF + local for uid, local in inst.members.items()
        }
        members: Dict[str, Any] = {}
        instances: Dict[str, Any] = {}
        for uid, local in inst.members.items():
            if uid in self.nodes:
                rec = self._node_record(uid)
                # Strip per-instance identity (uid + display name); the template is keyed
                # by local name and each instance mints its own.
                rec.pop("uid", None)
                rec.pop("name", None)
                _rewrite_record_nd(rec, display_to_local)
                members[local] = rec
            else:
                # A nested-instance member: reference its independent def (promoted by
                # share_instance before this snapshot runs).
                child = self._instances[uid]
                instances[local] = {"def": child.def_id, "pos": list(child.pos)}
        links = []
        for link in self._links:
            if self._membership.get(link["node_out"]) == inst_id and self._membership.get(link["node_in"]) == inst_id:
                links.append(self._local_link(link, inst_id))
        # Deep-copy the interface: entries are mutable Boundary ports that boundary
        # edits rewrite, and the def must not alias the source instance's ports.
        return SubPatchDef(
            members=members, links=links, interface=deepcopy(inst.interface), instances=instances
        )

    @mark_unsaved_changes
    def share_instance(self, inst_id: str, notify_gui: bool = True) -> str:
        """Promote a unique instance to a shared one backed by a new definition.

        Returns the definition id. Other instances can then be spawned from it
        with `instantiate_definition`; param edits mirror across all of them. A nested
        unique child is auto-promoted to its OWN independent def first (depth-first), so
        the parent def can reference it — once the parent is a multi-instance template a
        previously-"unique" child necessarily has one strict-mirrored copy per parent
        instance, which IS shared-by-a-def semantics."""
        self._reject_root(inst_id, "share")
        if inst_id not in self._instances:
            raise KeyError(f"No such sub-patch: {inst_id}")
        if self._instances[inst_id].def_id:
            return self._instances[inst_id].def_id
        # Atomic over the whole recursive promotion: a failure mid-snapshot rolls back
        # every child def + kind/def_id flip. (A single non-reentrant core under one
        # transaction, rather than nesting a _transaction per recursive call.)
        with self._transaction():
            def_id = self._promote_to_def(inst_id)
        if self._bridge is not None and notify_gui:
            self._bridge.control.on_subpatch_changed()
        return def_id

    def _promote_to_def(self, inst_id: str) -> str:
        """Depth-first: promote every unique nested child to its own def, then snapshot
        this instance into a fresh def referencing those children. No transaction here —
        the caller (share_instance) wraps the whole recursion in one."""
        inst = self._instances[inst_id]
        if inst.def_id:
            return inst.def_id
        for uid in list(inst.members):
            if uid in self._instances and self._instances[uid].def_id is None:
                self._promote_to_def(uid)
        def_id = self._fresh_def_id()
        self._definitions[def_id] = self._definition_from_instance(inst_id)
        inst.kind = "shared"
        inst.def_id = def_id
        return def_id

    @mark_unsaved_changes
    def instantiate_definition(self, def_id: str, pos=(0, 0), notify_gui: bool = True) -> str:
        """Spawn a fresh shared instance of a definition (strict-mirror sibling). A
        nesting-containing def recursively spawns each nested child as a fresh sibling
        of the child's own def family, so the whole subtree is replicated with disjoint
        uids."""
        if def_id not in self._definitions:
            raise KeyError(f"No such definition: {def_id}")
        # Atomic: a failure anywhere in the recursive spawn tears down every spawned
        # member + restores the maps. ONE outer transaction — the recursive core opens
        # none, so a mid-recursion failure unwinds the whole subtree byte-clean.
        with self._transaction():
            inst_id = self._instantiate_def_core(def_id, pos)
            # A freshly instantiated sibling is a top-level instance — a member of ROOT.
            self._attach_member(ROOT_ID, inst_id, self._unique_local_in(ROOT_ID, self._instances[inst_id].name))
        if self._bridge is not None and notify_gui:
            self._bridge.control.on_subpatch_changed()
        return inst_id

    def _instantiate_def_core(self, def_id: str, pos) -> str:
        """Spawn one instance of `def_id` (no transaction — the caller wraps the whole
        recursion in one). Recurses into nested-instance def references, attaching each
        fresh child under its template local. Mirrors `_restore_instance`'s tree shape,
        differing only in identity (fresh-minted here vs restored on load)."""
        d = self._definitions[def_id]
        inst_id = self._fresh_instance_id()
        self._instances[inst_id] = SubPatchInstance(
            uid=inst_id,
            name=self._fresh_instance_name(),
            kind="shared",
            def_id=def_id,
            members={},
            # Deep-copy so this sibling's boundary edits never cross-mutate the def or
            # other siblings (Boundary ports are mutable).
            interface=deepcopy(d.interface),
            pos=list(pos),
        )
        members: Dict[str, str] = {}  # uid -> local (node members only, for nd() rewrite)
        local_to_uid: Dict[str, str] = {}
        name_map: Dict[str, str] = {}  # template local -> this instance's fresh flat name
        for local, rec in d.members.items():
            disp = self._fresh_display_name(rec["_type"].lower())
            # Spawn silently like add_member_node's sibling mirror: the trailing
            # on_subpatch_changed surfaces every member atomically (and its wiring sweep
            # wires them). A per-member on_node_added would flash them at root and, if
            # this transaction rolls back, leave a phantom node the browser never clears.
            uid = self._add_node_from_record(
                disp, dict(rec),
                membership={"instance": inst_id, "local_name": local},
                notify_gui=False,
            )
            self._attach_member(inst_id, uid, local)
            members[uid] = local
            local_to_uid[local] = uid
            # Key by the `\x1f`-marked local so only INTERNAL refs are re-pointed;
            # EXTERNAL refs (stored verbatim) never collide with a marked key.
            name_map[_DEF_INTERNAL_REF + local] = disp
        # Recurse into nested-instance members: each spawns a fresh child of the child's
        # OWN def (joining that family), attached under the template local so the
        # deep-copied interface's chained boundaries (inner_node = child local) re-point.
        for local, ref in d.instances.items():
            child_uid = self._instantiate_def_core(ref["def"], ref.get("pos", (0, 0)))
            self._attach_member(inst_id, child_uid, local)
        for link in d.links:
            self.add_link(
                local_to_uid[link["node_out"]],
                local_to_uid[link["node_in"]],
                link["slot_out"], link["slot_in"], notify_gui=False,
            )
        # Re-point this level's intra-sub-patch nd() cross-refs from the def's template
        # locals to THIS instance's fresh member display names (node members only).
        self._rewrite_member_expressions(members.keys(), name_map)
        return inst_id

    @mark_unsaved_changes
    def make_unique(self, inst_id: str, notify_gui: bool = True) -> None:
        """Detach a shared instance into a private (unique) copy; GC an orphan def.

        Recurses: a nesting-containing instance privatizes its WHOLE subtree (depth
        first), so "make this instance unique" really makes it private — editing a leaf
        anywhere under it no longer mirrors to a sibling subtree. Each level GC-checks
        its OWN def, so a nested child whose family still has other members (under
        sibling parents) keeps that def alive. Privatizing a SINGLE nested child of a
        shared parent is rejected (it would orphan a def the parent def still
        references) — privatize from the top instead."""
        self._reject_root(inst_id, "make unique")
        if inst_id not in self._instances:
            raise KeyError(f"No such sub-patch: {inst_id}")
        self._reject_if_in_shared_parent(inst_id, "make unique")
        self._make_unique_core(inst_id)
        if self._bridge is not None and notify_gui:
            self._bridge.control.on_subpatch_changed()

    def _make_unique_core(self, inst_id: str) -> None:
        """Recursive privatization (no guard — the public entry checks the root once,
        then the whole subtree privatizes top-down)."""
        inst = self._instances[inst_id]
        # Privatize nested instance members first (depth-first), so the whole subtree
        # leaves its families before this level GC-checks.
        for uid in list(inst.members):
            if uid in self._instances:
                self._make_unique_core(uid)
        def_id = inst.def_id
        inst.kind = "unique"
        inst.def_id = None
        # Detach the interface from the def/siblings so later boundary edits on this
        # now-unique instance can't cross-mutate the family it just left.
        inst.interface = deepcopy(inst.interface)
        if def_id and not any(i.def_id == def_id for i in self._instances.values()):
            self._definitions.pop(def_id, None)

    # ------------------------------------------------------------------
    # In/Out boundary authoring (virtual nodes — interface entries only)
    # ------------------------------------------------------------------

    def _member_uid(self, inst_id: str, local: str) -> Optional[str]:
        """The live uid of the member with local name `local` in `inst_id`, or None.
        Members map uid -> local, so this is the reverse lookup. The uid is the key
        links / membership / the data route use, never the qualified display name.

        Local names are unique within an instance by construction (group dedup,
        _fresh_member_local, member-rename re-key). If that invariant is ever
        violated, fail loudly here rather than silently splicing onto whichever
        member iterates first."""
        matches = [uid for uid, l in self._instances[inst_id].members.items() if l == local]
        if len(matches) > 1:
            raise RuntimeError(
                f"sub-patch {inst_id!r} has duplicate local name {local!r} (members corrupt): {matches}"
            )
        return matches[0] if matches else None

    def _fresh_boundary_id(self, inst_id: str, dir: str) -> str:
        """Lowest unused `in0`/`out0`… among the instance's current interface keys."""
        iface = self._instances[inst_id].interface
        idx = 0
        while f"{dir}{idx}" in iface:
            idx += 1
        return f"{dir}{idx}"

    def _within_subtree(self, uid: Optional[str], root: str) -> bool:
        """True if entity `uid` is `root` or lives anywhere inside root's nesting subtree
        (walk the membership/parent chain up to root)."""
        cur, seen = uid, set()
        while cur is not None and cur not in seen:
            if cur == root:
                return True
            seen.add(cur)
            cur = self._membership.get(cur)
        return False

    def _ancestor_instances(self, uid: str) -> List[str]:
        """The real sub-patch instance uids on `uid`'s parent chain, innermost-first (its
        immediate owning instance, then that instance's parent, …). Empty for a top-level
        entity. ROOT is excluded — it is the canvas, not a collapsible sub-patch, so it
        never surfaces an error/event of its own."""
        out: List[str] = []
        cur = self._membership.get(uid)
        while cur is not None and cur != ROOT_ID:
            out.append(cur)
            cur = self._instances[cur].parent if cur in self._instances else None
        return out

    def _instance_error(self, inst_id: str) -> Optional[str]:
        """First errored descendant of `inst_id` across its whole subtree, or None — the
        error a collapsed group node surfaces. Single source for describe_instance + the
        live error-event propagation (no frontend re-derivation)."""
        return next(
            (self.nodes[u].last_error for u in self.nodes
             if self.nodes[u].last_error and self._within_subtree(u, inst_id)),
            None,
        )

    def _boundary_external_links(self, inst_id: str, dir: str, local: str, slot: str) -> List[dict]:
        """This instance's external flat links for the boundary mapping (local, slot):
        the member-side endpoint matches and the other end is OUTSIDE this instance's
        SUBTREE. Subtree-aware (not single-level), so a link whose other end lives in
        this instance's own nested child counts as INTERNAL and is left alone; a link to
        a sibling-instance member (a different subtree) is external."""
        uid = self._member_uid(inst_id, local)
        # A CHAINED boundary forwards to a nested instance (slot = its boundary id); the
        # real external flat link lives on the deep LEAF, so descend to it. An unwired
        # nested chain has no leaf and thus no external link.
        if uid in self._instances:
            try:
                uid, slot = self.resolve_boundary(uid, slot)
            except (KeyError, ValueError):
                return []
        out: List[dict] = []
        for link in self._links:
            if dir == "in" and link["node_in"] == uid and link["slot_in"] == slot:
                if not self._within_subtree(link["node_out"], inst_id):
                    out.append(link)
            elif dir == "out" and link["node_out"] == uid and link["slot_out"] == slot:
                if not self._within_subtree(link["node_in"], inst_id):
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
        def_id = self._instances[inst_id].def_id
        if not def_id:
            return []
        return [i for i, inst in self._instances.items() if i != inst_id and inst.def_id == def_id]

    def _mirror_boundary_entry(self, inst_id: str, bnd_id: str, entry: Optional[Boundary]) -> None:
        """Mirror a boundary's TOPOLOGY (dir/dtype/inner/pos) to the definition and
        every shared sibling (entry=None removes it). External wires stay per-instance."""
        def_id = self._instances[inst_id].def_id
        if not def_id:
            return
        if entry is None:
            self._definitions[def_id].interface.pop(bnd_id, None)
        else:
            self._definitions[def_id].interface[bnd_id] = deepcopy(entry)
        for sib in self._shared_siblings(inst_id):
            if entry is None:
                self._instances[sib].interface.pop(bnd_id, None)
            else:
                self._instances[sib].interface[bnd_id] = deepcopy(entry)

    @mark_unsaved_changes
    def add_boundary(self, inst_id: str, dir: str, dtype: str, pos=(0, 0), notify_gui: bool = True) -> str:
        """Add a virtual In/Out node to a sub-patch (unwired). Returns its boundary id."""
        self._reject_root(inst_id, "add a boundary to")
        if inst_id not in self._instances:
            raise KeyError(f"No such sub-patch: {inst_id}")
        if dir not in ("in", "out"):
            raise ValueError(f"dir must be in/out, got {dir!r}")
        bnd_id = self._fresh_boundary_id(inst_id, dir)
        entry = Boundary(dir=dir, dtype=dtype, inner_node=None, inner_slot=None, pos=list(pos))
        self._instances[inst_id].interface[bnd_id] = entry
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
        iface = inst.interface
        if bnd_id not in iface:
            raise KeyError(f"No such boundary {bnd_id} on {inst_id}")
        entry = iface[bnd_id]
        dir = entry.dir
        old_local, old_slot = entry.inner_node, entry.inner_slot
        siblings = self._shared_siblings(inst_id)

        if inner_node is None:
            if old_local is not None:
                for iid in [inst_id, *siblings]:
                    self._unsplice_instance(iid, dir, old_local, old_slot, notify_gui)
            new_entry = replace(entry, inner_node=None, inner_slot=None)
        else:
            uid = self._member_uid(inst_id, inner_node)
            if uid is None or self._membership.get(uid) != inst_id or (
                uid not in self.nodes and uid not in self._instances
            ):
                raise ValueError(f"{inner_node} is not a member of {inst_id}")
            # `dtype` is absent on legacy (pre-dtype) entries — tolerate it and heal
            # below by storing the resolved dtype.
            expected = entry.dtype
            if uid in self._instances:
                # The inner target is a NESTED INSTANCE: inner_slot names one of ITS
                # boundaries; dtype/dir come from that child port (the data route
                # descends through it recursively via resolve_boundary). No node-slot
                # lookup and no internal-feed scan — a flat link never keys on an
                # instance uid.
                child = self._instances[uid].interface.get(inner_slot)
                if child is None:
                    raise ValueError(f"no {dir} boundary {inner_slot!r} on nested {inner_node}")
                if child.inner_node is None:
                    # An unwired child boundary isn't an exposed port on the collapsed
                    # node — chaining onto it would forward to nothing (and the editor
                    # would draw a pill edge to a handle the synth node doesn't have).
                    raise ValueError(f"nested boundary {inner_node}.{inner_slot} is not wired yet")
                if child.dir != dir:
                    raise ValueError(
                        f"direction mismatch: nested {inner_node}.{inner_slot} is {child.dir}, port is {dir}"
                    )
                if expected is not None and child.dtype is not None and child.dtype != expected:
                    raise ValueError(
                        f"dtype mismatch: {inner_node}.{inner_slot} is {child.dtype}, boundary is {expected}"
                    )
                dt_name = child.dtype or expected
            else:
                slots = self.nodes[uid].input_slots if dir == "in" else self.nodes[uid].output_slots
                dt = slots.get(inner_slot)
                if dt is None:
                    raise ValueError(f"no {dir} slot {inner_slot!r} on {inner_node}")
                if expected is not None and dt.name != expected:
                    raise ValueError(
                        f"dtype mismatch: {inner_node}.{inner_slot} is {dt.name}, boundary is {expected}"
                    )
                dt_name = dt.name
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
            for k, e in iface.items():
                if k != bnd_id and e.dir == dir and e.inner_node == inner_node and e.inner_slot == inner_slot:
                    raise ValueError(f"inner slot {inner_node}.{inner_slot} already exposed by {k}")
            if old_local is not None and (old_local, old_slot) != (inner_node, inner_slot):
                # Re-pointing a CHAINED (nested-instance) boundary would re-splice an
                # external link onto an instance uid — unsupported in 3b. Unwire first.
                if uid in self._instances or self._member_uid(inst_id, old_local) in self._instances:
                    raise ValueError("re-pointing a chained boundary isn't supported yet; unwire it first")
                for iid in [inst_id, *siblings]:
                    self._resplice_instance(iid, dir, old_local, old_slot, inner_node, inner_slot, notify_gui)
            new_entry = replace(entry, dtype=dt_name, inner_node=inner_node, inner_slot=inner_slot)

        iface[bnd_id] = new_entry
        self._mirror_boundary_entry(inst_id, bnd_id, new_entry)
        if self._bridge is not None and notify_gui:
            self._bridge.control.on_subpatch_changed()

    @mark_unsaved_changes
    def wire_boundary_to_leaf(self, outer_inst, bnd, leaf_node_uid, leaf_slot, notify_gui: bool = True):
        """Auto-chain a pre-created OUTER boundary straight to a deeply-nested leaf node's
        slot, building + chaining an In/Out boundary at every intermediate level. Returns
        the auto-created (inst_id, bnd_id) pairs (innermost-first) so the caller can undo
        the whole chain as one step. The outer boundary's dir is reused; the dtype is the
        leaf slot's. The ancestor chain is walked via the 3a parent edges (no new index)."""
        if outer_inst not in self._instances or bnd not in self._instances[outer_inst].interface:
            raise KeyError(f"No such boundary {outer_inst}:{bnd}")
        if leaf_node_uid not in self.nodes:
            raise KeyError(f"No such node: {leaf_node_uid}")
        dir = self._instances[outer_inst].interface[bnd].dir
        dt = self._slot_dtype(leaf_node_uid, leaf_slot, dir)
        # Walk leaf -> ... -> outer_inst (innermost-first) along the membership/parent
        # chain; bail if outer_inst is not actually an ancestor of the leaf.
        chain: List[tuple] = []  # (inst_id, member_uid)
        member, inst = leaf_node_uid, self._membership.get(leaf_node_uid)
        while inst is not None:
            chain.append((inst, member))
            if inst == outer_inst:
                break
            member, inst = inst, self._instances[inst].parent
        else:
            raise ValueError(f"{outer_inst} is not an ancestor of {leaf_node_uid}")

        created: List[tuple] = []
        inner_slot = leaf_slot
        # Atomic: a mid-chain failure must leave no orphan intermediate boundary. Each
        # wire here is FRESH (old inner is None), so no live transport re-splice runs
        # inside the block — rollback is a pure map-restore.
        with self._transaction():
            for inst_id, member_uid in chain:  # innermost-first
                local = self._instances[inst_id].members[member_uid]
                if inst_id == outer_inst:
                    bnd_id = bnd  # reuse the pre-created outer boundary
                else:
                    bnd_id = self.add_boundary(inst_id, dir, dt, notify_gui=False)
                    created.append((inst_id, bnd_id))
                self.wire_boundary(inst_id, bnd_id, local, inner_slot, notify_gui=False)
                inner_slot = bnd_id  # the next (outer) level forwards to THIS boundary
        if self._bridge is not None and notify_gui:
            self._bridge.control.on_subpatch_changed()
        return created

    @mark_unsaved_changes
    def remove_boundary(self, inst_id: str, bnd_id: str, notify_gui: bool = True) -> None:
        """Delete an In/Out node, tearing down its external wires across siblings."""
        inst = self._instances[inst_id]
        iface = inst.interface
        if bnd_id not in iface:
            raise KeyError(f"No such boundary {bnd_id} on {inst_id}")
        entry = iface[bnd_id]
        if entry.inner_node is not None:
            for iid in [inst_id, *self._shared_siblings(inst_id)]:
                self._unsplice_instance(iid, entry.dir, entry.inner_node, entry.inner_slot, notify_gui)
        del iface[bnd_id]
        self._mirror_boundary_entry(inst_id, bnd_id, None)
        if self._bridge is not None and notify_gui:
            self._bridge.control.on_subpatch_changed()

    @mark_unsaved_changes
    def set_boundary_pos(self, inst_id: str, bnd_id: str, pos) -> List[tuple]:
        """Move an In/Out pill, mirroring the pos across shared siblings (like member
        pos). Returns the (inst_id, bnd_id) pairs changed so the bridge can broadcast."""
        pos = list(pos)
        iface = self._instances[inst_id].interface
        if bnd_id not in iface:
            raise KeyError(f"No such boundary {bnd_id} on {inst_id}")
        changed = [(inst_id, bnd_id)]
        iface[bnd_id] = replace(iface[bnd_id], pos=pos)
        def_id = self._instances[inst_id].def_id
        if def_id:
            if bnd_id in self._definitions[def_id].interface:
                self._definitions[def_id].interface[bnd_id].pos = pos
            for sib in self._shared_siblings(inst_id):
                if bnd_id in self._instances[sib].interface:
                    self._instances[sib].interface[bnd_id] = replace(self._instances[sib].interface[bnd_id], pos=pos)
                    changed.append((sib, bnd_id))
        return changed

    def resolve_boundary(self, inst_id: str, bnd_id: str) -> tuple:
        """Translate a (sub-patch, boundary) port to the real LEAF (node uid, slot) for
        the external-wire splice / data route. Descends chain-to-leaf: when a boundary
        forwards to a nested INSTANCE member, its `inner_slot` is that child's boundary
        id, so we recurse into the child and repeat until a real node is reached. Raises
        KeyError (unknown boundary) / ValueError (unwired or gone) anywhere in the chain
        — the exact surface the data route catches to close cleanly."""
        cur_inst, cur_bnd = inst_id, bnd_id
        seen: set = set()
        while True:
            # Defensive: the nesting forest is acyclic (invariant), so this can't loop;
            # the guard turns a hypothetical corruption into a loud error, not a hang.
            if (cur_inst, cur_bnd) in seen:
                raise ValueError(f"boundary chain cycles through {cur_inst}:{cur_bnd}")
            seen.add((cur_inst, cur_bnd))
            inst = self._instances.get(cur_inst)
            if inst is None or cur_bnd not in inst.interface:
                raise KeyError(f"No such boundary {cur_inst}:{cur_bnd}")
            entry = inst.interface[cur_bnd]
            if entry.inner_node is None:
                raise ValueError(f"boundary {cur_inst}:{cur_bnd} is not wired yet")
            uid = self._member_uid(cur_inst, entry.inner_node)
            if uid is None:
                raise ValueError(f"boundary {cur_inst}:{cur_bnd} inner member is gone")
            if uid in self._instances:  # forward to the nested instance's own boundary
                cur_inst, cur_bnd = uid, entry.inner_slot
                continue
            return uid, entry.inner_slot

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
        inst = self._instances.get(inst_id)
        if inst is None or not inst.def_id:
            return
        def_id = inst.def_id
        local = inst.members[node]
        # Update the definition's stored value (the save source of truth). If the
        # param carries a (possibly stashed) expression — stored as a {value,
        # expression, ...} dict — update only its value field so a value edit
        # doesn't wipe the binding.
        rec = self._definitions[def_id].members.get(local)
        if rec is not None:
            grp = rec.setdefault("params", {}).setdefault(group, {})
            existing = grp.get(name)
            if isinstance(existing, dict) and "expression" in existing:
                existing["value"] = value
            else:
                grp[name] = value
        # Propagate to every sibling instance's corresponding member.
        for other_id, other in self._instances.items():
            if other_id == inst_id or other.def_id != def_id:
                continue
            for onode, olocal in other.members.items():
                if olocal == local:
                    try:
                        self.nodes[onode].update_param(group, name, value)
                    except Exception as exc:
                        # Surface, don't swallow: a sibling that fails to mirror would
                        # silently drift from the family (and a later save would persist
                        # whichever sibling is live). Report it so the divergence is visible.
                        self._surface_mirror_failure(onode, f"{group}.{name}", exc)

    @mark_unsaved_changes
    def set_expression(
        self,
        uid: str,
        group: str,
        name: str,
        expression: Optional[str],
        enabled: bool = False,
        triggers_process: bool = False,
        autoeval: bool = False,
    ) -> None:
        """Bind/clear a param expression, mirroring across shared siblings.

        Strict mirror, same as `update_param` but for the expression binding: an
        edit on a shared member updates the definition (the save source of truth)
        and every sibling instance's corresponding member in lockstep. Must route
        through here, not straight to the NodeRef, or shared expressions silently
        desync (and never persist into the definition)."""
        self.nodes[uid].set_expression(group, name, expression, enabled, triggers_process, autoeval)

        inst_id = self._membership.get(uid)
        if not inst_id:
            return
        inst = self._instances.get(inst_id)
        if inst is None or not inst.def_id:
            return
        def_id = inst.def_id
        local = inst.members[uid]
        # Intra-sub-patch nd() cross-refs are authored against THIS instance's live
        # member display names; the definition stores them in TEMPLATE form (against
        # the local key), and each sibling carries its OWN member display names.
        me_by_local = {l: self.nodes[u].name for u, l in inst.members.items()}
        # Mark INTERNAL refs (`\x1f`-prefixed) so the def stores them distinct from an
        # EXTERNAL ref that happens to match a member's local key (kept verbatim).
        display_to_local = {disp: _DEF_INTERNAL_REF + l for l, disp in me_by_local.items()}
        # Record on the definition. Read the normalized binding back off the primary
        # node's param (the NodeRef just applied the same canonical gating the node
        # will), and store it in the {value, expression, ...} shape Param.serialize
        # emits — or a flat value when the expression was cleared.
        rec = self._definitions[def_id].members.get(local)
        if rec is not None:
            p = self.nodes[uid].params[group][name]
            params = rec.setdefault("params", {}).setdefault(group, {})
            if getattr(p, "expression", None) is not None:
                params[name] = {
                    "value": p._value,
                    "expression": rewrite_nd_refs(p.expression, display_to_local),
                    "expression_enabled": bool(getattr(p, "expression_enabled", False)),
                    "expression_triggers_process": bool(getattr(p, "expression_triggers_process", False)),
                    "expression_autoeval": bool(getattr(p, "expression_autoeval", False)),
                }
            else:
                params[name] = p._value
        # Propagate to every sibling instance's corresponding member, re-pointing any
        # intra-sub-patch cross-ref at the sibling's OWN members (this display -> sib display).
        for other_id, other in self._instances.items():
            if other_id == inst_id or other.def_id != def_id:
                continue
            sib_by_local = {l: self.nodes[u].name for u, l in other.members.items()}
            sib_map = {me_by_local[l]: sib_by_local[l] for l in me_by_local if l in sib_by_local}
            sib_expr = rewrite_nd_refs(expression, sib_map)
            for onode, olocal in other.members.items():
                if olocal == local and onode in self.nodes:
                    try:
                        self.nodes[onode].set_expression(group, name, sib_expr, enabled, triggers_process, autoeval)
                    except Exception as exc:
                        self._surface_mirror_failure(onode, f"{group}.{name} (expression)", exc)

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
        inst = self._instances.get(inst_id)
        if inst is None or not inst.def_id:
            return changed
        def_id = inst.def_id
        local = inst.members[name]
        # Record on the definition (the save source of truth). Assign a fresh dict
        # rather than mutating in place — `_node_record` can leave the def's member
        # gui_kwargs aliased to a live node's dict, and we must not touch that.
        rec = self._definitions[def_id].members.get(local)
        if rec is not None:
            rec["gui_kwargs"] = {**(rec.get("gui_kwargs") or {}), "pos": pos}
        # Propagate to every sibling instance's corresponding member. Tolerate a
        # stale members entry whose node was already removed (remove_node leaves
        # _membership/members untouched) — same defensive stance as update_param.
        for other_id, other in self._instances.items():
            if other_id == inst_id or other.def_id != def_id:
                continue
            for onode, olocal in other.members.items():
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
        # Overlay the AUTHORITATIVE live params: update_param/set_expression write
        # ref.params synchronously, but serialized_state only refreshes on the node's
        # next echo — so a save or share immediately after an edit would otherwise
        # snapshot the stale value (and lose a just-bound expression).
        state["params"] = ref.params.serialize()
        state["gui_kwargs"] = ref.gui_kwargs
        state.pop("output_subscribers", None)
        if ref.uid is not None:
            state["uid"] = ref.uid
        if ref.name is not None:
            state["name"] = ref.name
        return state

    def _local_link(self, link: Dict[str, str], inst_id: str) -> Dict[str, str]:
        m = self._instances[inst_id].members
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
        # ROOT is DISSOLVED into the top-level `root_nodes` / `root_links` / `instances`,
        # so the .gfi format is unchanged (ROOT is a runtime construct, never serialized
        # as an instance). root_nodes/links/instance members are keyed by uid; the
        # readable display name rides inside each node record (see _node_record).
        root = self._instances[ROOT_ID]
        root_nodes = {uid: self._node_record(uid) for uid in root.members if uid in self.nodes}

        internal: Dict[str, list] = {iid: [] for iid in self._instances}
        root_links: list = []
        for link in self._links:
            oi = self._membership.get(link["node_out"])
            ii = self._membership.get(link["node_in"])
            # Internal to a real sub-patch only when both ends share ONE non-root scope;
            # a link inside ROOT (both ends top-level) is a root_link, not "internal".
            if oi is not None and oi == ii and oi != ROOT_ID:
                internal[oi].append(self._local_link(link, oi))
            else:
                root_links.append(dict(link))

        # The top-level sub-patches are ROOT's instance members; each recursively emits
        # its child instances under its own `instances` sub-key, so the document mirrors
        # the live `inst.parent` forest (a nested instance is emitted exactly once, under
        # its parent — never a flat sibling).
        definitions: Dict[str, Any] = {}
        instances: Dict[str, Any] = {
            iid: self._emit_instance(iid, internal, definitions)
            for iid in root.members
            if iid in self._instances
        }
        return root_nodes, root_links, definitions, instances

    def _emit_instance(self, iid: str, internal: Dict[str, list], definitions: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize one instance for the save doc, recursing into nested-instance
        members. Node members serialize under `members` (full node records); nested
        instances serialize under `instances` (recursed, each carrying its parent-`local`
        so a chained boundary's inner_node resolves on load). Shared instances stay
        node-only (a definition containing nesting is rejected — that's 3d)."""
        inst = self._instances[iid]
        if inst.def_id:
            def_id = inst.def_id
            if def_id not in definitions:
                d = self._definitions[def_id]
                definitions[def_id] = {
                    "members": deepcopy(d.members),
                    "links": deepcopy(d.links),
                    "interface": _iface_to_dict(d.interface),
                    # Nested-instance members of the definition, by reference (3d).
                    "instances": deepcopy(d.instances),
                }
            entry: Dict[str, Any] = {
                "kind": "shared",
                "def": def_id,
                "pos": inst.pos,
                # Only per-instance state — topology+params live in the definition. The
                # flat display name is per-instance too (round-trips so a reload restores
                # the names the user saw). NODE members only here; nested-instance
                # members carry their own per-instance state under `instances` below.
                "members": {
                    local: {
                        "uid": self.nodes[nn].uid,
                        "name": self.nodes[nn].name,
                        "pos": (self.nodes[nn].gui_kwargs or {}).get("pos"),
                    }
                    for nn, local in inst.members.items()
                    if nn in self.nodes
                },
            }
            # A shared instance's nested-instance members are themselves instances with
            # their own per-instance identity — recurse them like the unique branch.
            children: Dict[str, Any] = {}
            for child_uid, local in inst.members.items():
                if child_uid in self._instances:
                    child = self._emit_instance(child_uid, internal, definitions)
                    child["local"] = local
                    children[child_uid] = child
            if children:
                entry["instances"] = children
        else:
            members = {
                local: self._node_record(uid)
                for uid, local in inst.members.items()
                if uid in self.nodes  # node members only — never serialize an instance here
            }
            children: Dict[str, Any] = {}
            for child_uid, local in inst.members.items():
                if child_uid in self._instances:
                    child = self._emit_instance(child_uid, internal, definitions)
                    child["local"] = local  # carry the parent-local explicitly
                    children[child_uid] = child
            entry = {
                "kind": "unique",
                "pos": inst.pos,
                "interface": _iface_to_dict(inst.interface),
                "members": members,
                "links": internal.get(iid, []),
                "instances": children,
            }
        entry["name"] = inst.name
        # Per-instance viewer state (collapsed sub-patch slots) rides on the record so a
        # reload keeps the kind/settings the user chose (backlog #17).
        if inst.viewers:
            entry["viewers"] = deepcopy(inst.viewers)
        return entry

    def _add_node_from_record(
        self, name: str, node: Dict[str, Any], membership=None, notify_gui: bool = True,
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
            **gk,
        )

    def _expand_doc(self, root_nodes, root_links, instances, definitions=None) -> None:
        """Atomically splice a v2 document into the live graph. Validates every
        referenced definition up front and runs the splice in a transaction, so a
        bad/partial doc fails fast and leaves no half-spliced graph or leaked node
        processes behind."""
        definitions = definitions or {}
        known = set(self._definitions) | set(definitions)

        def _check(insts):  # recurse nested `instances` so a bad def at any depth fails fast
            for inst_id, inst in insts.items():
                if inst.get("kind") == "shared" and inst.get("def") not in known:
                    raise KeyError(
                        f"instance {inst_id!r} references missing definition {inst.get('def')!r}"
                    )
                _check(inst.get("instances") or {})

        _check(instances)
        # A definition can also reference a child definition by-reference (def.instances);
        # validate those too so a corrupt doc fails fast at the precheck, not mid-splice.
        for def_id, draw in definitions.items():
            for local, ref in (draw.get("instances") or {}).items():
                if ref.get("def") not in known:
                    raise KeyError(
                        f"definition {def_id!r} references missing child definition {ref.get('def')!r}"
                    )
        with self._transaction():
            self._splice_doc(root_nodes, root_links, instances, definitions)

    def _restore_instance(self, saved_id: str, inst: Dict[str, Any], internal_links: List[tuple]) -> str:
        """Reconstruct one saved instance (post-order: spawn node members, recurse into
        nested-instance members, create the SubPatchInstance, then attach all members
        through the one `_attach_member` funnel). Returns the live (possibly re-minted)
        uid. Nested children are created+registered before the parent attaches them, so
        the funnel's `.parent` arm always sees the child already in `_instances`."""
        # Restore the saved uid if free (fresh load); mint a fresh one when splicing into
        # a graph that already holds it. The child is registered before its parent reads
        # the returned uid, so a re-minted child can't dangle.
        inst_id = saved_id if saved_id not in self._instances else self._mint_uid()
        kind = inst.get("kind", "unique")
        members_map: Dict[str, str] = {}  # uid -> local
        local_to_uid: Dict[str, str] = {}
        if kind == "shared":
            def_id = inst["def"]
            d = self._definitions[def_id]
            per = inst.get("members") or {}
            name_map: Dict[str, str] = {}  # template local -> restored flat display
            for local, rec in d.members.items():
                pm = per.get(local) or {}
                disp = self._restore_member_name(pm.get("name"), rec["_type"])
                node_rec = dict(rec)
                if pm.get("uid"):
                    node_rec["uid"] = pm["uid"]
                if pm.get("pos") is not None:
                    node_rec["gui_kwargs"] = {**(rec.get("gui_kwargs") or {}), "pos": pm["pos"]}
                uid = self._add_node_from_record(
                    disp, node_rec,
                    membership={"instance": inst_id, "local_name": local},
                )
                members_map[uid] = local
                local_to_uid[local] = uid
                # `\x1f`-mark so only INTERNAL refs re-point; EXTERNAL refs in the
                # def (verbatim) survive the splice untouched.
                name_map[_DEF_INTERNAL_REF + local] = disp
            # The def stores nd() cross-refs in template-local form — re-point them
            # at THIS instance's restored member display names.
            self._rewrite_member_expressions(members_map.keys(), name_map)
            # Recurse a shared instance's nested-instance members (its own per-instance
            # subtrees), exactly like the unique branch — each carries its parent-local.
            for child_saved_id, child_rec in (inst.get("instances") or {}).items():
                child_uid = self._restore_instance(child_saved_id, child_rec, internal_links)
                members_map[child_uid] = child_rec["local"]
            # Deep-copy: each loaded shared instance gets its own Boundary ports
            # so later boundary edits don't alias the definition / siblings.
            interface = deepcopy(d.interface)
            for link in d.links:
                internal_links.append((local_to_uid, link))
        else:
            for local, rec in (inst.get("members") or {}).items():
                uid = self._add_node_from_record(
                    self._restore_member_name(rec.get("name"), rec["_type"]), rec,
                    membership={"instance": inst_id, "local_name": local},
                )
                members_map[uid] = local
                local_to_uid[local] = uid
            # Recurse into nested-instance members; each carries its parent-local. The
            # child is fully restored (and registered in _instances) before we record it.
            for child_saved_id, child_rec in (inst.get("instances") or {}).items():
                child_uid = self._restore_instance(child_saved_id, child_rec, internal_links)
                members_map[child_uid] = child_rec["local"]
            interface = _iface_from_dict(inst.get("interface", {}))
            for link in inst.get("links", []):
                internal_links.append((local_to_uid, link))

        self._instances[inst_id] = SubPatchInstance(
            uid=inst_id,
            name=self._restore_instance_name(inst.get("name")),
            kind=kind,
            def_id=inst.get("def") if kind == "shared" else None,
            members={},
            interface=interface,
            pos=inst.get("pos", [0, 0]),
            viewers=inst.get("viewers") or {},
        )
        for uid, local in members_map.items():
            self._attach_member(inst_id, uid, local)
        return inst_id

    def _splice_doc(self, root_nodes, root_links, instances, definitions=None) -> None:
        """Splice a v2 document's root graph + sub-patch instances into the live
        flat graph. Add all nodes first, then all links (so add_link never races
        a not-yet-spawned endpoint). Handles both unique (inline) and shared
        (definition-backed) instances. Not atomic on its own — call via _expand_doc."""
        # Saved-doc definitions are plain dicts; rebuild the typed SubPatchDef records.
        for def_id, draw in (definitions or {}).items():
            self._definitions[def_id] = SubPatchDef(
                members=draw.get("members", {}),
                links=draw.get("links", []),
                interface=_iface_from_dict(draw.get("interface", {})),
                instances=draw.get("instances", {}),  # nested-instance refs (3d)
            )

        for key, node in root_nodes.items():
            # v2 records carry their display name; v1 used the dict key as the name.
            self._add_node_from_record(node.get("name", key), node)

        # (local_to_uid, local_link): resolve template-local link endpoints to the
        # freshly-minted member uids once all members of the instance exist.
        internal_links: List[tuple] = []
        for saved_id, inst in instances.items():
            # The doc's top-level `instances` are ROOT's instance members; nested ones are
            # reached only through their parent's `instances` sub-key (recursion below).
            iid = self._restore_instance(saved_id, inst, internal_links)
            self._attach_member(ROOT_ID, iid, self._unique_local_in(ROOT_ID, self._instances[iid].name))

        for local_to_uid, link in internal_links:
            self.add_link(
                local_to_uid[link["node_out"]],
                local_to_uid[link["node_in"]],
                link["slot_out"], link["slot_in"],
            )
        for link in root_links:
            # root_links carry verbatim saved node uids. On a fresh load (load() guards an
            # empty graph) these match the just-restored nodes, so they resolve. NOTE
            # (review #11, deferred): splicing into a POPULATED graph could collide a saved
            # uid and re-mint it, leaving these by-uid root/cross-level links dangling — a
            # paste/import feature would need a saved->live uid_map threaded through. No
            # public path reaches that today; revisit when paste lands.
            self.add_link(link["node_out"], link["node_in"], link["slot_out"], link["slot_in"])

    def _node_directory(self) -> Dict[str, str]:
        """Map each live DISPLAY name to its node's stable transport id, for
        `nd('name')` resolution. Display names are kept unique (add auto-numbers,
        rename disambiguates), so this lookup is unambiguous even though the graph
        itself keys on uid."""
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
