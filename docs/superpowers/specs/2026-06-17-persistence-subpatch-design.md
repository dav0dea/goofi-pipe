# goofi-pipe Persistence v2 + Sub-Patch Runtime + Phase-1 Full-FS Save/Load — Design Spec

> **Status:** design approved-in-principle (2026-06-17), pending spec review.
> **Author:** brainstormed with Phil via two adversarial design workflows (44 + 14 agents).
> **Supersedes:** the current flat-YAML `.gfi` save/load and the browser-only download path.

## Decision log (authoritative — overrides any contradicting text below)

These are the locked product decisions from the brainstorming session. Where the generated
body below still reflects an earlier draft, **this log wins**; the load-bearing sections (§0
corrections, §2.6 nd(), §2.13 strict mirror) have been corrected inline to match.

1. **Phase it.** Ship the save/load UX fix on today's flat format first; recursive format +
   sub-patch runtime follow.
2. **No node source code in `.gfi`, ever.** Nodes are referenced by `type`+`category` only.
   The sole executable content in a patch remains param-eval expressions. (The earlier
   "embed code / non-native indicator / switch-to-native" idea is cancelled.)
3. **Full sub-patch runtime** via flatten-at-runtime (group → run-as-unit → expand).
4. **Filesystem: full access, no jail.** The in-app browser exposes the whole backend FS over
   absolute paths. The LAN is trusted; bind stays `0.0.0.0`. Device auth is explicitly future
   work, out of scope here.
5. **Always write v2** + a one-shot converter rewrites `examples/*.gfi` once; v1 still reads.
6. **Sharing = strict mirror.** Shared instances are byte-identical clones (topology + params)
   in lockstep; only `gui_kwargs.pos` differs. "Make Unique" forks a private inline copy.
   Build the `unique`/inline path first as a working core, then add shared ref-tracking.
7. **Boundary = manifest-mapping**, rewritten to direct flat leaf links (no proxy nodes).
8. **nd() — cross-boundary references ALLOWED** *(reverses the earlier "forbid" draft)*. Because
   flatten-at-runtime keeps one flat name→id directory, referencing out-of-sub-patch nodes
   works for free and is convenient. **We do NOT build scoped directories** (this removes what
   verification flagged as the highest-risk change). Consequence: a bare `nd('sibling')` inside
   a freshly-grouped sub-patch resolves against the flat namespace, so `group_nodes` does a
   **best-effort rewrite of string-literal** nd() args that name a fellow member to the member's
   qualified `instance::name`. Non-literal / dynamic args are left as-is and resolve globally.
   The bare-name-shadows-a-top-level-node edge case is accepted as a minor quirk.
9. **Param drift: not handled (temporary).** Nodes may self-write params during `process()`.
   The definition stays the save source-of-truth; user edits propagate to siblings; node
   self-edits are not reconciled or specially guarded. Accepted as temporary — becomes a
   non-issue once nodes can no longer self-edit params. No reconcile step, no determinism check.
10. **Baked defaults** (resolved without further input): `member_uid` widened to `uuid4().hex[:12]`
    with the dedup-and-remap pass retained; the Examples menu degrades gracefully to empty under
    a wheel install where `examples/` is unpackaged; the brief respawn data-gap on topology edits
    is accepted (latest-wins; only changed-topology members respawn).

---

## 0. Overview & Decisions

This spec covers three bodies of work, ordered so each ships independently:

- **Phase 1** — backend-default save/load with a full-filesystem in-app browser, an Examples menu, and the `version:2` envelope + read-compat. Independent of sub-patches.
- **Phase 2a** — a working **UNIQUE (inline) sub-patch**: group/expand/ungroup, flatten-at-runtime, transactional splice, identity/namespacing, data-plane re-keying. No sharing.
- **Phase 2b** — **SHARED ref-tracking** layered on 2a: a definition store, strict-mirror param + topology propagation, make-unique.

### Locked decisions honored
1. Phase 1 = backend save by default; full-FS browser; fix the "downloads untitled" bug; no FS jail; bind stays `0.0.0.0`; device auth is out of scope.
2. Format v2 = recursive, no embedded node source; `definitions` store; instances are UNIQUE (inline) or SHARED (def ref); always write v2; one-shot converter; v1 read-compat.
3. Sharing = strict mirror; only `gui_kwargs.pos` may differ per instance; make-unique forks.
4. Boundary = manifest-mapping; parent wires to a boundary slot rewrite into direct flat leaf links; no proxy nodes.
5. Runtime = flatten-at-runtime into the existing flat `NodeContainer` + `_links`; sub-patch membership/interface/definitions are **first-class validated manager state**.
6. `nd()` cross-boundary references are **allowed** (see decision-log item 8). The flat global
   directory is kept; no scoped directories.

### Key cross-cutting corrections folded in (from verification)
- **nd() under namespacing — keep the flat global directory; allow cross-boundary refs.** Per
  decision-log item 8, we do NOT build scoped resolution. The flat directory (`node.py:355`
  `self._node_directory.get(name, name)`, `manager.py:407-419`) resolves any node by its
  flat/qualified name, so cross-boundary references work for free. The only handling needed is in
  `group_nodes`: best-effort rewrite of **string-literal** nd() args that name a fellow member to
  the qualified `instance::name`; dynamic args resolve globally and a bare name that collides with
  a top-level node is an accepted minor quirk. See §2.6 (corrected).
- **Identity spine must be plumbed through `add_node` and `load`** — the only load path reads `_type/category/params/gui_kwargs` and would silently drop a persisted `uid`/`membership`. `add_node` gains optional `member_uid` and `membership` parameters; `load` reads them; v1 mints fresh.
- **The /data route cannot carry a separator nor survive `node_id` re-mint.** Switch to an opaque, stable route `/data/by-uid/{uid}/{slot}` keyed on `member_uid`. (`%2F`-encoding a two-variable `/data/{node}/{slot}` route is not viable — aiohttp splits dynamic resources on `/`.)
- **Data fan-out must be multiplexed in the bridge.** `set_data_handler` is single-callback-per-slot and *evicting* (verified `node_helpers.py:405-431` — pops prev, closes its subscriber, `unregister_subscriber`, installs one). Two viewers on one slot already clobber today; respawn re-wire makes it worse. DataHub must hold ONE handler per `(uid, slot)` and fan out to a set of forwarders.
- **The 255-byte budget must be computed from the real service-name template across all four services.** Verified `transport.py:59-72`: `goofi.{instance_id}.{kind}.{node_id}.{slot}` with `node_id = {display_name}-{uuid8}`, `instance_id = get_instance_id()`. `.status.` is 2 bytes longer than `.data.`; check all of `{data, ctrl, status, self}` and the longest real output slot, over the FULL nested namespaced name.
- **Transactions must snapshot AND restore `self._links`, `self._node_groups`, NodeContainer keys, and deep-copy the subpatch dicts** — and must journal removes (with full node spec) so remove-before-add deltas restore correctly. `add_link` silently tears down a displaced wire (`manager.py:351-359`), which a naive rollback cannot undo.
- **Renames must rewrite ALL name-keyed state and re-wire the bridge.** `NodeContainer.rename` re-keying `_nodes` alone leaves `_node_groups` (`_same_group`), `_links` endpoints, `_membership`, and the bridge's name-captured `STATE_UPDATE`/`PROCESSING_ERROR` closures (`control.py` `_wire_node_status`) stale.
- **Reserved separator `::`** enforced on EVERY name ingress (auto-namer, `add_node(name=)`, v1 load replay). Reserve discriminator keys `{version, definitions, kind, def, inline, interface, membership, uid}` against node/param/group/slot names.
- **`flat_view` must hard-fail on non-empty `definitions` in the Phase-1 build**, so a real v2 sub-patch file cannot mis-load as flat before expansion lands.
- **Preserve the initial-load layout broadcast** (`manager.py:457-460`) in the rewritten load body.

---

## Phase 1 — backend save/load + full-FS browser

### 1.1 Versioned normalize boundary — `src/goofi/patch_format.py` (NEW)

Single read/write boundary for patch dicts. Touches no transport.

```python
CURRENT_VERSION = 2

def normalize_loaded(raw: dict) -> dict:
    version = raw.get("version")
    if version is None:
        return _v1_to_v2(raw)
    if version == 2:
        return raw
    raise ValueError(f"unsupported patch version {version!r}")

def _v1_to_v2(raw: dict) -> dict:
    doc = {"version": 2, "definitions": {},
           "root": {"nodes": raw["nodes"], "links": raw.get("links", [])}}
    if raw.get("layout") is not None:
        doc["layout"] = raw["layout"]
    return doc

def build_v2(nodes: dict, links: list, layout, definitions: dict, instances: dict) -> dict:
    doc = {"version": 2, "definitions": definitions,
           "root": {"nodes": nodes, "links": links, "instances": instances}}
    if layout is not None:
        doc["layout"] = layout
    return doc

def flat_view(doc: dict) -> tuple[dict, list, dict, dict, object]:
    """Return (root_nodes, root_links, root_instances, definitions, layout).
    PHASE-1 GUARD: until the recursive expander lands, refuse to silently
    discard sub-patch state."""
    defs = doc.get("definitions") or {}
    root = doc["root"]
    instances = root.get("instances") or {}
    if defs or instances:
        raise ValueError(
            "sub-patch definitions/instances require recursive expansion "
            "(not enabled in this build)")
    return root["nodes"], root.get("links", []), instances, defs, doc.get("layout")
```

**REQUIRED-FIX (folded):** `flat_view` raises if `definitions` OR `instances` is non-empty. Phase 2 replaces `flat_view` with the recursive expander; the Phase-1 guard guarantees a v2 sub-patch file errors loudly instead of loading a partial flat graph.

### 1.2 Manager.load / save changes

`load` (manager.py:435-460) becomes:

```python
raw = yaml.load(f, Loader=yaml.FullLoader)
from goofi.patch_format import normalize_loaded, flat_view
nodes_doc, links_doc, instances_doc, defs_doc, layout = flat_view(normalize_loaded(raw))
for name, node in nodes_doc.items():
    # reserved-separator ingress guard (also covers v1 replay)
    _reject_reserved_name(name)
    xpos, ypos = node["gui_kwargs"]["pos"]
    if xpos == np.iinfo(np.int32).min or ypos == np.iinfo(np.int32).min:
        print(f"WARNING: Node '{name}' corrupted position. Resetting to (0,0).")
        node["gui_kwargs"]["pos"] = (0, 0)
    self.add_node(node["_type"], node["category"], name=name, params=node["params"],
                  member_uid=node.get("uid"), membership=node.get("membership"),
                  **node["gui_kwargs"])
for link in links_doc:
    self.add_link(link["node_out"], link["node_in"], link["slot_out"], link["slot_in"])
self._layout = layout
# REQUIRED-FIX: keep the initial-load layout broadcast (manager.py:457-460)
if self._bridge is not None and layout is not None:
    self._bridge.control.broadcast_threadsafe({"event": "layout", "layout": layout})
```

`save` (manager.py:512-515) always emits v2 via `build_v2(...)` (Phase 1: empty `definitions`/`instances`). `version` is the first key; `yaml.dump(sort_keys=False)`.

**`add_node` signature change (Phase 1):** add `member_uid: str | None = None`, `membership: dict | None = None`. Mint a fresh `uuid.uuid4().hex[:12]` (decision-log item 10) only when `member_uid` is None; store on the `NodeRef`. **`save` must merge `ref.member_uid` and `ref.membership` into the persisted record** (serialized_state carries neither — verified `node.py ~445-452`).

### 1.3 One-shot migrate CLI

`python -m goofi.patch_format migrate <glob...>` (default `examples/*.gfi`). Idempotent (`version==2` ⇒ skip). **REQUIRED-FIX:** per-file try/except so one corrupt file is reported and skipped, not aborting the batch.

### 1.4 Control ops: full-FS browse

In `control.py` `_dispatch` (after existing save/load):

| op | payload | reply |
|---|---|---|
| `list_dir` | `{path?}` | `{entries[], parent, roots[]}` |
| `list_examples` | `{}` | `{entries[]}` |

`save` / `load` already exist (`control.py:217-234`); no backend change beyond `build_v2` in `Manager.save`. `_list_dir` runs in `run_in_executor`, one level non-recursive, dirs-before-files name-sorted, per-entry `OSError` skipped, hidden entries flagged. `roots = [Home, Examples, CWD]`.

```python
import goofi
EXAMPLES_DIR = Path(goofi.__file__).parents[1].parent / "examples"  # == manager.py:625 in checkout
```

**REQUIRED-FIX:** confirm `EXAMPLES_DIR` against an installed wheel; if `examples/` is not packaged, `list_examples` degrades to empty with a known marker and the Examples menu is documented checkout-only. Use one definition shared with `manager.py:625`.

No FS jail (locked decision 1). Path resolution delegates to `Manager.save` (`manager.py:481-491` handles dir-target/untitled/.gfi).

### 1.5 Component list (Phase 1)

- `src/goofi/patch_format.py` (NEW)
- `manager.py` load/save bodies; `add_node` uid/membership params; `_reject_reserved_name`
- `control.py` `list_dir`/`list_examples` ops + module helpers
- `examples/*.gfi` rewritten once by the CLI
- `frontend/src/lib/fs/FsBrowser.svelte` (NEW): modal save/load browser (breadcrumb + roots rail + editable path bar + filename field, save mode)
- `frontend/src/lib/editor/TopBar.svelte`: Save split-button (`Save ▾`: Save / Save As… / Save in browser); Load (FS browser) + Examples submenu + Upload fallback
- `frontend/src/lib/stores/graph.svelte.ts`: `listDir()`, `listExamples()`
- `frontend/src/lib/app/AppShell.svelte`: `triggerSave` fix

### 1.6 The "downloads untitled" bug fix

Root cause (verified): `AppShell.triggerSave` calls `g.save(undefined, true, ws.serialize())`, **discards the returned path**, and forces a Blob download.

```ts
async function saveBackend(path?: string): Promise<void> {
  const { path: saved } = await g.save(path ?? g.savePath ?? undefined, true, ws.serialize());
  g.savePath = saved;            // single source of truth — no new applySavedPath setter
  g.unsavedChanges = false;
}
function triggerSave(): void {
  if (g.savePath) void saveBackend();   // silent overwrite of the named patch
  else openFsBrowser('save');           // first save => Save-As
}
function saveInBrowser(): void { /* existing Blob download path, unchanged */ }
```

**REQUIRED-FIX:** do NOT add a third `applySavedPath` setter; set `g.savePath` directly from the returned `{path}` (or rely on existing `save_path_changed`/`applySnapshot`). Backend save failures now surface as real errors (a toast), no longer masked by the download fallback.

### 1.7 Z-order

`FsBrowser` renders in the top-level modal band ABOVE canvas, param panel, add-node menu, and the TopBar dropdown. Backdrop click + Esc cancels. One modal at a time.

---

## Phase 2 — Format v2 (recursive schema)

### 2.1 Document shape

```yaml
version: 2
definitions:                      # shared sub-patch graphs (SHARED instances reference these)
  def_ab12cd34:
    members:
      osc: {_type: Oscillator, category: inputs, params: {...}, gui_kwargs: {pos: [0,0]}}
      psd: {_type: PSD,        category: analysis, params: {...}, gui_kwargs: {pos: [180,0]}}
    links:
      - {node_out: osc, node_in: psd, slot_out: out, slot_in: data}
    interface:                    # manifest-mapping (decision #4)
      sig_in:  {dir: in,  inner_node: osc, inner_slot: data, dtype: ARRAY}
      spectrum:{dir: out, inner_node: psd, inner_slot: out,  dtype: ARRAY}
root:
  nodes:                          # top-level LEAF nodes (v1 shape + uid + membership)
    buffer0:
      uid: a1b2c3d4
      _type: Buffer
      category: signal
      params: {...}
      gui_kwargs: {pos: [40,60], viewers: {...}}
      membership: {instance: "", definition: null, local_name: buffer0}
  instances:                      # first-class sub-patch instances (decision #5/G4)
    subA:
      kind: shared                # shared | unique
      def: def_ab12cd34           # SHARED: ref. UNIQUE omits 'def', carries 'inline:<def graph>'
      gui_kwargs: {pos: [200,60]}
      members:                    # local_name -> {uid, gui_kwargs.pos}  (per-instance state ONLY)
        osc: {uid: 9f8e7d6c, gui_kwargs: {pos: [200,60]}}
        psd: {uid: 7c6b5a49, gui_kwargs: {pos: [360,60]}}
  links:                          # flat top-level links; boundary targets addressed as instance::slot
    - {node_out: "subA::spectrum", node_in: buffer0, slot_out: spectrum, slot_in: val}
layout: <opaque>
```

### 2.2 Schema rules

- **Discriminator** is at the `instances` map (not on leaf node records): a key under `root.instances` is an instance; a key under `root.nodes` is a leaf. **No `kind` key is read on a leaf record.** This sidesteps the discriminator-collision objection (a leaf with a stray `kind` param cannot be misread).
- **Reserved keys** `{version, definitions, kind, def, inline, interface, membership, uid}` are forbidden as node/param/group/slot names — loader-time assertion, parallel to the reserved `::` separator.
- **UNIQUE** instance: `inline:<graph>` (private copy), no `def`.
- **SHARED** instance: `def:<id>` into `definitions`; the only per-instance state is `gui_kwargs.pos` + the per-member `{uid, pos}` map. Topology/params live ONCE in the definition (strict mirror, decision #3).
- **Per-instance gui_kwargs derivation (REQUIRED-FIX):** the SHARED definition stores member-relative positions ONCE; the collapse-on-save derives per-instance member positions from the live nodes' `gui_kwargs` and must NOT leak per-member absolute positions into the shared definition, or strict-mirror breaks.
- **Parent links to a boundary slot** are stored as `{node_in: "subA::spectrum", slot_in: ...}` (instance::boundary). **REQUIRED-FIX:** the loader runs a manifest-resolution pass BEFORE `add_link`, resolving `subA::spectrum` to the leaf `(subA::psd, out)`; replaying the boundary form naively would `KeyError` in `add_link` (`manager.py:332-335`).
- **`interface.dtype`** is the goofi Data dtype string for the client-side soft link check.

### 2.3 Nesting

`::` is the segment separator. A member `core` inside instance `inner` inside instance `pipeline` flattens to `pipeline::inner::core`. Chained interface forwarding (outer boundary → inner boundary → leaf) resolves at flatten by walking until a leaf endpoint.

---

## Phase 2 — Runtime (flatten-at-runtime)

### 2.4 First-class manager state

Added in `Manager.__init__` (near `self.nodes` / `self._links`):

```python
SUBPATCH_SEP = "::"
@dataclass
class SubPatchDef:
    def_id: str
    members: dict[str, dict]   # local_name -> {_type, category, params, pos}
    links: list[dict]          # local-coord {node_out,node_in,slot_out,slot_in}
    interface: dict[str, dict] # boundary_slot -> {dir, inner_node, inner_slot, dtype}
@dataclass
class SubPatchInstance:
    inst_id: str
    def_id: str | None         # None => unique (inline graph held below)
    kind: str                  # 'shared' | 'unique'
    inline: SubPatchDef | None # set iff unique
    members: dict[str, str]    # local_name -> qualified display name (inst_id::local)
    member_uids: dict[str, str]# local_name -> member_uid (STABLE across respawn/reload)
    pos: list[float]

self._definitions: dict[str, SubPatchDef] = {}
self._instances:   dict[str, SubPatchInstance] = {}
self._membership:  dict[str, str] = {}   # qualified display name -> inst_id
self._refs_by_uid: dict[str, NodeRef] = {}
self._uid_by_name: dict[str, str] = {}   # display name -> member_uid
```

These are validated, persisted in v2, never re-derived from name prefixes (G4).

### 2.5 Identity model (three distinct ids)

- `node_id = f"{display_name}-{uuid8}"` (`manager.py:280`) — transport-only, **re-mints every spawn**, never a stable key.
- `display_name` — human label + container key + nd() sibling token; derived/namespaced for members; changes on group/expand/rename.
- `member_uid` — `uuid4().hex[:12]` (decision-log item 10), minted ONCE at first instantiation, **persisted**, carried across every respawn and reload. THE identity for membership/interface/data routing.

`ref_by_uid(uid)` / `name_for_uid(uid)` maintained on add/remove/respawn. **REQUIRED-FIX:** a uid-uniqueness pass on instantiate AND load detects collisions (now 48-bit per decision-log item 10, but the pass is retained regardless), re-mints, and remaps all membership/interface references before any spawn.

### 2.6 nd() resolution (CORRECTED — decision-log item 8: keep flat directory, allow cross-boundary)

**Scoped directories are NOT built.** Per the final decision, the flat global directory is kept
as-is (`node.py:355`, `manager.py:407-419` unchanged), which makes cross-boundary nd() references
work for free — a member can reference an out-of-sub-patch node by its flat name, and a top-level
node can reference a member by its qualified `instance::member` name. This is the convenient
behavior Phil wants, and it removes the highest-risk change from the design.

- **Resolution is global/flat.** `nd('name')` resolves `name` (or `instance::member`) against the
  one flat directory. No per-recipient scope prefix.
- **Group-time literal rewrite (the only handling needed):** when `group_nodes` namespaces members,
  it scans each member's param-expression sources; for any **string-literal** nd() argument that
  names a fellow member, it rewrites the literal to the qualified `instance::member` so existing
  intra-group references keep resolving. This is a narrow, safe AST/source transform on literals
  only.
- **Dynamic args** (variable / f-string nd() arguments) are left untouched and resolve globally;
  they may reference across boundaries, which is allowed.
- **Accepted minor quirk:** a bare `nd('gain')` inside a sub-patch whose sibling is `subA::gain`
  will resolve to a *top-level* node named `gain` if one exists (flat-directory shadowing). This is
  documented and accepted for now; users disambiguate with the qualified name. No validation error,
  no forbidding.
- **expand/ungroup** reverses the literal rewrite (qualified `instance::member` → bare `member`)
  so a round-tripped expression returns to its pre-group form.

> This replaces the earlier "forbid cross-boundary / build scoped directories" draft. It is
> strictly less code and removes the design's previously-#1 risk.

### 2.7 Namespacing, uniqueness, name budget

- Separator `::`, produced by no auto-namer (`f"{base}{idx}"`, `manager.py:272-276`) or example. **Enforced on EVERY name ingress** (auto-namer output, `add_node(name=)`, v1 load replay).
- **Uniqueness (G3):** the *instance prefix* is uniquified via the existing `name=None` auto-namer path applied to the instance label as a first-class uniqueness pass (NOT relying on the member container's auto-namer — verified it only iterates `self.nodes`). Each fully-qualified member name then uses `force_name=True`; a `KeyError` there is a real bug that aborts the transaction. Members are NEVER integer-suffixed (that would desync from the definition).
- **Name budget (G6) — REQUIRED-FIX, real template:**

```python
def _service_budget_ok(self, qualified_name: str, slots: list[str]) -> bool:
    iid = get_instance_id()                  # live, transport.py:54
    node_id = f"{qualified_name}-deadbeef"   # 8-hex placeholder
    longest_slot = max(slots, key=len, default="")
    candidates = [
        f"goofi.{iid}.data.{node_id}.{longest_slot}",
        f"goofi.{iid}.ctrl.{node_id}",
        f"goofi.{iid}.status.{node_id}",     # .status. is 2B longer than .data.
        f"goofi.{iid}.self.{node_id}",
    ]
    return max(len(c.encode()) for c in candidates) <= 255
```

Computed over the FULL nested namespaced name (all ancestor prefixes) and the member's real output slots, for ALL members, BEFORE the first spawn. Raises `SubPatchTooDeep` early.

### 2.8 Incremental API (operates on a NON-empty manager — G3)

```python
def group_nodes(self, member_names: list[str], interface_spec: dict,
                *, shared: bool = False, pos=(0,0)) -> str           # -> inst_id
def expand_instance(self, inst_id: str) -> list[str]
def ungroup(self, inst_id: str) -> list[str]                          # unique alias
def instantiate_definition(self, def_id: str, pos=(0,0), *, shared=True) -> str
def make_unique(self, inst_id: str) -> str                            # -> new def_id
def edit_shared_param(self, inst_id, local, group, name, value) -> None
def apply_topology_delta(self, def_id: str, delta: TopoDelta) -> None
def build_v2_tree(self) -> dict                                       # collapse-for-save
```

**`group_nodes`** snapshots live member state (no respawn). **REQUIRED-FIX:** ALL validation runs BEFORE any mutation — members exist and `serialized_state` is non-None (mirror `manager.py:502-503`; handle `serialization_pending` `node_helpers.py:257-259`), manifest validated against `ref.input_slots`/`ref.output_slots`, name-budget pre-flight, fan-in pre-scan, `::` lint — so a failure leaves the graph byte-untouched. Members are **renamed in place** (not respawned) so `node_id` and node_id-keyed data wires stay valid (G2).

**`expand_instance`/`ungroup`** inline members back into the parent namespace via rename; no respawn.

**`instantiate_definition`** spawns fresh members under a transaction.

### 2.9 Rename helper (REQUIRED-FIX — atomic, all name-keyed state)

`NodeContainer.rename(old, new)` re-keys `_nodes` only and is **insufficient alone**. A `Manager._rename(old, new)` wrapper atomically rewrites:
- `NodeContainer._nodes` key,
- every `self._links` endpoint (`node_out`/`node_in`),
- `self._node_groups` (keyed by name — breaks `_same_group`/in-process transport selection otherwise),
- `self._membership`, `self._uid_by_name`, `self._refs_by_uid` (uid stable; name remap),
- re-broadcasts the node directory,
- **fires `on_node_renamed(old, new)` to the bridge** so `control.py` discards `old` from `_wired_nodes`, re-registers the `STATE_UPDATE`/`PROCESSING_ERROR` handlers under the live ref, emits a `node_renamed` event, and the DataHub re-keys its `(uid, slot)` bookkeeping. Without this, a renamed node's state/errors reach the editor under the wrong identity (verified `control.py` closures capture the name at wire time).

### 2.10 Transaction (G5 — corrected)

```python
class _Splice:
    def __init__(self, mgr): self.mgr=mgr; self.done=[]
    def add_node(self, **kw):
        n = self.mgr.add_node(notify_gui=False, **kw); self.done.append(("add_node", n, kw)); return n
    def add_link(self, l):
        # record the displaced wire (if any) BEFORE add_link tears it down
        displaced = self.mgr._wire_on_input(l["node_in"], l["slot_in"])
        self.mgr.add_link(notify_gui=False, **l); self.done.append(("add_link", l, displaced))
    def remove_node(self, name):
        spec = self.mgr._node_spec(name)            # full record for restore
        self.mgr.remove_node(name, notify_gui=False); self.done.append(("remove_node", spec))
    def remove_link(self, l):
        self.mgr.remove_link(notify_gui=False, **l); self.done.append(("remove_link", l))
    def rollback(self):
        for kind, *rest in reversed(self.done):
            ... # add_node->remove_node; add_link->remove_link + RESTORE displaced wire;
                # remove_node->re-add_node from spec; remove_link->re-add_link
```

```python
@contextlib.contextmanager
def _transaction(self):
    sp = _Splice(self)
    snap = (copy.deepcopy(self._definitions), copy.deepcopy(self._instances),
            dict(self._membership), list(self._links), copy.deepcopy(self._node_groups))
    try:
        yield sp
        self._broadcast_node_directory()
        self._bridge.control.on_subpatch_changed()   # ONE consolidated event on success
    except Exception:
        sp.rollback()
        (self._definitions, self._instances, self._membership,
         self._links, self._node_groups) = snap        # deep restore
        # REQUIRED-FIX: re-broadcast a corrected snapshot so bridge side-channels converge
        self._bridge.control.broadcast_snapshot()
        raise
```

**REQUIRED-FIXES folded:** deep-copy `_definitions`/`_instances`/`_node_groups` (in-place `.members`/`.interface` mutations must roll back); snapshot+restore `self._links` and `self._node_groups`; journal removes with full specs so remove-before-add deltas restore; record and restore the wire `add_link` displaces (`manager.py:351-359`); suppress per-node events with `notify_gui=False` and emit one `on_subpatch_changed`; re-broadcast a corrected snapshot after rollback.

### 2.11 Data plane (G1 + G2 — opaque uid route)

- **Route change** `server.py:209`: `web.get("/data/by-uid/{uid}/{slot}", self.data.handler)`. `uid` is hex8 (URL-safe); slot is a plain identifier. No separator ever enters a URL segment.
- `data.py` resolves the **current** ref via `manager.ref_by_uid(uid)` at (re)subscribe time; unknown uid ⇒ close 4004.
- **Multiplexed fan-out (REQUIRED-FIX):** DataHub keeps ONE `set_data_handler(slot, _demux)` per `(uid, slot)` that fans out to a set of `_SlotForwarder`s. `set_data_handler(slot, None)` is called only when the LAST forwarder for that `(uid, slot)` closes. This fixes the single-callback-evicting bug for both multiple viewers and respawn re-wire.
- **Respawn (G2):** `instantiate` (fresh spawn) AND group/expand (rename) emit a control event the client uses to force re-subscribe by uid. On a fresh-spawn respawn, the manager detaches the demux handler from the dead ref and the DataHub re-attaches to `ref_by_uid(uid)`'s new ref — **server-side teardown is tied to `member_uid`, not a captured NodeRef instance** (the finally block must target the current ref to avoid leaking a `REGISTER_SUBSCRIBER` on the new node). Accepted brief gap (latest-wins).
- Frontend `data.ts`: URL `.../data/by-uid/${uid}/${encodeURIComponent(slot)}`; subs keyed by `${uid} ${slot}`; `onRespawn(uid)` force-reconnect hook driven by the control event. Control snapshot (`schemas.py`) gains `"uid": ref.member_uid`; viewers subscribe by `info.uid`, never by name.

### 2.12 Boundary rewrite (G7 — fully-resolved leaf endpoints)

At flatten/group, each parent link to `inst::boundary` resolves through the interface (walking chained boundary→boundary→leaf across nesting depths) to a single `(leaf_qualified_name, leaf_slot)`. **REQUIRED-FIX:** the one-wire-per-input dedup runs on the FULLY chained-resolved leaf endpoints, not direct parent links only (artifact (d) shows multi-hop forwarding that can collapse two boundary slots onto one inner input). Two parent links (or two boundary slots across depths) resolving to the same inner input is a hard error, not a silent `add_link` teardown.

### 2.13 Strict mirror (Phase 2b)

- **Param edit (cheap, no respawn):** `edit_shared_param` updates the definition (source of truth) then propagates via per-sibling `update_param` (`node_helpers.py:287`). **REQUIRED-FIXES:** make it transactional and tolerant — guard each `update_param` against missing group/param (raises otherwise, `node_helpers.py:288-291`) and against un-pushed/mid-respawn siblings (skip-and-record; they re-converge from definition params on next spawn); roll back already-applied siblings on hard failure (all-or-nothing).
- **Param drift at source — NOT handled (decision-log item 9, temporary).** A node that self-modifies a param during `process()` makes its live `serialized_state.params` diverge from the definition. We deliberately do **not** add reconcile-from-sibling or a determinism-validation pass. The definition remains the save source-of-truth (self-edits are not persisted); user edits propagate; whatever cross-instance effect the simple mirror produces is accepted. This is explicitly temporary: it becomes a non-issue once nodes can no longer self-edit params. `make_unique`'s deepcopy fork is therefore "good enough" rather than provably drift-free, which is acceptable under this decision.
- **Topology edit:** `apply_topology_delta` edits the definition, then diff-splices (`remove links → remove nodes → add nodes → add links`) into every instance inside ONE transaction, journaling removes for rollback. Boundary resolution recomputed on any boundary inner-node replacement.
- **make_unique:** fork SHARED → private `inline` deepcopy; pure bookkeeping, zero data interruption (live nodes already byte-identical under strict mirror).
- **GC:** `instances_of(def_id)`; drop a definition only when zero instances reference it; orphan defs dropped at save.

### 2.14 Collapse-for-save (`build_v2_tree`)

Reads first-class membership/instances/definitions, NOT name prefixes (G4). Roots = nodes with `_membership.get(name)` is None. SHARED → `{def, gui_kwargs, members:{local:{uid,pos}}}`; UNIQUE → inline graph. Member-relative positions stored once in the shared def; per-instance positions derived from live nodes. `version:2` always. Merges `ref.member_uid` + membership into each record (serialized_state has neither).

---

## Identity & namespacing summary

- Stable id = `member_uid`; persisted; plumbed through `add_node`/`load`/respawn.
- `/data/by-uid/{uid}/{slot}` route (G1 fixed); resolve-by-uid (G2 fixed); multiplexed fan-out.
- `::` reserved separator, enforced on all ingress; reserved discriminator key set.
- 255-byte budget over the real four-service template, full nested name, real slots (G6 fixed).

---

## Migration

- One-shot converter rewrites `examples/*.gfi` to v2 (Phase 1: flat-wrapped; the flat→v2 envelope is forward-compatible with the recursive expander, which only populates `definitions`/`instances` when sub-patches exist).
- Always write v2 on save. v1 (no `version` key) still loads via `normalize_loaded`. Per-file try/except in the CLI batch.

---

## Gotcha resolutions

- **G1** — opaque `/data/by-uid/{uid}/{slot}`; `::` never enters a URL.
- **G2** — uid-keyed data plane; rename-not-respawn for group/expand keeps `node_id` stable; fresh-spawn respawn detaches/re-attaches server-side by uid + forces client re-subscribe; group/expand ALSO force re-subscribe (display name in URL would have changed — corrected from the original "no churn" claim).
- **G3** — incremental ops on a non-empty manager; instance-prefix uniqueness pass; `force_name=True` on qualified members; `::` provably never auto-generated, enforced on ingress.
- **G4** — `_definitions`/`_instances`/`_membership`/`_refs_by_uid` are explicit validated persisted state.
- **G5** — `_transaction` deep-snapshots all mutable state (incl. `_links`, `_node_groups`), journals adds AND removes AND displaced wires, rolls back across all instances, re-broadcasts corrected snapshot.
- **G6** — `_service_budget_ok` over all four service templates, full nested name, real slots, before first spawn.
- **G7** — dedup/validate on fully chained-resolved leaf endpoints; collision = hard error.

---

## Testing plan

**Phase 1**
- `normalize_loaded`: no version ⇒ v2; v2 ⇒ identity; v3/non-int ⇒ ValueError.
- `flat_view` raises on non-empty `definitions`/`instances`.
- Every `examples/*.gfi` normalizes to v2 with non-empty `root.nodes`; v1 and v2 fixtures load into fresh managers with identical node-name sets + link counts.
- `_migrate_file` idempotent (byte-identical second run); CLI skips a corrupt file and continues.
- Initial-load layout broadcast fires for a client connected during load.
- `list_dir` (dirs-before-files, roots present, parent correct, file path ⇒ its dir); `list_examples` count == 10 post-migration.
- `triggerSave` with `savePath` set calls backend save and creates NO `<a download>`; with null opens FsBrowser save.
- Save-As writes to a chosen dir; TopBar path + unsaved-dot update from the returned path.

**Phase 2a (UNIQUE)**
- `group_nodes` on a 2-node chain ⇒ one instance, two `::`-qualified membership entries, `self.nodes` count unchanged; each member's `node_id` byte-identical before/after (rename-not-respawn, G2).
- A node keeps the SAME `member_uid` across group→expand; `ref_by_uid` returns the new ref with a new `node_id`, old uid.
- An open `/data/by-uid/{uid}/{slot}` WS keeps delivering after expand; two tabs on one slot both receive frames (multiplex).
- All `group_nodes` validation failures (None serialized_state, bad manifest, budget overflow, fan-in, `::` lint) leave the graph byte-untouched.
- `expand_instance` restores flat names, removes instance + membership, links point at renamed nodes, data still flows.
- `_reserve_names`/uniqueness pass: `{filt, filt0}` ⇒ instance prefix bumps; `force_name` never KeyErrors.
- Budget pre-flight raises `SubPatchTooDeep` for an over-deep nested name; nothing spawned; `/dev/shm/iox2_*` count unchanged.
- nd() sibling resolves post-group via group-time literal rewrite (`nd('gain')` → `nd('subA::gain')`); expand reverses it; cross-boundary nd() resolves globally (allowed); a dynamic nd() arg is left untouched.
- Reserved `::` rejected at top-level `add_node(name=)` and v1 load replay.

**Phase 2b (SHARED)**
- Param edit on instance A propagates to B via `update_param` with NO add/remove (assert pids unchanged); both emit `state_update`; definition reflects the new value.
- `edit_shared_param` with a mid-respawn sibling skips-and-records; on hard failure all siblings roll back (all-or-nothing).
- Topology add-member splices into every instance; a forced `wait_for_state` timeout leaves every instance + definition byte-identical (full rollback incl. displaced-wire restore).
- `make_unique`: pids unchanged, `kind=='unique'`, `def_id` None, deepcopy; subsequent edits don't touch former siblings.
- GC drops a def only when its last instance is removed.
- Respawn re-wire: subscribe `/data`, topology-edit that respawns the node, assert data resumes after re-attach, no JS error, no producer leak.
- Boundary rewrite: exactly one wire per input after a boundary inner node is replaced; chained `top_out` resolves to the deepest leaf; two boundary slots onto one inner input ⇒ load error.
- `build_v2_tree` round-trip: save→load→save identical YAML for one shared + one unique instance; shared def stores member-relative positions, per-instance positions derived (no leak).
- v1 file loads; saving re-emits `version:2`.

---

## Implementation phase plan

Phase 1 (ships independently, on current flat format):
1. Add `src/goofi/patch_format.py` with `normalize_loaded`/`_v1_to_v2`/`build_v2`/`flat_view` (with the non-empty definitions/instances hard-fail guard) + unit tests `tests/test_patch_format.py`. Independently testable.
2. Rewrite `Manager.load` to go through `normalize_loaded`+`flat_view`; KEEP the initial-load layout broadcast; add the `::`-reserved-name ingress guard. Rewrite `Manager.save` to always emit v2 via `build_v2`. Add `member_uid`/`membership` params to `add_node` (mint-if-absent) and merge them into the save record. Test: v1+v2 load equivalence into a fresh manager.
3. Add `python -m goofi.patch_format migrate` CLI (idempotent, per-file try/except). Run once on `examples/*.gfi`. Test: idempotent re-run is byte-identical.
4. Add `list_dir`/`list_examples` control ops + module helpers (executor, no jail, roots=Home/Examples/CWD; confirm EXAMPLES_DIR on an installed wheel). Test the RPC shapes.
5. Frontend: `FsBrowser.svelte` modal (breadcrumb/roots/path bar/filename); TopBar Save split-button + Load/Examples/Upload; `triggerSave` fix (stop discarding path, set g.savePath from return, no Blob on normal Save). E2e: Save-As + Load round-trip, no untitled download.

Phase 2a (UNIQUE/inline sub-patch — working testable core, sharing OFF):
6. Add first-class state dataclasses (`SubPatchDef`/`SubPatchInstance`) + `_definitions`/`_instances`/`_membership`/`_refs_by_uid`/`_uid_by_name` + `ref_by_uid`/`name_for_uid`; uid-uniqueness pass on add/load.
7. Switch the data plane to `/data/by-uid/{uid}/{slot}`: route change, resolve-by-uid in data.py, multiplexed per-(uid,slot) fan-out in DataHub, snapshot/schemas expose uid, frontend data.ts subscribes by uid + onRespawn hook. Test multiplex + respawn re-subscribe before any sub-patch op exists.
8. Add `Manager._rename` (atomic: _nodes, _links, _node_groups, membership, uid indices, directory rebroadcast, bridge on_node_renamed re-wire). Test rename keeps node_id, state/errors reach editor under new identity, data keeps flowing.
9. Add `_service_budget_ok` (real four-service template, nested name, real slots) + `_reject_reserved_name` ingress enforcement everywhere.
10. Add `_transaction`/`_Splice` (deep snapshot of all mutable state incl. _links/_node_groups; journal adds+removes+displaced wires; suppress per-node events; corrected snapshot on rollback).
11. Implement `group_nodes` (all-validation-before-mutation; rename-in-place) and `expand_instance`/`ungroup`. Control ops + single on_subpatch_changed event. Implement group-time literal nd() rewrite (no scoped directories; cross-boundary nd() allowed) and its inverse on expand. Boundary rewrite on fully chained-resolved leaf endpoints with G7 dedup.
12. `build_v2_tree` collapse-for-save for UNIQUE (inline) instances; replace the Phase-1 `flat_view` guard with the recursive expander on load. Round-trip tests for a single unique instance + 2-deep nesting.

Phase 2b (SHARED ref-tracking on top of 2a):
13. `definitions` store + SHARED instance encoding; `instantiate_definition` (fresh spawn under _transaction; force client re-subscribe by uid).
14. `edit_shared_param` (transactional, tolerant of missing-group/un-pushed/mid-respawn siblings; all-or-nothing rollback). No param-drift reconciliation/validation (decision-log item 9).
15. `apply_topology_delta` (diff-splice into all instances, one transaction, journaled removes, boundary recompute) + `make_unique` (bookkeeping fork) + GC.
16. Respawn re-wire for /data on topology respawn (server-side detach/re-attach by uid). Strict-mirror end-to-end tests + collapse-for-save (member-relative positions in def, derived per-instance positions).
---

## Top risks (post-decision)

The decision log removed the previously-#1 risk (scoped nd() directories). The remaining
load-bearing risks, in priority order:

1. **Transaction completeness.** `add_link` silently tears down a displaced wire
   (`manager.py:351-359`) and `remove_node` terminates real processes irreversibly. The
   `_Splice`/`_transaction` machinery must deep-snapshot `_links`/`_node_groups`/`_definitions`/
   `_instances`, journal removes with full specs, and restore displaced wires — or a failed
   splice corrupts the live graph despite "state restored". Must be tested with injected
   mid-splice failures + `/dev/shm/iox2_*` leak checks.
2. **Data-plane single-callback eviction is a real *current* bug** (`node_helpers.py:405-431`):
   `set_data_handler` closes the previous subscriber, so two viewers on one slot already clobber
   each other today. The DataHub per-`(uid, slot)` multiplex MUST be built — the migration fixes
   this bug, it does not introduce it.
3. **Rename fan-out.** `Manager._rename` must rewrite `_nodes`, `_links`, `_node_groups`,
   `_membership`, uid indices, rebroadcast the directory, and re-wire the bridge's name-captured
   `STATE_UPDATE`/`PROCESSING_ERROR` closures — or group/expand silently break control + transport.
4. **255-byte iceoryx2 ServiceName overflow at deep nesting.** `_service_budget_ok` must use the
   real four-service template (`.status.` is 2B longer than `.data.`), the full nested name, and
   real slot names, checked before the first spawn, so overflow is an early clean error rather
   than a late mid-splice crash + SHM leak.
5. **Phase-1 `flat_view` hard-fail.** If the guard on non-empty `definitions`/`instances` is
   omitted, a real v2 sub-patch file (once Phase 2 writes them) would load as a partial flat graph
   with no error before the recursive expander ships — silent data loss across the 1→2 boundary.

## Deferred / out of scope (recorded, not chosen)

- **Device auth on the bridge.** The full-FS browser + `0.0.0.0` bind trust the LAN; recognized-
  device auth is explicit future work, not in this spec.
- **Removing node self-param-editing.** Decision-log item 9 is temporary and assumes a future where
  nodes cannot self-write params; that change is out of scope here.
- **Reusable-definition library UI / palette.** Shared definitions exist in the format+runtime, but
  a browsable cross-patch library of sub-patches is a later layer.
