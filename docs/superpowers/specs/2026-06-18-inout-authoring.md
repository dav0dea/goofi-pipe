# Spec: In/Out boundary node authoring

2026-06-18, branch `feat/persistence-subpatch`. Designed via a 3-proposal →
judge → adversarial-critique workflow. The critique found 3 blockers in the
naive design; this spec folds in the fixes + one simplification that dissolves
most of them.

## Model (canonical-interface, extended in place)

The per-instance `interface` dict stays the single source of truth (it already
round-trips verbatim through build_v2_tree/_expand_doc/snapshot). Each entry is
extended:

```
interface[bnd_id] = {
  dir: "in" | "out",
  dtype: "ARRAY" | "STRING" | "TABLE",   # set when wired (inner slot is truth)
  inner_node: <local> | None,            # None = UNWIRED (the new state)
  inner_slot: <slot>  | None,
  pos: [x, y],                           # In/Out pill position inside the view
}
```

- `bnd_id` is a stable per-instance id minted by `_fresh_boundary_id` by
  **scanning existing keys** (`in0/in1/out0…`); the auto/group path keeps its
  `"<local>.<slot>"` key (no test churn). Never recomputed from inner_*, so
  rewiring can't orphan external wires.
- In/Out nodes are **purely virtual** — never a NodeRef / in `self.nodes` /
  `_links`. They exist only as interface entries.

### Key simplification — ONE boundary per inner (node, slot)
`wire_boundary`/`add_boundary` reject a second boundary on an inner slot already
exposed. This dissolves the critique's two display/collision blockers:
- **Output fan-out** is achieved on the *external* side (one Out port → many
  consumers; outputs fan out naturally) — never two Out pills on one inner slot
  — so `drawEndpoint`'s inner-slot reverse-lookup stays unambiguous.
- **Input fan-in**: an In port maps one inner input (single-source); a second
  external wire to it evicts the first via `add_link`'s existing input-slot
  eviction (which emits `link_removed`, so it's visible — not silent).

Collapsed-node ports render for **wired boundaries only**; unwired pills show
only inside the entered view (so there's no dangling non-connectable handle).

## Backend
- **Aliasing fix (blocker #1):** deep-copy `interface` at every fan-out/detach
  point — `instantiate_definition`, `_definition_from_instance`, `_expand_doc`
  (shared), `make_unique` — and assign fresh entry dicts on edit (the
  `set_node_pos` discipline). Otherwise sibling boundary edits cross-mutate the
  whole shared family.
- New `@mark_unsaved_changes` methods: `add_boundary(inst,dir,dtype,pos)->bnd_id`
  (unwired); `wire_boundary(inst,bnd_id,inner_node|None,inner_slot|None)` (set/
  clear single inner target; dtype guard via `DataType.name`; re-splice external
  links on rewire; reject nested-instance inner_node — v1 wires to real members
  only); `remove_boundary(inst,bnd_id)` (drop + un-splice its external links);
  `set_boundary_pos(inst,bnd_id,pos)->changed[]` (def-first mirror; returns the
  dragged + sibling (inst,bnd) for broadcast).
- Shared: boundary topology (dir/dtype/inner/pos) is **def-mirrored** to all
  siblings; external wires stay **per-instance** (re-splice/un-splice runs per
  sibling against its own members + links; `membership.get(other)!=inst` defines
  "external", handling nesting).
- `_derive_interface` (group auto-create) emits the new fields (dtype from the
  live ref slot; default beside-member pos); auto boundaries are born wired.
- `remove_node` defensive unwire — scoped: only for a member of a still-present
  **unique** instance (skip during `remove_instance` teardown by popping
  membership first; shared single-member delete is out of scope).
- Re-splice scan branches on `dir` (IN matches node_in/slot_in; OUT matches
  node_out/slot_out) and scopes "external" to non-members of this instance.

## Bridge (splice lives here, like the remove/pos instance special-casing)
- `resolve(inst,bnd_id)` → inner (member display, slot); raises a clear
  "boundary not wired" error (not KeyError) if unwired.
- `add_link`/`remove_link` branches translate an instance+bnd_id endpoint to the
  inner member flat link before calling the manager. Persisted shape is the same
  flat `osc.out → inst::member.slot` that group/expand already round-trip.
- New ops: `add_boundary`, `wire_boundary`, `remove_boundary`, `set_boundary_pos`.
- New lightweight `boundary_moved {inst,bnd,pos}` event (one per changed sibling,
  like `node_moved`); other mutations reuse `subpatch_changed`.

## Frontend
- `control.ts`: extend `SubPatchPort` (dtype?, nullable inner_*, pos?); add
  `boundary_moved`; a 6-type pseudo-set (InArray/InString/InTable/OutArray/
  OutString/OutTable).
- `graph.svelte.ts`: addBoundary/wireBoundary/removeBoundary/setBoundaryPos RPCs;
  handle `boundary_moved` (update interface[bnd].pos in place).
- `AddNodeMenu`: inside a sub-patch, prepend the 6 boundary types (category
  `boundary`); picking one → `addBoundary` at the drop pos. Omitted at top level.
- `NodeEditorPanel`: render pills from `inst.interface` incl. unwired (entry.pos,
  entry.dtype; edge to inner only when wired); pills selectable/draggable/
  deletable. onConnect: inside + boundary endpoint → `wireBoundary` (soft dtype
  check); top-level + instance endpoint → `add_link` as-is (bridge splices).
  Drag a pill → `setBoundaryPos`; Delete a pill → `removeBoundary`; delete a
  wired In→member edge → `wireBoundary(...,null,null)`.
- `SubPatchNode`: collapsed ports = wired boundaries only.
- `BoundaryNode`: dimmed/dashed "unwired" state.

## Decisions taken (sensible defaults, match shipped behavior)
1. **One boundary per inner slot** (the simplification above).
2. **Boundary pos mirrors** across shared siblings (consistent with member-pos
   strict-mirror).
3. **Deleting a pill with external wires** silently tears them down (the port no
   longer exists) — flagged as destructive.
4. **Nesting**: v1 wires In/Out to real member nodes only (not nested instance
   ports); legacy interface-only `.gfi` self-heal on next save.
