# Finish Loose Ends — Convergence Implementation Plan (2026-06-22)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:test-driven-development for every task.
> Executed **inline, full-context** by the orchestrating session. Each task specifies the failing
> test, the interfaces, and acceptance; the minimal implementation is discovered via Red→Green→Refactor
> (do NOT pre-write production code — Iron Law). Steps use checkbox (`- [ ]`) tracking.

**Goal:** Converge the landed branch stack into a finished state — implement every undone spec feature,
close all 25 backlog loose ends from the 2026-06-22 deep-dive, validate end-to-end, leave `main`
untouched and `frontend` updated.

**Architecture:** Work proceeds on branch `feat/finish-loose-ends` (off `frontend@92fd3d9`). Six
dependency-ordered phases, each independently testable. Phase 1 is the headline perf rework (viewer
adapters, reverting A0); Phases 2–5 close correctness/UX loose ends; Phase 6 is verification +
merge-back into `frontend`. TDD throughout; one focused commit per task.

**Tech Stack:** Python 3.12 (aiohttp bridge, numpy, iceoryx2 transport — *out of scope to modify*),
pytest; SvelteKit 5 + TypeScript strict, Vitest, Playwright e2e (gitignored, boots a real Manager).

## Global Constraints

- **TDD Iron Law:** no production code without a failing test first. Watch every test fail for the
  right reason before implementing.
- **Out of scope (do NOT touch):** `src/goofi/transport.py`, `src/goofi/codec.py`, the iceoryx2
  setup (CLAUDE.md §13). If a task appears to need them, STOP and surface to the user. (This blocks
  backlog #16 `take_latest_view` — documented, not implemented.)
- **`main` stays untouched.** All work lands on `feat/finish-loose-ends`, merges to `frontend` only.
- **No-regression gate per task:** the new test(s) pass; `pytest tests/` stays green (currently
  **950 passed, 7 skipped**); `cd frontend && npm run check` clean; `npm test` (vitest, **121**)
  green. e2e built before running.
- **Test commands:**
  - Backend: `rm -f /dev/shm/iox2_*; .venv/bin/python -m pytest tests/ -p no:cacheprovider -q`
  - FE units: `cd frontend && npm test` · typecheck `npm run check`
  - e2e: `(cd frontend && npm run build); .venv/bin/python -m pytest e2e/ -q` (plain `grep`/`ls`, not git — e2e is gitignored)
- **Commit style:** repo log convention, `type(scope): summary (backlog #N)`; end with the
  Co-Authored-By trailer. Commit per task. Do not push without authorization.

---

## Phase 1 — Viewer adapters + A0 revert (backlog #3; THE perf rework)

> Implements `docs/superpowers/specs/2026-06-21-viewer-adapters-design.md`. Nodes process pure float
> again; per-viewer-kind adapters in the bridge convert float `Data` → uint8 (image) / float16 (line/
> trajectory/topomap) / passthrough (string/table) at the viewer boundary, with `meta.__view__` carrying
> float range/stats so range and the metadata inspector stay float-accurate. Reverses A0 (uint8 nodes)
> and A1 (verbatim forward — the bridge now decodes once per slot and re-encodes once per kind/frame).

**File structure:**
- Create: `src/goofi/bridge/adapters.py` — `ADAPTERS` registry, per-kind `adapt(Data)->Data`, stats helper.
- Modify: `src/goofi/bridge/data.py` — kind-grouped forwarders, decode-once/adapt-per-kind, route arg.
- Modify: `src/goofi/bridge/server.py:209` — route `/data/{node}/{slot}/{kind}`.
- Modify: nodes — revert A0: producers emit float (`videostream.py`, `loadfile.py`,
  `imagegeneration.py`, `edgedetector.py`); consumers drop coercion (`colorenhancer.py`,
  `hsvtorgb.py`, `rgbtohsv.py`, `poseestimation.py`); cv2/mediapipe coerce internally only.
- Modify: `src/goofi/image_utils.py` — retained; consumers become the adapter layer.
- Modify (frontend): `frontend/src/lib/codec/` decode (add `<f2`), `api/data.ts` + `api/dataWorker.ts`
  (thread `kind`), `viewers/ImageViewer.svelte`, `viewers/ArrayViewer.svelte`,
  `viewers/TrajectoryViewer.svelte`, `viewers/TopomapViewer.svelte`, high-dim fallback, `viewers/kind.ts`.
- Replace: `tests/test_image_nodes_uint8.py` → assert float emit.
- Create tests: `tests/test_view_adapters.py`, `frontend/src/lib/codec/decodeFloat16.test.ts`.

### Task 1.1 — Adapter registry + stats helper (backend, pure)
**Files:** Create `src/goofi/bridge/adapters.py`; Test `tests/test_view_adapters.py`.
**Interfaces produced:** `adapt(data: Data, kind: str) -> Data`; `ADAPTERS: dict[str, Callable]`;
`view_stats(arr) -> dict` (`{min,mean,max}` on float). `meta["__view__"]` per spec wire contract.
- [ ] Test: RGB float image (`ndim==3`, ch3, `[0,1]`) → `adapt(.,"image")` returns `uint8`, no
  normalization (preserves colour), `meta.__view__.range==[fmin,fmax]`, `.stats` from float.
- [ ] Test: grayscale float (`ndim==2`) → uint8 normalized `[fmin,fmax]→[0,255]`; flat image
  (`fmax==fmin`) guarded by epsilon (no NaN).
- [ ] Test: line float32 1D/2D → `adapt(.,"line")` returns `float16`; `.stats` computed on float32
  pre-downcast (exact).
- [ ] Test: `adapt(.,"string")` and `adapt(.,"table")` passthrough (dtype unchanged, no `__view__`).
- [ ] Test: non-renderable (`ndim>3`) any kind → summary frame, no array body,
  `meta.__view__.summary == {shape,dtype,min,mean,max}` from float.
- [ ] Test: unknown kind → `"raw"` fallback returns `data` unchanged.
- [ ] Implement minimal `adapters.py` to green; refactor stats helper to dedupe. Commit.

### Task 1.2 — A0 revert: producers/consumers emit & accept float
**Files:** Modify the 8 node files above; Replace `tests/test_image_nodes_uint8.py` →
`tests/test_image_nodes_float.py`.
**Interfaces:** node `process()` output image dtype is float (`float32`/`float64`), values in `[0,1]`
for producers; consumers accept float and emit float.
- [ ] Test (per producer): `videostream`/`loadfile`/`imagegeneration`/`edgedetector` emit float image
  Data (dtype kind `f`), not uint8.
- [ ] Test (per consumer): `colorenhancer`/`hsvtorgb`/`rgbtohsv`/`poseestimation` given float image
  produce float output; internal cv2/mediapipe coercion does not leak uint8 to the output slot.
- [ ] Run new tests → fail (current code emits uint8). Implement reverts to green. Keep cv2/mediapipe
  coercion *internal* (coerce to uint8 for the C call, convert result back to float for the slot).
- [ ] Delete old `test_image_nodes_uint8.py`; ensure full `pytest tests/` green. Commit.

### Task 1.3 — Data plane: kind route + decode-once/adapt-per-kind forwarders
**Files:** Modify `src/goofi/bridge/data.py`, `src/goofi/bridge/server.py:209`; Test
`tests/test_datahub_mux.py` (extend) + `tests/test_dataplane_adapt.py` (new).
**Interfaces consumed:** `adapt`, `ADAPTERS` (1.1). **Produced:** route `/data/{node}/{slot}/{kind}`;
`_SlotMux` registers `set_data_handler(slot, on_frame, raw=False)` (decoded `Data`); forwarders grouped
by kind; `bytes_k = encode_data(adapt(data, kind))` memoized once per (kind, frame).
- [ ] Test: two forwarders of different kinds (`image`, `line`) on one slot each receive their own
  representation (uint8 vs float16) from a single decoded frame; decode happens once.
- [ ] Test: N forwarders of the same kind share one adapt/encode (memoization — assert adapt called
  once per kind/frame, e.g. via a spy/counter).
- [ ] Test: latest-wins backpressure preserved per forwarder.
- [ ] Implement to green; update `data.py` module docstring (no longer "verbatim"). Commit.

### Task 1.4 — Frontend codec: float16 decode
**Files:** Modify `frontend/src/lib/codec/` decoder; Test `frontend/src/lib/codec/decodeFloat16.test.ts`.
**Interfaces:** decoder handles `<f2` body → upcast to `Float32Array` via `DataView` (no reliance on
`Float16Array`). uint8 path already exists.
- [ ] Test: a GOOF frame with `<f2` array body decodes to the correct `Float32Array` values
  (half→float reference values, incl. subnormal/zero/negative).
- [ ] Implement half-float read to green. Commit.

### Task 1.5 — Frontend data plumbing: thread `kind`
**Files:** Modify `frontend/src/lib/api/data.ts`, `frontend/src/lib/api/dataWorker.ts`; Test
`frontend/src/lib/api/data.test.ts` (extend or new).
**Interfaces:** `subscribeData(node, slot, kind, cb)`; worker `sub`/`unsub` carry `kind`; WS URL
`/data/<node>/<slot>/<kind>`; ref-count key is `(node, slot, kind)`.
- [ ] Test: `subscribeData` with kind builds the `/data/<node>/<slot>/<kind>` URL and ref-counts per
  `(node,slot,kind)`; switching kind unsubscribes the old, subscribes the new.
- [ ] Implement to green. Commit.

### Task 1.6 — Viewers read `meta.__view__`
**Files:** Modify `ImageViewer.svelte`, `ArrayViewer.svelte`, `TrajectoryViewer.svelte`,
`TopomapViewer.svelte`, high-dim fallback, `viewers/kind.ts`/`viewerSettings`.
**Interfaces consumed:** float16 frames (1.4), `meta.__view__.{range,stats,summary}` (1.1/1.3).
- [ ] Test (vitest, pure helpers where possible): ImageViewer colormap window uses
  `meta.__view__.range` (float), not 0–255; grayscale reconstructs `fmin+(u/255)*(fmax-fmin)`.
- [ ] Test: high-dim fallback renders from `meta.__view__.summary`; absent `__view__` falls back to
  computing from the received array (defensive).
- [ ] Test: Array/Trajectory/Topomap auto-range uses `meta.__view__.stats`.
- [ ] Implement to green; subscribe each viewer with its resolved kind. Commit.

### Task 1.7 — Phase-1 e2e: representation correctness
**Files:** `e2e/tests/viewers.spec.ts` (extend) or new `e2e/tests/adapters.spec.ts`.
- [ ] e2e: image viewer shows the float range (not 0–255); a line stream renders from a float16 frame;
  high-dim stats reflect float values. Build FE first. Commit.
- [ ] Mark `docs/.../2026-06-21-viewer-adapters-design.md` Status → "implemented (2026-06-22)".

---

## Phase 2 — Sub-patch correctness (backlog #1, #2, #8, #5, #17)

### Task 2.1 — nd() string-literal rewrite on group/expand (#1, HIGH)
**Files:** Modify `src/goofi/manager.py` (`group_nodes` ~:629, `expand_instance`); Test
`tests/test_manager.py` (new `test_group_rewrites_nd_cross_refs`).
**Interface:** on group, rewrite string-literal `nd('member')` args naming a fellow grouped member →
`nd('instance::member')`; inverse on expand.
- [ ] Test: two members where A's param expression contains `nd('B')`; after `group_nodes([A,B])`,
  A's expression resolves to `subpatchK::B` and the live cross-ref still works; after `expand`, it is
  restored to `nd('B')`.
- [ ] Implement best-effort literal rewrite (string scan over expression params for the grouped member
  names) to green. Commit.

### Task 2.2 — `_transaction`/`_Splice` rollback for multi-node ops (#2, HIGH, large)
**Files:** Modify `src/goofi/manager.py` (add `_transaction` ctx + journaled splice; wrap
`group_nodes`, `expand_instance`, `instantiate_definition`, `add_member_node` sibling-mirror,
`wire_boundary` re-splice, `remove_instance`); Test `tests/test_manager.py`
(`test_failed_splice_rolls_back`).
**Interface:** `with self._transaction(): ...` snapshots `_links/_node_groups/_definitions/_instances/
_membership`; on exception restores them and tears down any processes spawned during the txn.
- [ ] Test: inject a failure partway through `instantiate_definition` (e.g. a forced raise after the
  first add_node); assert the graph (`_links`, `_instances`, `_membership`, live node set) is
  byte-identical to before the call and no orphan process remains.
- [ ] Test: `group_nodes` failure mid-rename rolls back names + displaced wires (the `add_link`
  displacement at manager.py:351-359 is restored).
- [ ] Implement the snapshot/journal/restore seam to green; route the mutating ops through it. Commit.

### Task 2.3 — Strict-mirror surfaces failures + converges (#8)
**Files:** Modify `src/goofi/manager.py:1093-1165` (replace `except Exception: pass`); Test
`tests/test_manager.py` (`test_shared_mirror_surfaces_failure`).
- [ ] Test: a shared family where one sibling's `update_param` mirror fails → the failure is surfaced
  (bridge `error` event / raised, per chosen contract) rather than silently swallowed; the family does
  not silently diverge.
- [ ] Implement: capture per-sibling failures, surface them, attempt convergence/retry (or tie into
  2.2 transaction). Commit.

### Task 2.4 — uid-stable data subscription + onRespawn rebind (#5)
**Files:** Modify `src/goofi/bridge/data.py`/`server.py` (accept uid-keyed lookups / expose a stable
id), `frontend/src/lib/api/data.ts` (rebind on respawn/rename); Test `tests/test_datahub_mux.py` +
`frontend/src/lib/api/data.test.ts`.
> NOTE: Phase 1 made the route `/data/<node>/<slot>/<kind>` (display-name). This task adds an
> `onRespawn(uid)`/rename force-reconnect so an open viewer rebinds when a group/expand renames its
> node, instead of silently dropping. Keep the `<kind>` segment from Phase 1.
- [ ] Test: after a node rename (group), an open subscription is re-pointed and continues receiving
  frames (no permanent drop).
- [ ] Implement rebind hook to green. Commit.

### Task 2.5 — Sub-patch instance viewer-state persistence (#17)
**Files:** Modify `src/goofi/bridge/control.py:232-243` (`set_node_viewers` no longer accept-and-ignore
for instances), `src/goofi/manager.py` v2 serialize (persist per-instance viewer kind/settings); Test
`tests/test_patch_format.py` / `tests/test_control_ops.py`.
- [ ] Test: set a viewer kind on a collapsed sub-patch instance's output slot, save, reload → the kind
  persists.
- [ ] Implement persistence into the v2 envelope under the instance to green. Commit.

---

## Phase 3 — Node hygiene §K/§L (backlog #6, #7, #14)

### Task 3.1 — Deepcopy-meta sweep across the §K cluster (#6)
**Files:** Modify `src/goofi/nodes/.../psd.py:119`, and the §K cluster (Padding, Reduce, Join, Hilbert
sfreq, Normalization int-truncation); Test `tests/test_nodes.py` / new `tests/test_meta_aliasing.py`.
**Interface:** nodes that add/modify nested meta keys deepcopy first (match `fft.py:51`).
- [ ] Test: drive `psd` on a fan-out (producer + sibling consumer of the same source `Data`); assert
  the producer's `meta['channels']` is NOT mutated after psd runs (no shared-dict aliasing).
- [ ] Implement `deepcopy(data.meta)` in psd and the cluster to green; add per-node assertions. Commit.

### Task 3.2 — I/O node terminate hygiene + unblock delay (#7)
**Files:** Modify `sharedmemout.py`, `outputs/zeromqout.py`, `inputs/zeromqin.py` (add `terminate()`
freeing shm/sockets), `inputs/delay.py:35` (move `time.sleep` off the processing thread); Test
`tests/test_nodes.py` (terminate releases resource) + a delay responsiveness test.
- [ ] Test: `sharedmemout` after `terminate()` releases its `/dev/shm` segment (no leak); zeromq nodes
  close sockets.
- [ ] Test: a node wrapping `delay` remains responsive to `terminate()` during its delay window.
- [ ] Implement to green. Commit.

### Task 3.3 — `_service_budget_ok` uses real slot names (#14, trivial)
**Files:** Modify `src/goofi/manager.py:297-307`; Test `tests/test_manager.py`.
- [ ] Test: a node whose real output slot name is long is checked against its actual slots (not a
  synthetic template); a >48-char real slot is rejected pre-spawn.
- [ ] Implement to green. Commit.

---

## Phase 4 — Undo/redo fidelity (backlog #9, #10, #19, #20, #21, #22, #23)

### Task 4.1 — Surface `history.lastError` (#9)
**Files:** Modify a toast/notification surface + `history.svelte.ts` reader wiring; Test
`frontend/src/lib/stores/history.test.ts` + a component/e2e check.
- [ ] Test: a failing undo sets `lastError` AND a subscriber/toast renders it (currently no reader).
- [ ] Implement toast wiring to green. Commit.

### Task 4.2 — Multi-select drag = one undo step (#10)
**Files:** Modify `frontend/src/lib/panels/NodeEditorPanel.svelte:594-602` (wrap drag-stop loop in
`history().transaction()`); Test `frontend/src/lib/stores/...` (one entry for N moved nodes).
- [ ] Test: moving N selected nodes records a single coalesced history entry; one Ctrl+Z restores all.
- [ ] Implement transaction wrap to green. Commit.

### Task 4.3 — Undo highlight-pulse + `awaitEvent` (#19)
**Files:** Modify `frontend/src/lib/workspace/navContext.ts:42-65` + a `.undo-flash` style + `awaitEvent`
helper; Test unit for `awaitEvent`, e2e for the flash.
- [ ] Test: `awaitEvent(pred, timeout)` resolves on predicate / rejects on timeout; restored node gets
  `.undo-flash` class after the async echo.
- [ ] Implement to green. Commit.

### Task 4.4 — load_patch redo restores layout (#20)
**Files:** Modify `frontend/src/lib/stores/graph.svelte.ts:514` + `graphExecutors.ts:269-273`
(capture-after on `graph_replaced`); Test `graphExecutors.test.ts`.
- [ ] Test: redo of a load restores the prior layout (afterLayout no longer hardcoded null).
- [ ] Implement capture-after to green. Commit.

### Task 4.5 — make_unique undo restores prior shared definition (#21)
**Files:** Modify `frontend/src/lib/stores/graphExecutors.ts:262` + backend `restore_node`/
`duplicate_shared` (`control.py:308`) to reconstruct the recorded definition; Test backend + executor.
- [ ] Test: undo of make_unique restores the exact prior shared definition identity.
- [ ] Implement backend restore support + executor inverse to green. Commit.

### Task 4.6 — NavContext restore fallbacks (#22)
**Files:** Modify `frontend/src/lib/workspace/navContext.ts:42-65` (fall back to any node-editor panel /
pop to nearest valid depth); Test `navContext.test.ts`.
- [ ] Test: undo after a structural change where the original panel/path no longer resolves focuses a
  valid panel/depth (not a no-op).
- [ ] Implement fallbacks to green. Commit.

### Task 4.7 — Multi-node paste/duplicate/delete = one undo step (#23)
**Files:** Modify `frontend/src/lib/panels/NodeEditorPanel.svelte:771,796` + `graph.svelte.ts:793-866`
(wrap in `history().transaction()`); Test executor/store.
- [ ] Test: paste/duplicate/delete of N nodes records a single history entry (one Ctrl+Z). (Spec
  decision-log(3) allowed N entries; the user asked to finish loose ends → coalesce to one.)
- [ ] Implement transaction wraps to green. Commit.

---

## Phase 5 — UX completion (backlog #11, #12, #13, #24, #25)

### Task 5.1 — Examples menu UI (#11, small, high ROI)
**Files:** Modify `frontend/src/lib/editor/TopBar.svelte` + `frontend/src/lib/fs/FsBrowser.svelte`
(examples mode) wiring `graph.svelte.ts:658 listExamples()`; Test FsBrowser/TopBar unit + e2e.
- [ ] Test: an Examples affordance lists `examples/*.gfi` from `listExamples()` and loading one calls
  `Manager.load` with that path.
- [ ] Implement to green. Commit.

### Task 5.2 — fps / data-rate / drop HUD (#12)
**Files:** Modify `frontend/src/lib/editor/TopBar.svelte:46-47` + a per-viewer rate counter in the data
worker/frames layer; Test unit for the rate counter, e2e screenshot.
- [ ] Test: the rate counter computes frames/sec and drop count from the data stream; HUD renders it.
- [ ] Implement to green. Commit.

### Task 5.3 — TableViewer recursive expandable tree (#13)
**Files:** Modify `frontend/src/lib/viewers/TableViewer.svelte:15-33`; Test
`frontend/src/lib/viewers/TableViewer.*.test.ts`.
- [ ] Test: nested TABLE Data renders an expandable tree (nested fields reachable), not the flat
  `{N fields}` collapse.
- [ ] Implement recursive tree to green. Commit.

### Task 5.4 — Errors panel with stack traces (#24)
**Files:** Modify `frontend/src/lib/panels/register.ts` (new `errors` panel) + an ErrorsPanel component
replacing the truncated floating chip; Test unit + e2e (click → focus node).
- [ ] Test: error nodes are listed with full stack traces; click focuses the node (focus path exists).
- [ ] Implement to green. Commit.

### Task 5.5 — Crashed-node restart + edge re-targeting + empty-state (#25)
**Files:** Modify `NodeEditorPanel.svelte` (respawn action on errored node; wire Svelte Flow
`onreconnect`/`edgesReconnectable`); blank-canvas hint component; Test unit + e2e.
- [ ] Test: errored node exposes a respawn action that calls the restart op; an edge can be
  re-targeted (onreconnect) without delete+redraw; blank canvas shows the first-run hint.
- [ ] Implement to green. Commit.

---

## Phase 6 — Verification & merge-back (backlog #4, #18, + cleanup)

### Task 6.1 — Real HD-video fps e2e (#4, explicitly requested)
**Files:** `e2e/tests/stress.spec.ts` (rewrite) / new; uses `test.gfi` (Oscillator + PSD + 8 Buffers +
VideoStream).
- [ ] Test: load `test.gfi`, open ≥10 visible viewers incl. the HD VideoStream, run 60s, assert a
  meaningful median frame-time threshold (hardware-GL target, or a documented software-GL-adjusted
  one). Replaces the synthetic-viewer 45ms placeholder. Commit.

### Task 6.2 — HD zero-copy perf assertion in a CI lane (#18)
**Files:** `tests/test_dataplane_perf.py:74` (un-gate or add a perf lane marker); doc note.
- [ ] Decide: run the timing test in a perf lane or document the structural
  `test_raw_forward_does_zero_codec_work` sibling as the CI guard. Implement/annotate. Commit.
- [ ] NOTE the adapters rework (Phase 1) changes the forward path from verbatim → decode/adapt/encode;
  update/replace the A1 raw-forward perf assertions accordingly so they reflect the new architecture.

### Task 6.3 — Clean the 14 svelte-check warnings + scope notes
**Files:** the 8 files flagged by `npm run check` (state_referenced_locally lints).
- [ ] Fix the reactive-reference warnings (derived/closure) so `npm run check` is 0 warnings; document
  backlog #15 (OffscreenCanvas, large — optional after adapters) and #16 (`take_latest_view`,
  transport.py out-of-scope) as explicitly deferred with rationale. Commit.

### Task 6.4 — Full validation + merge to frontend
- [ ] `pytest tests/` green; `npm run check` clean; `npm test` green; `(cd frontend && npm run build)`
  then `pytest e2e/ -q` green (re-run flaky e2e individually per memory `e2e_flaky_tests`).
- [ ] Update `CLAUDE.md` §14 goal-condition status if warranted; mark this plan's tasks complete.
- [ ] `git checkout frontend && git merge --ff-only feat/finish-loose-ends` (or `--no-ff` marker if the
  user prefers); delete `feat/finish-loose-ends`. Leave `main` untouched. Confirm with the user before
  the final merge.

---

## Self-review (spec coverage)

- Viewer-adapters spec → Phase 1 (1.1 adapters/stats, 1.2 A0 revert, 1.3 data plane, 1.4 float16
  decode, 1.5 plumbing, 1.6 viewers, 1.7 e2e). ✓ All spec sections mapped.
- Persistence spec gaps → Phase 2 (nd() §2.6, transaction §2.10, mirror §2.8, uid-route §2.11,
  viewer-state). ✓
- Deep-analysis §K/§L → Phase 3. ✓
- Undo spec §5.2/§5.3/§6.1/§6.3 + re-entrancy → Phase 4. ✓
- §6 feature inventory gaps (Examples §6.5, Table §6.4, Errors §6.9, HUD §7) → Phase 5. ✓
- §7 perf acceptance → Phase 6.1. ✓
- Out-of-scope (transport.py #16) → documented in Global Constraints + 6.3. ✓
