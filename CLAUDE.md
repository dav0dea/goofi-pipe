# goofi-pipe

A real-time, node-based data-processing platform for biosignals (EEG, ECG,
audio, video). Users build **patches** in a browser node-graph: each node is a
process that ingests, transforms, or emits `Data` objects; edges carry data
between nodes' output and input slots. The platform targets live, high-rate
streams (kHz EEG, HD video) with many simultaneous viewers.

This file is the orientation for a Claude session working in this repo. Read it
end-to-end before touching code, then read the specific subsystem you're
changing. **The "How we work" section is not optional — it is how changes are
expected to be made here.**

---

## How we work (read first)

These are hard expectations, in priority order. They override speed.

1. **Test-driven development.** No production code without a failing test first.
   Write the test, watch it fail for the right reason, write the minimal code to
   pass, then refactor green. This is the Iron Law for new behavior, bug fixes,
   and refactors alike. A bug fix starts with a test that reproduces the bug.
   - Pure logic (codec, reducers, geometry, stores' core) is unit-tested directly.
   - Svelte component/rune glue can't mount in vitest — verify it by typecheck +
     an `e2e/` Playwright test, and keep the testable logic in a `.ts`/`.svelte.ts`
     module that *is* unit-tested.

2. **Root cause before fix.** Investigate until you understand *why* something
   breaks — read the error, reproduce it, trace the data flow to its origin.
   Fix the source, not the symptom. If three fixes fail, the architecture is
   wrong; stop and reconsider it rather than piling on a fourth patch.

3. **Structural edits over shallow hacks.** Prefer the change that makes the
   codebase *correct by construction* over the one that silences the symptom.
   When two code paths should agree, unify them at one source of truth instead of
   duplicating a workaround in both. A larger, well-reasoned refactor is welcome
   when it removes a class of bugs — the user does not gate refactor scope.

4. **Deep code analysis.** Before changing a subsystem, hold enough of it in
   context to reason about the change's blast radius. Trust documented internal
   contracts; verify the ones you're about to depend on. Skim the relevant spec
   in `docs/superpowers/specs/` — most subsystems have one.

5. **Minimum diff, maximum clarity.** Match the surrounding code's idiom, naming,
   and comment density. Comments explain *why*, not *what*. Don't reformat code
   you aren't changing. **Never run Prettier on this repo** — there is no config
   and its defaults fight the codebase's tabs + single-quotes style; hand-match.

6. **Honest reporting.** If tests fail, say so with the output. If a step was
   skipped, say that. State what is verified plainly; don't claim done what you
   haven't run.

### Audit-driven hardening (multi-agent)

The proven way to harden a subsystem — or the whole codebase — is a **broad,
top-model audit run to convergence**, not a single read-through. Use the Workflow
tool: fan *finders* across subsystem dimensions in parallel, then **adversarially
verify every candidate** before believing it.

- **Top model only.** Every finder *and* verifier runs on the most capable model
  (Opus, or Fable when available) — never Haiku/Sonnet, even for cheap breadth; a
  weaker finder under-finds. The `Explore` agent type pins Haiku in its frontmatter,
  so pass `model: 'opus'` to override it (or drop `agentType`). Verify the run's
  reported per-agent model is never haiku/sonnet.
- **Verify is a gate, not a rubber stamp.** Each finding earns an explicit verdict:
  - *correctness* → `real && reachable`. The hard lesson: a verifier readily confirms
    "this path crashes" but misses "**is it reachable**" — so it must trace a real
    caller (bridge RPC routing, the supervisor thread, frontend events, a normal node
    tick) and **check the upstream guards**, not just the local function.
  - *leanness* → `safe && !falsePositive && !servesArchitecture && !overAbstraction`.
    Treat the system as an architecture with a purpose; cut **inflation** (dead code,
    duplication, parallel paths) but **reject reshapes / speculative abstractions** —
    over-abstraction is itself inflation. Before calling anything dead, clear this
    codebase's *dynamic dispatch*: node auto-discovery (the AST catalog in
    `registry.py` — specs, not imports; multiple Node classes per file; underscore
    modules skipped), string-keyed RPC ops, the graph-executor + viewer-kind
    registries, the `window.goofi`/`$lib/agent` façade + the **gitignored** `e2e/`
    (plain `grep`, not `git grep`), Svelte-template usage, `Data`/codec meta string keys.
- **Iterate to convergence, don't chase zero.** Re-run after each fix round.
  Convergence shows as the confirmed count shrinking *and* shifting from structural to
  trivial (e.g. 16→4, or 11→9→4→3). An adversarial finder always surfaces *something*
  marginal — stop when only trivial/over-abstraction items remain; don't manufacture
  churn. Re-auditing also catches regressions a fix itself introduced.
- **Fix under the Iron Law.** Each confirmed finding is fixed TDD-first (or with a
  characterization test where coverage was missing), behind green suites, committed in
  small focused steps. Verifiers reliably *correct* finder over-reaches (a dropped
  guard, a mischaracterized return, an order-changing hoist) — trust the corrected
  proposal, not the finder's.

---

## Architecture

Three layers, connected by two transports.

```
   ┌──────────────── browser ────────────────┐
   │  SvelteKit SPA  (frontend/)              │
   │   · Svelte Flow node editor (zoom/pan)   │
   │   · viewers (uPlot / canvas / WebGL)     │
   │   · dockable workspace panels            │
   │   · undo/redo, params, sub-patches       │
   └───────┬───────────────────────┬──────────┘
           │ HTTP + /control WS     │ /data/<node>/<slot>/<kind> WS
           │ (JSON RPC + events)    │ (binary GOOF frames)
   ┌───────▼───────────────────────▼──────────┐
   │  goofi.bridge  (src/goofi/bridge/)        │
   │   aiohttp server in the manager process   │
   │   serves the built SPA + the two planes   │
   └───────┬───────────────────────────────────┘
           │ Python calls
   ┌───────▼───────────────┐   iceoryx2 (zero-copy SHM)
   │  Manager + NodeRefs    │◄──────────────────────────┐
   │  (graph, links, .gfi)  │   ctrl pub / status sub    │
   └────────────────────────┘                            │
   ┌─────────────────────────────────────────────────────▼─┐
   │  Node processes (1 per node, or shared process group)  │
   │   each runs its own tick loop; publishes Data to SHM    │
   └────────────────────────────────────────────────────────┘
```

- **Backend is Python, process-per-node**, orchestrated by the **manager**, which
  owns the graph and persists patches as `.gfi` YAML. Nodes talk to each other
  over **iceoryx2** shared memory (zero-copy publish). The backend transport and
  wire format are stable; treat them as fixed (see Hard constraints).
- **The bridge** lives in the manager process and exposes the manager's API +
  live data to the browser over HTTP/WebSocket. It serves the built SPA from
  `frontend/build/`.
- **The frontend** is the only UI (the old dearpygui GUI is gone). One manager ↔
  one browser tab.

### Control plane — `/control` WS
JSON, bidirectional. Client sends RPCs (`add_node`, `add_link`, `update_param`,
`group_nodes`, `save`, `load`, …); server broadcasts events (`hello`,
`state_update`, `node_added`, `graph_replaced`, `error`, …). The browser is never
authoritative — it issues RPCs and reconciles from the echoed events. Undo/redo
replays inverse/forward RPCs; the backend never learns "undo" exists.

### Data plane — `/data/<node>/<slot>/<kind>` WS
One binary WS per (node, slot, viewer-kind) a client is viewing. The bridge
decodes each slot **once**, runs the **viewer adapter** for each subscribed kind
(image→uint8, line/trajectory/topomap→float16, string/table passthrough), and
re-encodes once per kind — float range/stats ride in `meta["__view__"]` so
viewers and the metadata inspector stay float-accurate. Latest-wins backpressure
(drop oldest) mirrors iceoryx2. See `docs/superpowers/specs/2026-06-21-viewer-adapters-design.md`.

---

## Running, testing, building

```bash
# Backend + bridge (serves the prebuilt SPA, prints the URL to open):
uv run goofi-pipe                      # launches manager + bridge
uv run goofi-pipe --headless test.gfi  # no UI; run a patch
uv run goofi-pipe --headless --duration 5 test.gfi   # auto-stop
#   flags: --port N (default 8000), --bind HOST (default 127.0.0.1)

# Backend tests (must stay green; ~990 pass):
.venv/bin/python -m pytest tests/

# Frontend (run from frontend/):
npm run dev      # Vite dev server; proxies /control + /data to the bridge
npm run test     # vitest (unit)
npm run check    # svelte-check + tsc strict — keep 0 errors
npm run build    # static SPA → frontend/build/  (what the bridge serves)

# e2e (frontend/../e2e/, GITIGNORED): Playwright + a real Manager, driven via
# window.goofi. Boots the full stack; use plain grep (not git grep) to find refs.
```

If `/dev/shm/iox2_*` accumulates after a crash:
```bash
.venv/bin/python -c "import iceoryx2 as i; i.Node.try_cleanup_dead_nodes(i.ServiceType.Ipc, i.config.global_config())"
```

The reference stress-patch shape is Oscillator + PSD + 8 Buffers (+
VideoStream); build one via the UI or the agent façade and save it — patches
are gitignored, so don't assume a `test.gfi` exists on disk.

Node authoring: plain top-level imports for ALL deps (never defer into
`setup()`; no import guards). Config hooks (`config_*`) may reference only
builtins, `goofi.params`/`DataType`/slot types, and same-file static
declarations — the AST registry execs them without importing the module
(runtime state like device lists goes behind a try/except fallback there).
Multiple `Node` classes per file are fine; type names are globally unique.
A missing dep greys the palette entry; a broken one fails the node's
bootstrap and surfaces on its error channel.

---

## Backend map (`src/goofi/`)

| file | owns |
|---|---|
| `transport.py` | iceoryx2 `Publisher`/`Subscriber`/`Listener`/`Notifier`/`WaitSet` + a thread variant. **Stable — do not touch.** |
| `codec.py` | the binary `Data` wire format (12-byte header, msgpack meta, dtype body). **Stable.** Mirrored in `frontend/src/lib/codec/`. |
| `data.py` | the `Data` object (dtype, value, meta) + meta conventions (`channels`, coords). |
| `params.py` | `Float/Int/Bool/StringParam` descriptors + serialization. |
| `node.py` | the node base: tick `_processing_loop`, slots, SHM publish, ctrl handling. |
| `node_helpers.py` | `NodeRef` — the manager-side proxy: ctrl pub/notifier (+ pre-ready ctrl queue), status sub, lifecycle `stage`, the per-NodeRef data pump that decodes slot frames for the bridge. |
| `registry.py` | the AST node catalog: `NodeSpec` + `build_catalog`. **The manager never imports node modules** — config hooks are exec'd from the AST in a whitelisted namespace (`__import__` blocked); implementations import only in the node's child process. Purity contract enforced at build + by `tests/test_registry_parity.py`. |
| `manager.py` | `Manager` + `NodeContainer`: graph, `_links`, async spawn/teardown (add returns immediately; stages creating→setup→ready/error), save/load, sub-patch runtime (group/expand/share, `_instances`/`_definitions`), bridge bootstrap. |
| `node_log.py` | per-node SSE log server (peer-to-peer; the proven template for the future P2P data plane). |
| `patch_format.py` | `.gfi` v2 (recursive, sub-patch-aware) build/expand. |
| `bridge/server.py` | aiohttp HTTP + WS server; routes; static SPA serving. |
| `bridge/control.py` | `/control` RPC dispatch + state/event broadcast. |
| `bridge/data.py` | `/data` plane: `_SlotMux` decode-once + per-kind adapt/encode fan-out. |
| `bridge/adapters.py` | the viewer adapters (float→uint8/float16, `__view__` stats). |
| `bridge/fsbrowse.py` | filesystem browse RPC for save/load. |
| `bridge/schemas.py` | request/response + snapshot shapes. |
| `nodes/` | the node library (analysis, array, inputs, misc, outputs, signal). |

## Frontend map (`frontend/src/lib/`)

| dir | owns |
|---|---|
| `api/` | transport clients: `control.ts` (RPC + events), `data.ts`/`dataWorker.ts` (binary stream, off-thread decode), `frames.ts` (rAF paint coalescer + per-slot latest frame), `perfStats`/`rateMeter` (fps HUD), `awaitEvent`. |
| `codec/` | the TS port of `codec.py` (`decode.ts`, incl. float16). |
| `stores/` | reactive state (Svelte 5 runes): `graph.svelte.ts` (server-authoritative graph mirror), `history.svelte.ts` + `graphExecutors.ts` (unified undo/redo), `selection`, `ui`, `console`, `flash`, `logStream`. |
| `editor/` | the Svelte Flow canvas: `GoofiNode.svelte` (every node, incl. sub-patch instances — **one component, no per-kind branches**), `snap.ts` + `nodeMetrics.ts` (alignment snapping), placement, boundary nodes. |
| `viewers/` | one component per viewer kind (`ArrayViewer`, `ImageViewer`, `TopomapViewer`, `TrajectoryViewer`, `StringViewer`, `TableViewer`) + `ViewerFeed` (subscribe lifecycle), `kind.ts`, `decimate.ts`, `imageGL.ts`. |
| `params/` | parameter widgets + expression editor. |
| `panels/` | dockable panel content (node-editor, parameters, viewer, metadata, console, errors) + the panel registry. |
| `workspace/` | the panel layout engine: `model.ts` (pure tree algebra), `workspace.svelte.ts`, `navContext.ts` (undo focus restore). |
| `fs/` | the filesystem browser for save/load. |
| `agent/` | the automation façade (`window.goofi`: commands + query) — the seam e2e drives, and the basis for a planned in-app AI-agent panel. |

---

## Key subsystems & their specs

Most non-trivial subsystems have a design doc in `docs/superpowers/specs/`. Read
the relevant one before changing the area.

- **Undo/redo** (`2026-06-19-undo-redo-redesign-design.md`) — one history stack
  spanning the graph domain (replayed as inverse/forward RPCs) and the layout
  domain (restored as `WorkspaceState` snapshots). `history().transaction()` folds
  N records into one entry. Each action carries a `NavContext` so undo reorients
  to where the change happened.
- **Sub-patches** (`2026-06-17-persistence-subpatch-design.md`,
  `2026-06-18-virtual-subpatch-nodes.md`) — group/expand/share; a sub-patch
  instance is a **virtual node** that renders through `GoofiNode` with no
  special-casing (its wired boundaries are its slots; its sharing/expand controls
  live in the inspector). Shared instances strict-mirror across siblings. v2
  `.gfi` is recursive. Invariant: the synth node `nodeByName(instId)` returns a
  **stable reference** when unchanged, like a real node.
- **Viewer adapters** (`2026-06-21-viewer-adapters-design.md`) — the data-plane
  reduction described above.
- **Per-viewer view state** (`2026-06-18-per-viewer-instance-view-state-design.md`)
  — each slot's chosen viewer kind + settings, persisted into the `.gfi`.
- **In/Out authoring** (`2026-06-18-inout-authoring.md`) — sub-patch boundary
  ports.
- **Node discovery & instantiation**
  (`2026-07-06-node-discovery-instantiation-design.md`) — the AST registry
  (`registry.py`), child-side imports, async `add_node` with lifecycle stages
  (creating→setup→ready/error), pre-ready ctrl queue, bootstrap-error pipe
  (no restart loop), node-authoritative `param_options` reconciliation (the
  seam the planned param-refresh feature reuses).

Analysis reports live in `docs/analysis/`. The performance ceiling and a future,
more aggressive data plane are tracked in **§ Future** below.

---

## Hard constraints

- **Do not touch** `transport.py`, `codec.py`, or the iceoryx2 setup unless
  truly unavoidable (and surface it first). The node↔node zero-copy publish path
  is load-bearing.
- **Leave `main` alone.** Work happens on the `frontend` branch. Don't push or
  force-push without authorization; branch before committing on a default branch.
- No auth on the WS endpoints — single-user, local/trusted-LAN app.
- Desktop browser only (no mobile/touch). One theme, done well (no dark-mode toggle).
- Don't reintroduce dearpygui or zmq.
- Commit messages end with: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- Commit in small, focused, readable steps — not one mega-commit.

---

## Future: P2P data plane + node-side thalamus (designed, not built)

`docs/p2p-data-thalamus-spec.md` is an implementation-ready, self-contained spec
for the **next major architecture step**: move the
viewer-data path **peer-to-peer** (browser connects directly to the node
process), and reduce each stream **inside the node** to exactly what the viewer
can display (per-axis: envelope for waveforms, area for images, subsample for
channels) on a dedicated reducer thread — removing the manager from the data
path entirely.

It **supersedes** the shipped viewer-adapters plane (its §9.1 deletes
`bridge/data.py`): the adapters do a bridge-side dtype downcast with the manager
still transcoding; the thalamus does node-side capacity reduction P2P (~1300× for
a kHz buffer) and directly removes the measured perf ceiling. It is a clean
10-step plan, not a rebase — undertake it as its own project when prioritized.
