# goofi-pipe

A real-time, node-based data-processing platform for biosignals (EEG, ECG,
audio, video). Users build **patches** in a browser node-graph: each node ingests,
transforms, or emits `Data` objects; edges carry data between nodes' output and
input slots. The platform targets live, high-rate streams (kHz EEG, HD video)
with many simultaneous viewers.

**This branch (`rust-rewrite`) is a ground-up Rust rewrite of the backend.** The
original Python implementation is *deleted from this branch* — it lives on `main`
and, for reference, at `../../goofi-pipe/`. The frontend (SvelteKit) carried over
and is the only UI.

This file is the orientation for a Claude session working in this repo. Read it
end-to-end before touching code, then read the specific subsystem you're changing.
**The "How we work" section is not optional — it is how changes are expected to be
made here.**

---

## How we work (read first)

These are hard expectations, in priority order. They override speed.

1. **Test-driven development.** No production code without a failing test first.
   Write the test, watch it fail for the right reason, write the minimal code to
   pass, then refactor green. This is the Iron Law for new behavior, bug fixes,
   and refactors alike. A bug fix starts with a test that reproduces the bug.
   - Rust: unit tests in-module (`#[cfg(test)]`), cross-crate behavior in
     `tests/` (the bridge's `protocol.rs` drives the real WS surface).
   - Svelte component/rune glue can't mount in vitest — verify it by typecheck +
     a `tests/e2e/` Playwright test, and keep the testable logic in a `.ts`/`.svelte.ts`
     module that *is* unit-tested.

2. **Root cause before fix.** Investigate until you understand *why* something
   breaks — read the error, reproduce it, trace the data flow to its origin.
   Fix the source, not the symptom. If three fixes fail, the architecture is
   wrong; stop and reconsider it rather than piling on a fourth patch.

3. **Structural edits over shallow hacks.** Prefer the change that makes the
   codebase *correct by construction* over the one that silences the symptom.
   Condense by making errors **compile-time-impossible** (typed extraction, serde,
   a shared schema, an unconstructible invalid state) rather than by adding
   defensive runtime handling — but keep genuine boundary errors: a Python node
   *can* raise, so propagate it, never panic. When two code paths should agree,
   unify them at one source of truth instead of duplicating a workaround in both.
   A larger, well-reasoned refactor is welcome when it removes a class of bugs —
   the user does not gate refactor scope.

4. **Deep code analysis.** Before changing a subsystem, hold enough of it in
   context to reason about the change's blast radius. Trust documented internal
   contracts; verify the ones you're about to depend on. Skim the relevant spec in
   `docs/superpowers/specs/` (**gitignored** — present on disk, not in git).

5. **Minimum diff, maximum clarity.** Match the surrounding code's idiom, naming,
   and comment density. Comments explain *why*, not *what*. Don't reformat code
   you aren't changing. Rust is **4 spaces**; the frontend is **tabs + single
   quotes**. There is no rustfmt.toml and no Prettier config — **never run
   Prettier**, and hand-match style instead.

6. **Zero warnings.** A task is not done at "Finished" — grep the build output for
   `warning:` (dead_code, unused, clippy) and clear it. Remove the dead field or
   function; do not silence it with a `_` prefix or an `#[allow]`.

7. **Honest reporting.** If tests fail, say so with the output. If a step was
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
  so pass `model: 'opus'` to override it (or drop `agentType`).
- **Verify is a gate, not a rubber stamp.** Each finding earns an explicit verdict:
  - *correctness* → `real && reachable`. The hard lesson: a verifier readily confirms
    "this path crashes" but misses "**is it reachable**" — so it must trace a real
    caller (bridge RPC dispatch, the tick thread, a frontend event, a normal node
    tick) and **check the upstream guards**, not just the local function.
  - *leanness* → `safe && !falsePositive && !servesArchitecture && !overAbstraction`.
    Cut **inflation** (dead code, duplication, parallel paths) but **reject reshapes /
    speculative abstractions** — over-abstraction is itself inflation. Before calling
    anything dead, clear this codebase's *dynamic dispatch*: `inventory`-registered
    native nodes, string-keyed RPC ops, the Python introspection probe (a manifest, not
    an import), the `window.goofi`/`$lib/agent` façade, the **committed** `tests/e2e/`,
    Svelte-template usage, and `Meta`/codec string keys.
- **Iterate to convergence, don't chase zero.** Re-run after each fix round.
  Convergence shows as the confirmed count shrinking *and* shifting from structural to
  trivial (e.g. 16→4, or 11→9→4→3, or 7→10→7→0). An adversarial finder always surfaces
  *something* marginal — stop when only trivial/over-abstraction items remain; don't
  manufacture churn. Re-auditing also catches regressions a fix itself introduced.
- **Fix under the Iron Law.** Each confirmed finding is fixed TDD-first (or with a
  characterization test where coverage was missing), behind green suites, committed in
  small focused steps. Verifiers reliably *correct* finder over-reaches (a dropped
  guard, a mischaracterized return, an order-changing hoist) — trust the corrected
  proposal, not the finder's.

---

## Architecture

One Rust process holds the graph, the engine, and the web server. Node execution
is tiered; the browser is a read-only replica driven by commands.

```
   ┌──────────────── browser ────────────────┐
   │  SvelteKit SPA  (frontend/)              │
   │   · Svelte Flow node editor (zoom/pan)   │
   │   · viewers (uPlot / canvas / WebGL)     │
   │   · dockable workspace panels            │
   │   · undo/redo, params, sub-patches       │
   └───────┬───────────────────────┬──────────┘
           │ /control WS            │ /data/<node>/<slot> WS
           │ JSON commands +        │ (binary GOOF frames,
           │ binary CRDT sync       │  ViewSpec-reduced)
   ┌───────▼───────────────────────▼──────────┐
   │  goofi-bridge  (axum)                     │
   │   · command dispatch + per-session history│
   │   · CRDT mirror of graph state (yrs)      │
   │   · ViewSpec reduction, off the tick path │
   │   · serves the built SPA from frontend/build/
   └───────┬───────────────────────────────────┘
   ┌───────▼───────────────────────────────────┐
   │  goofi-engine — Graph, scheduler, scopes   │
   │   adaptive tick under one graph mutex      │
   └───┬───────────────┬───────────────┬────────┘
       │ native Rust   │ in-process    │ detached worker
       │ (inventory)   │ Python (FT)   │ + iceoryx2 SHM
       ▼               ▼               ▼
    goofi-nodes    goofi-py/PyNode   goofi-subproc → `python -c "import goofi; goofi.serve()"`
```

### Control plane — `/control` WS
Two interleaved channels on one socket: **JSON** (RPC requests + `hello`/`error`
events) and **binary** (CRDT sync frames). A client's replica of the graph is
**read-only** — it never writes the doc. Every mutation is a **command** sent as an
RPC; the manager applies it, re-mirrors the graph into the doc, and broadcasts the
delta. Reads come from the doc, not from event echoes.

**Undo/redo is manager-owned.** Each browser tab mints a `sessionStorage` session id
sent on every request. `goofi-engine/src/command.rs` gives every command an exact
inverse; `CommandHistory` stores one *toggle* per entry (its inverse when applied,
the forward when undone) so redo is uid-stable, filtered per session. The client
records exactly one `graph_cmd` per successful mutating RPC and delegates
undo/redo to the manager — **so `CommandHistory::apply` must record every command,
including a forward no-op**, or the two stacks desync 1:1. Layout/view undo (panels,
tabs, viewport) stays client-local.

### Data plane — `/data/<node>/<slot>` WS
**One stream per (node, output slot)**, regardless of how many viewers watch it.
Each viewer publishes a **ViewSpec** — a payload-free constraint algebra (dtype,
dim-count comparisons, per-dim length comparisons, a desired reduction length per
dim) — inband as `{op:"view", specs:[…]}`. The bridge folds all specs against the
real frame (richest-per-dim: envelope > area > subsample), reduces **after the graph
lock drops** so it never blocks a tick, and ships one reduced frame to all
subscribers. Array `Data` is always **f32**; viewers render full-dtype reduced
frames (there are no per-kind adapters and no `__view__` sidecar).

### Node execution tiers
| tier | where | when |
|---|---|---|
| **native Rust** | inline on the tick thread | `goofi-nodes`, registered via `inventory` |
| **in-process Python** | inline, free-threaded 3.14t via pyo3 | a node whose imports are free-threading-safe |
| **subprocess Python** | a **detached off-tick worker**, iceoryx2 SHM | a node that needs the GIL or is missing on the FT interpreter |

Both Python tiers run the **same `goofi.Node` contract** and share one marshalling
seam (`goofi_pymod::exec::{run_setup, run_process}`), so they cannot drift — proven
by a cross-tier parity test. The detached tier is why a slow or hung subprocess node
can no longer stall the tick.

---

## Running, testing, building

```bash
cargo run                       # launches the backend + bridge, prints the URL
#   flags: --port N (default 8000), --bind HOST (default 127.0.0.1),
#          --python-nodes DIR | --subproc-nodes DIR | --auto-nodes DIR,
#          --subproc-python BIN, --list-nodes
# With no --*-nodes flag it auto-discovers ./nodes/ and routes each node by tier.
# --subproc-python defaults to the repo-local .venv when present.

cargo test --workspace                      # must stay green, and warning-free
cargo test -p goofi-py --features embed     # in-process Python host (needs .ftvenv)
cargo build -p goofi-cli 2>&1 | grep warning:   # ALWAYS check before declaring done

# Frontend (from frontend/):
npm run check    # svelte-check + tsc strict — keep 0 errors
npm run test     # vitest (unit)
npm run build    # static SPA → frontend/build/  (what the bridge serves)

# e2e (tests/e2e/, COMMITTED): Playwright + the real binary via window.goofi
cd tests/e2e && npm run e2e     # builds the backend, spawns it, runs, tears down
```

`goofi-cli/build.rs` rebuilds `frontend/build/` automatically when a frontend source
changed (skip with `GOOFI_SKIP_FRONTEND_BUILD=1`). Cargo **replays** a build script's
`cargo:warning` lines on later no-op builds — the rebuild line is written in the past
tense with its measured duration for exactly that reason.

**Python interpreters (machine-local, gitignored, `uv` venvs — use `uv pip`, never `pip`):**
- `.ftvenv` — free-threaded 3.14t: the in-process host **and** the introspection probe.
- `.venv` — a GIL python: the subprocess child.

Both need the `goofi` wheel; provision reproducibly with `scripts/provision-goofi-py.sh`.
Python-tier tests need env vars or they **silently skip**:
`GOOFI_SUBPROC_TEST_PYTHON=.venv/bin/python`, and for the `#[ignore]`d cross-lang probes
`GOOFI_PYMOD_TEST_PYTHON=.ftvenv/bin/python`.

If `/dev/shm/iox2_*` accumulates after a crash, delete the stale files before rerunning.

**Never** background CPU load with `(cmd &)` subshells when benchmarking — leaked
processes outlive the test and corrupt every later latency measurement.

---

## Backend map (`crates/`)

| crate | owns |
|---|---|
| `goofi-core` | `Data` (always-f32 arrays, string, table) + `Meta` (an `IndexMap` with typed accessors; builtin keys `sfreq`/`ufreq`/`index`/`channels`/`reduced`), `Param`, reduction kernels, globals, the introspection probe schema. |
| `goofi-codec` | the binary `Data` wire format (GOOF frame: header, msgpack meta, f32 body) + the subprocess request/response frames. Mirrored in `frontend/src/lib/codec/`. |
| `goofi-node` | the `Node` trait, `NodeManifest`, `SlotDecl`/`OutputDecl`/`ParamDecl`, the `ExprEvaluator` seam, and `discover.rs` (the Python introspection probe → a rich multi-slot + param manifest). |
| `goofi-nodes` | the native node library — deliberately **Oscillator + Buffer** (+ a test source) after the tabula-rasa reset. |
| `goofi-engine` | `Graph`: nodes, links, scheduling (adaptive tick, `next_run_delay`), param expressions (`nd()`), `.gfi` v6 save/load, `subpatch.rs` (flat scopes + stubs), `command.rs` (commands + inverses + `CommandHistory`), `detached.rs` (the off-tick worker tier). |
| `goofi-view` | the payload-free ViewSpec algebra: `plan(specs, frame)` folds many viewers' constraints into one reduction. |
| `goofi-crdt` | the yrs document: graph mirror, sync handshake, idempotent reconcile. |
| `goofi-bridge` | the axum server: `/control` dispatch + CRDT mirror + `/data` reduction/fan-out + `schemas.rs` (wire shapes) + the tick/stats workers. |
| `goofi-py` | the in-process Python tier: `PyNode` (a `Node` adapter over a live `goofi.Node`), the pyo3 param-expression evaluator, discovery. Feature-gated `embed`. |
| `goofi-pymod` | the `goofi` Python package itself, in Rust (pyo3): `Node`/`Data`/`Meta`/params, `introspect()`, the shared `exec` marshalling, and `serve()` — the iceoryx2 child loop. Dual-built: an abi3 wheel for GIL pythons, an rlib linked into the FT host. |
| `goofi-subproc` | `RemoteNode` — the manager side of the subprocess tier (spawn, seq-framed iceoryx2 round-trip, error frames). |
| `goofi-cli` | the `goofi-pipe` binary: arg parsing, tier routing/registration, `build.rs` (frontend build + pyo3 config). |

## Frontend map (`frontend/src/lib/`)

| dir | owns |
|---|---|
| `api/` | transport clients: `control.ts` (commands + events + session id), `data.ts`/`dataWorker.ts` (binary stream, off-thread decode), `frames.ts` (rAF paint coalescer), `perfStats`/`rateMeter`. |
| `crdt/` | the client replica: `SyncClient` (read-only) + the doc readers. |
| `codec/` | the TS port of the GOOF frame decoder (incl. float16). |
| `stores/` | reactive state (Svelte 5 runes): `graph.svelte.ts` (doc-authoritative mirror), `history.svelte.ts` (one linear client stack; graph steps delegate to the manager), `selection`, `ui`, `console`, `flash`, `logStream`. |
| `editor/` | the Svelte Flow canvas: `GoofiNode.svelte` (every node, incl. sub-patch instances), `snap.ts` + `nodeMetrics.ts`, placement, boundary nodes. |
| `viewers/` | one component per viewer kind + `ViewerFeed` (subscribe lifecycle), `capacity.ts` (emits the backend-shaped ViewSpec), `decimate.ts`, `imageGL.ts`. |
| `params/` | parameter widgets + the expression editor. |
| `panels/` | dockable panel content (node-editor, parameters, viewer, metadata, console, errors, globals) + the panel registry. |
| `workspace/` | the panel layout engine: `model.ts` (pure tree algebra), `workspace.svelte.ts`, `navContext.ts`. |
| `fs/` | the filesystem browser for save/load. |
| `agent/` | the automation façade (`window.goofi`) — the seam `tests/e2e/` drives. |

---

## Authoring a node

**Native (Rust).** Static declarations + `Default`-and-replay construction:
`static PARAMS: &[ParamDecl]`, slots as `&[SlotDecl]`/`&[OutputDecl]`, teardown via
`Drop`, positional axes labels. Register with `inventory` so the catalog finds it.
A param may declare a `default_expr` (e.g. `"globals.default_ufreq"`) — the engine
seeds a live binding instead of a literal.

**Python (`nodes/`).** Subclass `goofi.Node`: declare `config_input_slots()` /
`config_output_slots()` / `config_params()`, implement `setup()` and `process()`.
Plain top-level imports for all deps. The same file works on **both** Python tiers —
the discovery probe imports it in a real interpreter and reports whether it is
free-threading-safe; a node that isn't (or whose deps are missing on `.ftvenv`)
routes to the subprocess tier and shows as `Name (subproc)`. A missing dep greys the
palette entry; a raise inside `process()` is a per-tick error frame, not a crash.

---

## Key subsystems & their specs

Specs live in `docs/superpowers/specs/` (**gitignored** — on disk only). Read the
relevant one before changing the area.

- **Rust backend architecture** (`2026-07-16-rust-backend-architecture.md`) — the
  adopted design + the M1–M9 build plan.
- **ViewSpec data-plane reduction** (`2026-07-16-viewspec-data-plane-reduction-design.md`)
  — one stream per slot, constraint merge, reduce off the tick path.
- **Flat sub-patch scopes** (`2026-07-18-flat-subpatch-scopes-design.md`) — a scope
  tree of uids + `Stub` symlinks; `scope_of` is the single SSOT; sharing was
  deliberately **dropped**. `.gfi` v6.
- **Unified command API** (`2026-07-18-unified-command-api-design.md`) — everything
  is a manager command with an exact inverse; per-session history; the client replica
  is read-only.
- **CRDT control plane** (`2026-07-17-crdt-control-plane-design.md`) — the doc as the
  control-plane SSOT, mirror/reconcile, sync handshake.
- **Param expressions** (`2026-07-16-param-expressions-design.md`) — `nd('node')`
  in a param, lifted into the DAG so the reference runs first (same-tick, no latency).
- **Globals panel** (`2026-07-17-globals-panel-design.md`) — patch-scoped system +
  user globals as a doc root; `default_ufreq` is the producer rate reference.
- **Isolated node tier** (`2026-07-19-isolated-node-tier-design.md`) — off-tick
  detached execution + the typed-sfreq/opaque-meta SHM transport.
- **Python node unification** (`2026-07-20-python-node-unification-design.md`) — one
  `goofi.Node` contract across both tiers, `goofi.serve`, the shared exec seam.
- **f32-only + Meta-as-map** (`2026-07-20-f32-only-and-meta-map-design.md`).
- **ufreq convention** (`2026-07-16-ufreq-meta-design.md`) — the measured update
  frequency is the rate of record; `sfreq` is intra-frame only.

Analysis reports live in `docs/analysis/` (also gitignored).

---

## Hard constraints

- **Work happens on `rust-rewrite`. Leave `main` alone.** Never push or force-push
  without explicit authorization; branch before committing on a default branch.
- **`docs/` is gitignored on this branch** — specs and plans are on disk, not in git.
  Don't be surprised when `git status` ignores them, and don't "restore" them.
- Commit in small, focused, readable steps at green checkpoints — not one mega-commit.
- Commit messages end with: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- No auth on the WS endpoints — single-user, local/trusted-LAN app.
- Desktop browser only (no mobile/touch). One theme, done well (no dark-mode toggle).
- Don't reintroduce dearpygui, zmq, or a Python manager.
- The iceoryx2 transport and the GOOF wire format are load-bearing; changing either
  means changing `frontend/src/lib/codec/` in lockstep (`codec_golden.json` pins it).

---

## Current state

The **framework** is essentially complete and has been audited to convergence
several times: CRDT control plane, unified commands + manager-owned undo/redo, flat
sub-patches, the ViewSpec data plane, globals, both Python node tiers, e2e.

The **node library is a deliberate tabula rasa** — Oscillator + Buffer only. Growing
it (sinks, filters/PSD, real biosignal inputs, recording, array math) is the next
major project, and is meant to be **co-designed with the user**, not chosen
unilaterally. Longer-horizon: the audio and video pillars (each its own runtime,
sharing the core graph/`.gfi`/undo/sub-patch machinery), designed but not built.
