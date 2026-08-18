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

This file is the orientation for an agent session working in this repo. Read it
end-to-end before touching code, then read the specific subsystem you're changing.
**The "How we work" section is not optional — it is how changes are expected to be
made here.**

## Mandatory work ethic

All communication, whether internal thought, explaining work, planning, sub-agent communication or exchanges with the user, must be held in ASD-STE100 Simplified Technical English only.

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

6. **Zero warnings.** A task is not done at "Finished" — run
   `cargo build --workspace --all-targets 2>&1 | grep -n '^warning'` and clear what it prints.
   Remove the dead field or function; do not silence it with a `_` prefix or an `#[allow]`.
   **`--all-targets` is the load-bearing part**: a plain `cargo build` does not compile the
   integration-test targets, so a warning in `tests/*.rs` sails through.
   Clippy is a **separate, currently-unmet** bar: `cargo clippy --workspace --all-targets` has a
   pre-existing backlog (~47 across `goofi-core`, `goofi-engine`, `goofi-bridge`, `goofi-node`).
   The rule that binds today is: **add no new clippy warning in a file you touch.** Clearing the
   backlog is its own task; until then do not claim "clippy clean" workspace-wide.

7. **Honest reporting.** If tests fail, say so with the output. If a step was
   skipped, say that. State what is verified plainly; don't claim done what you
   haven't run.

8. **A double or fixture that cannot express the failure makes a mutation proof theatre.**
   This cost six real defects in one day, including a Critical. `FakeSocket.readyState` was
   hard-coded to `OPEN`, so a resize proposal dropped on a CONNECTING socket was invisible.
   `FakeTerm.open` didn't insert its element, hiding two terminals stacked in one host. A
   single-node fixture hid an fps counter summing across streams for two days. A uid test
   loading into a *fresh* instance passed against code that renumbered every node.
   The question is not "does this stand in for the real thing?" but **"can this reproduce the
   failure I am trying to prevent?"** — and the way to answer it is to **run the broken variant
   against your fixture and watch it pass**. Three implementers did exactly that and caught
   their own hollow tests before shipping them.

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
    caller (bridge RPC dispatch, the status-drain worker, a frontend event, a node's own
    wake loop) and **check the upstream guards**, not just the local function.
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
   │   · status-drain worker (the node reports)│
   │   · ViewSpec reduction, on its own sub    │
   │   · serves the built SPA from frontend/build/
   └───────┬───────────────────────────────────┘
   ┌───────▼───────────────────────────────────┐
   │  goofi-engine — Graph, wiring, scopes      │
   │   no tick: no node RUNS under the mutex    │
   └───┬───────────────┬───────────────┬────────┘
       │ native Rust   │ in-process    │ subprocess
       │ (inventory)   │ Python (FT)   │ Python (GIL)
       ▼               ▼               ▼
    goofi-nodes  goofi-python::inproc  goofi-python::subproc → `python -c "import goofi; goofi.serve()"`
       └───────────────┴───────────────┴──── one THREAD each, self-scheduled,
                                             talking iceoryx2 shared memory
```

### Control plane — `/control` WS
Two interleaved channels on one socket: **JSON** (RPC requests + `hello`/`error`
events) and **binary** (CRDT sync frames). A client's replica of the graph is
**read-only** — it never writes the doc. Every mutation is a **command** sent as an
RPC; the manager applies it, re-mirrors the graph into the doc, and broadcasts the
delta. Reads come from the doc, not from event echoes.

**The doc is the ONLY graph projection.** The `hello`/`graph_replaced` snapshot carries
no nodes, links or sub-patch forest — those live in the doc alone, and the client
assembles each node from doc + catalog. (It once carried both; they drifted, keying
scope members by display name on one side and by uid on the other.) What the snapshot
does carry is the session frame — instance id, palette, save path, layout — plus a
`runtime` overlay (`{uid: {stage, error}}`): the one per-node truth the doc never holds,
seeded here because its live stream (the status-drain worker) pushes only *transitions* —
each one stamped by the node itself — so a client joining a running graph would otherwise
draw an errored node as healthy.
`node_added` is a bare `{uid}` announcement. **Do not re-add graph state to a payload** —
if a client needs it, it is in the doc.

**Undo/redo is manager-owned.** Each browser tab mints a `sessionStorage` session id
sent on every request. `goofi-engine/src/command.rs` gives every command an exact
inverse; `CommandHistory` stores one *toggle* per entry (its inverse when applied,
the forward when undone) so redo is uid-stable, filtered per session. The client
records exactly one `graph_cmd` per successful mutating RPC and delegates
undo/redo to the manager — **so `CommandHistory::apply` must record every command,
including a forward no-op**, or the two stacks desync 1:1. **Layout undo is manager-owned
too** (A moved the arrangement onto the doc as an ordinary command); only the *viewpoint* —
the page in front, the focused panel, each editor's sub-patch depth — stays client-local,
because it is this client's alone and is never converged.

`apply` also **gates on `Command::precondition`, and `flip` deliberately does not.** The
idempotent guards inside `execute` exist so a stale toggle converges instead of wedging a
session's undo stack; a first-hand RPC earns no such benefit of the doubt, and eight ops
once answered `{ok:true}` for work they had not done. Tolerance belongs to replay,
strictness to the fresh caller — separated at one seam rather than duplicated per call site.

### Data plane — `/data/<node>/<slot>` WS
**One stream per (node, output slot)**, regardless of how many viewers watch it.
Each viewer publishes a **ViewSpec** — a payload-free constraint algebra (dtype,
dim-count comparisons, per-dim length comparisons, a desired reduction length per
dim) — inband as `{op:"view", specs:[…]}`. The bridge folds all specs against the
real frame (richest-per-dim: envelope > area > subsample), reduces on **its own
subscription** to the producer's output service — the same door a test's `OutputProbe`
opens, so no number of viewers can slow a `process()` down — and ships one reduced frame
to all subscribers. Array `Data` is always **f32**; viewers render full-dtype reduced
frames (there are no per-kind adapters and no `__view__` sidecar).

### Node execution — every node schedules itself
**There is no tick.** Each node owns one manager-side thread that parks on its own doorbell
and wakes for a `Control` message (a param edit, a wire change, a ⟳, a stop), a frame on an
input slot, a frame on a producer one of its **expression bindings** references, or its own
rate cap elapsing. Frames travel node to node over iceoryx2 shared memory, never through the
graph — so **no node runs under the graph mutex**, and no user action ever waits on a `process()`.
The steady state is not lock-FREE, though: the `/data` reducer re-derives its slot's service name
under a brief lock once a second (`REHOME_INTERVAL`), so a restart re-homes the stream instead of
leaving the viewer on the dead generation's name.

A node's state travels the other way, on its **status** service: `Ready`, `Stage`, `Fault`,
`BindingErrors`, `ParamValues`, `Ufreq`, `RefreshOptions`, and the `Ack` that advances a
wire's three-phase attach. The **status-drain worker** (`goofi-bridge`'s `spawn_stats`) is
the one thing that applies them — so it drains at 1 ms while broadcasting its five events at 2 Hz:
`node_stats`, `param_values`, `error`, `node_stage`, and the `state_update` that echoes a
`RefreshOptions`. The drain is the runtime's clock (a node is not addressable, and a cable does not
attach, until it runs); the broadcast is the UI's.

Two consequences worth holding: a node is **known when `add_node` answers and addressable
only when it reports `Ready`** (§4's birth barrier — pub/sub has no history, so a `Control`
sent before its subscriber exists is simply lost, which is why the graph queues them); and
a test observes a node the way `/data` does, through `goofi_engine::testing`'s `wait_for`
and `OutputProbe`, never by stepping it.

| tier | where | when |
|---|---|---|
| **native Rust** | its own thread in this process | `goofi-nodes`, registered via `inventory` |
| **in-process Python** | its own thread, free-threaded 3.14t via pyo3 | a node whose imports are free-threading-safe |
| **subprocess Python** | a child interpreter, iceoryx2 SHM | a node that needs the GIL or is missing on the FT interpreter |

Both Python tiers run the **same `goofi.Node` contract** and share one marshalling
seam (`goofi_pymod::exec::{run_setup, run_process}`), so they cannot drift — proven
by a cross-tier parity test.

**Exit is a real teardown.** Ctrl-C and SIGTERM reach `Graph::shutdown` through the CLI's one
exit path: every node is stopped and *waited for* (a ceiling, not a join — a wedged node must
not wedge the exit), which is what releases its shared memory. What a crash leaves behind is
reclaimed by the next start's sweep.

---

## Running, testing, building

```bash
cargo run                       # launches the backend + bridge, prints the URL
#   flags: --port N (default 8000), --bind HOST (default 127.0.0.1),
#          --extra-nodes DIR, --list-nodes
# It scans ./nodes/ when present and routes each node by tier; --extra-nodes ADDS a
# directory to that (repeatable, later wins a shared type name). NEITHER the tier nor
# the interpreter is selectable: one probe per node file routes, and the subprocess tier
# always runs .gfivenv, which `cargo run -p goofi-init` provisions.

cargo test --workspace                     # must stay green, and warning-free
cargo test -p goofi-tests --features embed # the suite PLUS the in-process Python tier, which
#                                            # LINKS libpython and needs .gfivenv-ft. Without the
#                                            # feature only the subprocess tier's suite compiles.
cargo test -p goofi-pymod --features host  # the goofi package's own decode tests (InputSlot → probe::Slot)
cargo build --workspace --all-targets 2>&1 | grep -n '^warning'   # ALWAYS check before declaring done
#   `--all-targets` is load-bearing: a plain `cargo build` never compiles the integration-test
#   targets, so a warning inside `tests/*.rs` (a non-snake-case test name, an unused import)
#   passes the gate and ships. Anchor the grep to `^warning` — `warning:` alone also matches a
#   runtime log line a test happens to print, which reads as a failing gate when it is not.

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

**Setup is ONE command, and it is not a shell script:**

```bash
cargo run -p goofi-init     # once per clone; needs `uv` on PATH
```

`goofi-init` (`backend/goofi-init/`) creates both venvs, installs both wheels, and writes the
gitignored `.cargo/config.toml` that points pyo3 at the free-threaded interpreter. After it,
`cargo build`, `cargo test` and `cargo run` all work **first time**. Until it, `goofi-cli`'s
build script fails with one line telling you to run it.

**Never resolve the interpreter path.** The config names `.gfivenv-ft/bin/python` *relative*
(`relative = true`, which cargo expands against the directory holding `.cargo/`) and canonicalizes
nothing. On unix a venv's `python` is a symlink into uv's base install — and the base install is
exactly where the `goofi` wheel is *not*, since it was installed into the venv. A `canonicalize` on
that path therefore hands pyo3 and the discovery probe an interpreter that cannot `import goofi`:
nothing errors, every Python node just silently drops to the subprocess tier (`--list-nodes` reports
`0 in-process`). Windows venvs hold a real `python.exe`, which is why this reads as harmless there.
`PYTHONHOME` and the rpath stay absolute because they name uv's install, outside the repo entirely.
Readiness is likewise checked through the *environment* (`goofi_init::interpreter`), never by
matching the config file's text — cargo has already expanded `[env]` by the time a build script runs.

Upgrading a clone from before this: `.venv` and `.ftvenv` are superseded by `.gfivenv`/`.gfivenv-ft`
and can be deleted (uv writes a `.gitignore` inside each, so git never mentions them).

**Why a crate and not a build script:** pyo3 reads `PYO3_PYTHON` from the environment, and cargo
reads `.cargo/config.toml` exactly once, at startup. A build script writing that file cannot reach
the build it is part of — so provisioning from `build.rs` made the first `cargo build` link against
whatever interpreter was on `PATH`, or fail outright on a machine with none, and needed a second
`cargo run` to come good. **Why Rust and not `.sh`:** one command in PowerShell, cmd, bash, zsh and
fish alike, with no `.sh`/`.ps1` pair to keep in sync. It depends on no goofi crate and no pyo3, so
`-p goofi-init` can never trigger the build it exists to configure. It is excluded from
`default-members` so a bare `cargo run` stays unambiguous.

**The two interpreters** (machine-local, gitignored; use `uv pip`, never `pip`):
- `.gfivenv-ft` — free-threaded 3.14t: the in-process host pyo3 LINKS against, and the probe.
- `.gfivenv` — a GIL python (pinned 3.12, since the subprocess tier exists precisely for packages
  that are *not* free-threading-safe): the subprocess child, with no flag to point elsewhere.

Wheels are built through `uv tool run maturin`, numpy riding along as the wheel's own declared
dependency. The names are deliberate: a generic `.venv` is claimed by editors and by `uv` itself,
and a stale one is not inert — this repo's own `.venv` once held an editable install of the OLD
Python goofi, which answered `import goofi` perfectly well and then had no `introspect`.

**Known gap:** provisioning reinstalls when `goofi` is *missing or broken*, never when it is merely
*stale*. After changing `goofi-pymod`, delete the venv (or `uv pip uninstall goofi` from it), or the
probe keeps running the old wheel and a node using a new authoring feature (a `doc=` kwarg, say)
silently disappears from the palette.

The cross-language tests find these interpreters themselves and **fail with an actionable
message** when none can `import goofi` — they never skip, and nothing in the suite is
`#[ignore]`d. `GOOFI_SUBPROC_TEST_PYTHON` / `GOOFI_PYMOD_TEST_PYTHON` / `GOOFI_FT_PYTHON`
override the interpreter choice.

**`/dev/shm/iox2_*` is not a leak, and two reviews have now misread it as one.** The count PEAKS
during a run — 1181 measured across a full workspace run — and settles back to 1, because every
node releases its shared memory when it drops. A crashed run drops nothing, and the next process's
startup sweep (`runtime::reclaim_stale_resources`) reclaims what it left. Delete the files by hand
only when you want a clean measurement, never as a fix.

`/tmp/iceoryx2/nodes` used to grow by about 1000 empty directories per `goofi-engine` run, which
made every later test binary slower — `goofi-python`'s subprocess tests took 529 s instead of 8.8 s.
That was a drop-ORDER defect, not a sweep failure: four structs declared their iceoryx2 node before
the ports built from it, and Rust drops a struct's fields in declaration order, so the node could
not remove its own directory. Fixed in `11bf182c`. **A struct that owns an iceoryx2 node beside its
ports must declare the node LAST.**

**Never** background CPU load with `(cmd &)` subshells when benchmarking — leaked
processes outlive the test and corrupt every later latency measurement.

---

## Backend map (`backend/`)

| crate | owns |
|---|---|
| `goofi-core` | `Data` (always-f32 arrays, string, table) + `Meta` (an `IndexMap` with typed accessors; builtin keys `sfreq`/`ufreq`/`index`/`channels`/`reduced`), `Param`, reduction kernels, globals, the introspection probe schema. |
| `goofi-codec` | the binary `Data` wire format (GOOF frame: header, msgpack meta, f32 body) + the subprocess request/response frames. Mirrored in `frontend/src/lib/codec/`. |
| `goofi-node` | the `Node` trait, `NodeManifest`, `SlotDecl`/`OutputDecl`/`ParamDecl`, the `ExprEvaluator` seam, and `discover.rs` (the Python introspection probe → a rich multi-slot + param manifest). |
| `goofi-nodes` | the native node library — deliberately **Oscillator + Buffer** (+ a test source) after the tabula-rasa reset. |
| `goofi-engine` | `Graph`: nodes, links, param expressions (`nd()`), `.gfi` v7 save/load — a zip of `patch.yaml` + `workspace/` (`archive.rs`), `subpatch.rs` (flat scopes + stubs), `command.rs` (commands + inverses + `CommandHistory`), and `runtime/` — the per-node threads and their iceoryx2 transport (`mod.rs` the wake loop, `wire.rs` the message shapes, `plan.rs` the three-phase wire planner, `mailbox.rs` a node's inputs, `transport.rs` the ports and the startup sweep). `testing.rs` is `wait_for`/`OutputProbe`, public because the bridge and Python suites need the same two shapes. |
| `goofi-view` | the payload-free ViewSpec algebra: `plan(specs, frame)` folds many viewers' constraints into one reduction. |
| `goofi-bridge` | the axum server: `/control` dispatch + `/data` reduction/fan-out + `schemas.rs` (wire shapes) + the status-drain worker (`spawn_stats`), and the yrs document itself — `crdt.rs` (shape-agnostic: graph mirror, sync handshake, idempotent reconcile) beside `crdt_mirror.rs`, its only caller and the one place the doc's roots are named. |
| `goofi-python` | the manager side of BOTH Python tiers, one crate because the probe that routes between them is the same probe: `inproc` (`PyNode` — a `Node` adapter over a live `goofi.Node` — plus the pyo3 param-expression evaluator; feature-gated `embed`, since it LINKS libpython) and `subproc` (`RemoteNode`: spawn, seq-framed iceoryx2 round-trip, error frames; unconditional, since it only spawns one). Both expose the same `probe`/`node_type_from` pair. |
| `goofi-pymod` | the `goofi` Python package itself, in Rust (pyo3): `Node`/`Data`/`Meta`/params, `introspect()`, the shared `exec` marshalling, and `serve()` — the iceoryx2 child loop. Dual-built: an abi3 wheel for GIL pythons, an rlib linked into the FT host. |
| `goofi-cli` | the `goofi-pipe` binary: arg parsing, tier routing/registration, `build.rs` (frontend build + pyo3 config). |

## Frontend map (`frontend/src/lib/`)

`src/app.css` is the **styling SSOT** — every colour, spacing, type, radius and motion token lives
in its `:root`, and a component states its own layout, never another component's. Build UI by
composing `$lib/ui` primitives; reach for a bespoke `<style>` only when no primitive expresses it,
and say why in a comment. `lib/theme/` is the enforcement.

| dir | owns |
|---|---|
| `api/` | transport clients: `control.ts` (commands + events + session id), `data.ts`/`dataWorker.ts` (binary stream, off-thread decode), `frames.ts` (rAF paint coalescer), `perfStats`/`rateMeter`. |
| `crdt/` | the client replica: `SyncClient` (read-only) + the doc readers. |
| `codec/` | the TS port of the GOOF frame decoder (arrays are always f32). |
| `stores/` | reactive state (Svelte 5 runes): `graph.svelte.ts` (doc-authoritative mirror), `history.svelte.ts` (one linear client stack; graph steps delegate to the manager), `selection`, `ui`, `console`, `flash`, `undoFlash`, `device` (the `--kb-inset` seam). |
| `editor/` | the Svelte Flow canvas: `GoofiNode.svelte` (every node, incl. sub-patch instances), `snap.ts` + `nodeMetrics.ts`, placement, boundary nodes. |
| `viewers/` | one component per viewer kind + `ViewerFeed` (subscribe lifecycle), `capacity.ts` (emits the backend-shaped ViewSpec), `decimate.ts`, `imageGL.ts`. |
| `ui/` | the primitive library — `Button`, `IconButton`, `Chip`, `Badge`, `Field`, `TextInput`, `NumberInput`, `Slider`, `Select`, `Toggle`, `Trigger`, `Tabs`, `Disclosure`, `Popover`, `Dialog`, `Bar`, `ScrollArea`, `StatusDot`, `EmptyState`, plus shared helpers such as `field.ts`'s label⇄control handshake, `liveValue`'s number↔slider latch, `clampToViewport` and `dragGesture`. One barrel (`index.ts`); per-instance CSS-var hooks are the documented escape hatch. |
| `inspector/` | the parameter inspector, and the north star's own target: `ParamForm` + `ParamField` (a 15-line declarative control dispatch inside one `<Field>`), `controlKind.ts`, `showWhen.ts` (the fail-closed dependency algebra — built and proven in the gallery, not yet exercised by any backend descriptor). |
| `theme/` | the styling ENFORCEMENT: `styleDrift.test.ts` (raw spacing/type/motion literals, the coarse idiom, `:global(.ui-*)` reach-ins, no gradients, no fault ink on the frame counters) and `tokens.test.ts` (contrast ratios), over the shared `contrast.ts`. Every exemption is one enumerated line with a reason — add the failing fixture BEFORE widening a scanner. |
| `app/` | the shell: `AppShell.svelte` (mount, layout push, keyboard), `Toast`, `TitleTip` (the coarse-pointer door onto every `title=`), `undoKeys.ts`. |
| `panels/` | dockable panel content (node-editor, parameters, viewer, metadata, console, globals — there is **no** `errors` panel; a legacy `errors` type migrates to `console` on load) + `register.ts`. |
| `workspace/` | the panel layout engine: `model.ts` (pure tree algebra), `workspace.svelte.ts`, `navContext.ts`, `registry.ts` (the panel-type seam). |
| `fs/` | the filesystem browser for save/load. |
| `agent/` | the automation façade (`window.goofi`) — the seam `tests/e2e/` drives. |

`src/routes/dev/ui` and `src/routes/dev/inspector` are **gallery routes**, not product: they are how
a primitive is pinned by an e2e before it has a product consumer. They ship in the built SPA (the
e2e drives them against the real binary) but nothing in the app links to them — a third of the e2e
suite targets them, which is worth knowing when reading the suite's numbers.

---

## Authoring a node

**Native (Rust).** Static declarations + `Default`-and-replay construction:
`static PARAMS: &[ParamDecl]`, slots as `&[SlotDecl]`/`&[OutputDecl]`, teardown via
`Drop`, positional axes labels. Register with `inventory` so the catalog finds it.
A param may declare a `default_expr` (e.g. `"globals.default_ufreq"`) — the engine
seeds a live binding instead of a literal.

**Python (`nodes/`).** Subclass `goofi.Node`: declare `config_input_slots()` /
`config_output_slots()` / `config_params()`, implement `setup()` and `process()`.
An input slot is a bare `goofi.DataType` or a `goofi.InputSlot(dtype, required=…, trigger=…)`, and
`process()` receives one kwarg per **declared** slot — `None` when that slot holds no frame.
A `required=True` slot never arrives empty (the node refuses the run and reports the error
before `process` is entered), so it may be read unconditionally; `trigger=` is authorable the same
way, defaulting to today's `True`.
A `StringParam(..., refresh=True)` gets a ⟳ button in the UI; the node answers it with a
`refresh_{group}_{name}(self) -> list[str]` method (the Rust analogue is
`Node::on_param_refreshed`). The hook runs on the node's OWN thread, so a multi-second device
scan costs that node its runs and nothing else — and the RPC cannot carry its answer: the reply
says only that the request went out, and the options reach the client on the status worker's
echo. Not yet wired for the subprocess tier.
Plain top-level imports for all deps. The same file works on **both** Python tiers —
the discovery probe imports it in a real interpreter and reports whether it is
free-threading-safe; a node that isn't (or whose deps are missing on `.gfivenv-ft`)
routes to the subprocess tier. (The palette is one flat list and no longer groups by category —
each row carries its *provenance*, builtin vs this patch, not its tier.)
A node whose deps are missing on BOTH interpreters fails its probe and is registered as
**unavailable**: it appears in the palette greyed and unclickable, labelled `unavailable` and
with its tooltip naming the missing module — a node that cannot load explains itself
instead of silently vanishing. A raise inside `process()` is a per-run error frame, not a
crash. A raise inside `setup()` leaves the node uninitialized — its error stands and nothing runs
against it, `process()` included — until a later wake or param interaction retries the whole
initialization on the same instance.

---

## Key subsystems & their specs

Specs live in `docs/superpowers/specs/` (**gitignored** — on disk only). Read the
relevant one before changing the area.

- **Rust backend architecture** (`2026-07-16-rust-backend-architecture.md`) — the
  adopted design + the M1–M9 build plan.
- **ViewSpec data-plane reduction** (`2026-07-16-viewspec-data-plane-reduction-design.md`)
  — one stream per slot, constraint merge, reduce off the graph lock.
- **Flat sub-patch scopes** (`2026-07-18-flat-subpatch-scopes-design.md`) — a scope
  tree of uids + `Stub` symlinks; `scope_of` is the single SSOT; sharing was
  deliberately **dropped**. `.gfi` v7.
- **Unified command API** (`2026-07-18-unified-command-api-design.md`) — everything
  is a manager command with an exact inverse; per-session history; the client replica
  is read-only.
- **CRDT control plane** (`2026-07-17-crdt-control-plane-design.md`) — the doc as the
  control-plane SSOT, mirror/reconcile, sync handshake.
- **Param expressions** (`2026-07-16-param-expressions-design.md`) — `nd('node')`
  in a param. **Superseded in part:** the lifting into a topo DAG went with the tick — a node
  evaluates its own bindings from the frames its references publish (see the async runtime spec).
- **Globals panel** (`2026-07-17-globals-panel-design.md`) — patch-scoped system +
  user globals as a doc root; `default_ufreq` is the producer rate reference.
- **Async node runtime** (`2026-08-14-async-node-runtime-design.md`) — the CURRENT execution
  model: no tick, one self-scheduling thread per node, iceoryx2 between them, the status-drain
  worker, the three-phase wire attach. Read this before anything else in the engine.
- **Isolated node tier** (`2026-07-19-isolated-node-tier-design.md`) — **superseded** by the
  above (every node is off-tick now, not just one tier); still the reference for the
  typed-sfreq/opaque-meta SHM transport.
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
- **The goofi version lives in ONE place: `[workspace.package] version` in the root `Cargo.toml`.**
  Every crate inherits it (`version.workspace = true`), `mcp.rs` reports it through
  `env!("CARGO_PKG_VERSION")`, and the Python wheel derives it because
  `goofi-pymod/pyproject.toml` declares `dynamic = ["version"]` instead of restating it. Bumping it
  also *re-provisions the venvs*: `goofi-init` compares the installed wheel's version against
  this one, so an older build in `.gfivenv`/`.gfivenv-ft` is rebuilt rather than silently kept.
  (An edit that does NOT change the version still needs the venv deleted.)
- **`docs/` is gitignored on this branch** — specs and plans are on disk, not in git.
  Don't be surprised when `git status` ignores them, and don't "restore" them.
- Commit in small, focused, readable steps at green checkpoints — not one mega-commit.
- Commit messages end with a `Co-Authored-By:` trailer naming the model that wrote them —
  e.g. `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`. Use the model
  actually in the session rather than copying this example, which was pinned to one version
  and went stale the moment the next one shipped.
- No auth on the WS endpoints — single-user, local/trusted-LAN app.
- **Responsive and touch-capable across desktop, tablet and phone** (both orientations).
  Desktop is the primary target and its behaviour is the reference — but "desktop only"
  is retired, so don't write a mouse-only affordance. Concretely: no interaction may exist
  *solely* behind `:hover`, right-click or a keyboard chord; touch needs its own door in,
  gated on `@media (hover: none) and (pointer: coarse)` — the ONE spelling (D-R7), because bare
  `(pointer: coarse)` catches a hybrid laptop and `any-pointer: coarse` catches a desktop with a
  drawing tablet plugged in. `theme/styleDrift.test.ts` turns any other spelling red, so this
  is an enforced rule, not a convention. Panel width is independent
  of viewport width, so `@container` is the default tool and `@media` is reserved for real
  device-class questions (pointer, hover, orientation). Size in `rem`/relative units — the
  `html` base is a responsive `clamp()`. One theme, done well (no dark-mode toggle).
- **The workspace/panel system and the cable-drag feel are frozen UX.** Restyle them, don't
  redesign them. **There is no phone-only layout mode** (reversed 2026-07-28 — an earlier
  "ephemeral display-only projection" plan was dropped by the user before it was built). A phone
  renders the *same* panel tree as the desktop; **panel maximize is the small-screen mechanism**,
  since it already shows one panel at a time and provably never persists (`toggleMaximize` touches
  `maximizedPanelId` only, which is not in `WorkspaceState`). The work is to make maximize/restore
  and the rest of the chrome touch-friendly — not to build a second navigation model.
- **Mobile and desktop are ONE system with different UI representations, never two tracks.** The
  infrastructure is shared — including dirty tracking and undo/redo. When a behaviour is wrong on
  phone, fix it for both; a phone-only guard is the wrong shape. Corollary the user set explicitly:
  **navigation must not dirty the patch on either platform** — entering a sub-patch is navigation
  (must not dirty), changing a viewer *type* is a real view setting (dirties). That taxonomy
  shipped: every layout write declares a `LayoutIntent` (`workspace.svelte.ts`), AppShell folds the
  debounce window and sends it with `set_layout`, and the bridge gates the dirty flag on it
  (`layout_write_dirties`). Persistence and dirtiness are separate axes — a navigation write still
  rides the `.gfi`. **Unclassified ⇒ authoring**, so forgetting to classify can only cost a
  spurious dot, never a lost change; entering a sub-patch and switching a layout tab (D-R11) are
  the navigation cases. Pinned by `layoutIntent.test.ts` and `tests/e2e/tests/dirty-taxonomy.spec.ts`.
- **`$lib/ui` must not import `$lib/stores`.** The primitives are a leaf layer. Importing a store into
  one reshuffles Vite's CSS chunk graph, which changed the emitted `<link>` order and gave the app a
  **first-paint FOUC** (caught by `inspector-gallery.spec.ts`, not by any unit test). When a primitive
  needs device state, read the published DOM/CSS property instead — `ui/clampToViewport.ts`'s
  `overlayViewport()` is the pattern.
- Don't reintroduce dearpygui, zmq, or a Python manager.
- The iceoryx2 transport and the GOOF wire format are load-bearing; changing either
  means changing `frontend/src/lib/codec/` in lockstep (`codec_golden.json` pins it).

---

## Current state

The **framework** is essentially complete and has been audited to convergence
several times: CRDT control plane, unified commands + manager-owned undo/redo, flat
sub-patches, the ViewSpec data plane, globals, both Python node tiers, e2e. The
2026-07-23 pass was a leanness audit executed end to end: it cut every plane the frontend
carried that the backend never grew, built the four the user wanted real (palette
availability, layout persistence, dirty tracking, detached bootstrap stage), and collapsed
the snapshot's duplicate graph projection onto the doc.

The **frontend design-system overhaul (F→P→N→M→R) then shipped**, each sub-project
spec→plan→execute→audit-to-convergence:

- **F — substrate.** `app.css` became the token SSOT; `viewport-fit=cover`, the safe-area insets,
  `--kb-inset`, the coarse type/hit floors.
- **P — primitives.** `lib/ui/`, which the panels, editor chrome and inspector now compose;
  **zero `:global(.ui-*)` consumer reach-ins**, grep-verified and pinned by a test.
- **N — inspector.** The north star: `lib/params/` → `lib/inspector/`, 1 527 → 857 lines (−44 %)
  while *gaining* the fx editor, `showWhen` dependency filtering and `@container` reflow.
  `ParametersPanel.svelte` is 19 lines.
- **M — migration + saliency.** Panel/node chrome onto the primitives; borders traded for the
  surface ladder; `gradient(` count is **zero**, pinned.
- **R — responsive shell.** Phone/tablet made real: progressive TopBar overflow, touch doors for
  every hover/right-click affordance, the layout `LayoutIntent` taxonomy, four e2e device profiles.

The trade, stated plainly: centralizing **added** production lines — a primitive library and two
galleries — and a great deal more test code. What it bought is a declarative inspector, no
component-to-component style reach-ins, and an app usable on a phone. The 2026-07-29 capstone audit
found **zero critical and zero correctness defects in the graph, data or CRDT planes**; its
important-tier items are fixed.

### H — an agent runs inside the patch (2026-08-10)

`spawn_harness` starts a harness (`claude`, `codex`, `opencode`) on a PTY with **cwd = the workspace
mount**, streamed to an `agent` panel over `/term/<instance_id>` (binary frames are PTY bytes, text
frames are JSON control). `list_harnesses`/`spawn_harness`/`stop_harness` are **control-only** — an
agent must not spawn or kill agents.

**Identity is structural: one MCP server, many addresses.** `/mcp` is the central endpoint any
external agent uses; `spawn_harness` mints **`/mcp/<instance_id>`** and writes *that* URL into the
config it hands its harness, so the address is minted by goofi and never travels through the agent —
nothing to spoof, nothing to validate. `stop_harness` drops the route. A **path, not a port**: a port
per harness buys the same property plus a listener, an accept loop and `TIME_WAIT` on relaunch.

**Orientation is `AGENTS.md` alone.** Seeded into a **new** workspace only (never on load), beside a
`CLAUDE.md` holding `@AGENTS.md` — one text for all three harnesses. The MCP `initialize.instructions`
channel was **removed**: measurement showed codex *does* read it but surfaces it as one namespace
blurb among ~180 tools, and never acted on it. Source is `goofi-bridge/src/orientation.md`,
`include_str!`d. **Known gap:** a `.gfi` saved before this gets no orientation at all.

**No server-side terminal emulator.** The spec's `wezterm-term` grid was cut: the xterm.js `Terminal`
lives in a client store keyed by `instance_id`, so scrollback survives closing and reopening a panel,
and a resize nudge makes a full-screen TUI repaint on a fresh attach. **History is allowed to be lost
on a page reload** — for an alternate-screen TUI a replay reconstructs a stale screen the app is about
to overwrite anyway. **Closing a view is not killing an agent**: the badge raises detach-or-kill.

**The Origin/Host allowlist covers every route *and the WebSocket upgrades*.** `/control`'s WS is
CORS-exempt and allowed full cross-origin read+write — strictly more exposure than `/mcp`. `/term`
made it urgent. It is a drive-by guard, not auth; this app stays single-user and local by design.

**`.goofiignore`** (not `.ignore` — ripgrep and fd read that as a *search* ignore, and the workspace is
an agent's cwd) says what not to package. Its rules are read **inside** the one `files()` walk that
serves both the pack and the fingerprint, so the two cannot be handed different lists.

**A load restores the uids the patch was saved with.** `load_doc` used to mint fresh ones, so loading
into an instance that had held other nodes renumbered everything and broke every panel binding — while
links survived, because they are remapped inside the load. The uid was never missing: a node record's
map *key* has always been its uid hex. `clear()` also resets the node clock, so a patch loaded an hour
in behaves as it does at boot.

### B — boundary hardening (2026-08-10)

Driven by two external reviews (`gpt-5.6-terra`, one Rust and one frontend). Roughly half of
what they raised was declined with reasons — the plan's exclusion table records which and why,
and one `high` finding was built entirely on a **stale sentence in this file**, now fixed. What
survived verification shared one shape: **a tolerant path reached by a strict caller.**

**Tolerance belongs to replay; strictness belongs to the fresh caller.** `Command::execute`'s
idempotent guards exist because an `Err` inside `CommandHistory::flip` permanently wedges a
session's undo stack. But `apply` — the first-hand RPC path — called the same `execute`, so
eight ops answered `{ok:true}` for work they had not done. The gate is `Command::precondition`,
checked in `apply` and never in `flip`. `wire_boundary`'s inner check reuses `set_stub_inner`'s
own algebra (extracted as `stub_wire_dtype`) rather than growing a second copy.

**A `Compound` now rolls back.** It is the restoration unit undo replays, and the bridge gates
its CRDT re-mirror on `is_ok()` — so a half-applied failure was a graph mutation no client was
told about. Rollback, not preflight: a compound's later children are validated against a graph
its earlier children build, so there is no pre-state to check them against.

**A restarted node carries ONE manifest.** `restart_node` never assigned `entry.manifest`,
defended by a comment that was true when it only served crash recovery — and made false by W1's
live patch-node editing without a line of that function changing. Under rescan the `type_name`
is stable while the *interface* is not. This is the cautionary shape of the whole pass: **a new
feature can invalidate an old assumption at a distance, and nothing in the diff shows it.**

**A lifecycle panic is a node error.** `process` was already contained; `setup`,
`on_param_changed` and `on_param_refreshed` were not — and unlike `process` they run under the
graph mutex the bridge holds, which is locked with `.lock().unwrap()` throughout. One node's
panic poisoned it and took the control plane down permanently.

**A fresh session is a generation boundary, not a document swap.** `SyncClient.reset()` installed
an empty replica while `nodes`/`links`/`instances`/`globals` and the per-uid view stores stayed
mounted — and an empty manager doc answers with a transaction that changes no Yjs type, so the
observer never fired. Restarting the backend under an open tab left the old graph on screen.
**The fixture had to deliver an empty transaction to catch it**; one that seeded the replacement
doc passes against the bug.

Also: `resolve_stub` carries a visited set (a self-referential stub in a hand-edited `.gfi`
overflowed the stack, which *aborts* rather than panicking); `Uid::from_hex` admits only the
canonical 12-hex domain, which makes `next_uid`'s `+ 1` total at every site; `rename_node`
refuses quotes and backslashes, because a display name is spliced into `nd()` expression source;
a refused cable no longer stays drawn; and the drop counter moved from the header to the stream
that drops.

### A — one op vocabulary, and layout stops being the frontend's (2026-08-10)

**The op registry** (`goofi-bridge/src/ops.rs`) is one row per op — name, doc, params/result schema,
`surface` (`mcp` | `control-only`). A coverage test fails if a dispatch arm has no row **or** a row
has no arm, and the table **generates** both the frontend's TS op constants (`api/ops.ts`, checked in
with a drift test) and the MCP tool list. That is what closed the old hazard: `dispatch` was a
string-keyed match with a silent missing arm, and `op` was a free string at scattered call sites.
**`mcp__goofi__<name>` ≤ 64 chars is a loud test**, because busting it makes Claude and OpenAI reject
the *entire* tool list with a 400. `control-only` keeps `load`, `save`, `new`, `serialize`,
`list_dir`, `load_text`, `set_layout` away from agents — `new` would wipe the patch and the undo stack.

**Layout is the fifth CRDT doc root**, flat and id-keyed like `nodes`: every page, split and panel is
one entry with `parentId`, `orderIndex` and a `size` fraction. Flat because the reconciler mirrors
nested *maps* but **erases nested arrays**; flat also makes move/reorder/reparent field edits with
panel identity preserved. Persistence, undo and broadcast reuse the graph machinery — layout ops are
ordinary `CommandHistory` commands. **The opaque `layout` blob is gone from the snapshot**, so layout
has exactly one projection. The frontend is now purely a replica: that migration **deleted 225 more
production lines than it added.**

**The rule four fix rounds bought, and the guard that keeps it:** *no layout inverse restores raw
state — every inverse re-plans through the forward planners.* A raw slot restore can pin an entry
back into a position a concurrent peer has taken, stranding their panels and corrupting the
arrangement on the next save. Every instance was found by **driving two sessions over the real WS**,
never by reading. A guard test walks every layout write op **from the registry** and asserts the
safe/unsafe sets **both ways**, so striking an op off without fixing it fails too. One recorded
exception: `session_reorder_page`, where the order *is* the content.

**`/mcp`** is a route on the existing axum server — **one server per goofi instance**, HTTP, because
a stdio server is spawned per client and Claude never spawns one. goofi prints the URL at startup.
`dispatch` stays **synchronous**: no tool awaits under the graph lock, and an async seam would force
re-opening W's on-lock save (whose off-lock version once cost ~450 lines of race machinery).

### W — a patch is an archive, and the manager owns its name (2026-08-09)

Stated here in full rather than by reference, because the plans and audit reports live in
**gitignored** `docs/` and would not survive a fresh clone.

**A `.gfi` is a zip**, manifest **v7**: `patch.yaml` beside a `workspace/` tree. At boot a run mints
`<temp>/goofi-<128-bit hex>/workspace` — one `PathBuf` on `AppState`, deleted on a graceful exit and
simply left behind after a crash, because a reboot clears `/tmp`. A save packs manifest + mount to a
temp sibling and renames onto the target; a load extracts into a **fresh** mount, parses, and only
then swaps — graph and workspace, or neither. Extraction uses `zip`'s own `extract()`; hand-rolling
zip-slip checks duplicates `safe_prepare_path`. Symlink creation and the absence of a size cap are
**accepted, by decision**. `goofi-engine/src/archive.rs` is the whole seam, at 74 production lines.

**C38 is closed.** The manager holds `save_path`, `schemas::snapshot` carries it, and
`save_path_changed` is broadcast from **save as well as load** — so a save converges other tabs and a
reload remembers. `new` and `open_workspace` joined the op set; `new` is literally *a load of an
empty patch* (it shares the `load` arm), which is why it cannot drift from load on the mount, the
history clear, the dropped path or the layout reset.

**Workspace dirtiness has no watcher** (decision): `archive::fingerprint` walks the mount and is
compared inside `is_dirty()`, which runs at `hello` and lag-recovery only, **off the graph lock**. So
an external edit surfaces on the asker's next snapshot and no thread hunts for one. The save stays
**on** the graph lock; taking it off cost a previous attempt ~450 lines guarding a race that only
exists once it is off-lock.

**Deferred, wanted, not now:** autosave + a flocked registry of active patches + offer-reopen-on-
startup. That is the answer to crash recovery here — not `read_stable`, RAII temp guards or fsync
sequences, which buy nothing a user can see when the mount is disposable.

**The cautionary tale, worth keeping:** the first W delivery shipped +6042/−621 for this and was
**reset by the user on leanness grounds**. The rebuild is +1932/−542 for the same feature *plus* C38
and two live bug fixes. The four inflation sources were: moving the save off-lock, a filesystem
watcher with no consumer, hand-rolled hardening a dependency already did, and a mount lifecycle with
registries and RAII where one `PathBuf` was wanted. Per-task audits could not catch it — every task
was correct against a brief. **Nothing in a task loop reviews the plan against the feature; only the
human does.**

The **node library is a deliberate tabula rasa** — Oscillator + Buffer only. Growing
it (sinks, filters/PSD, real biosignal inputs, recording, array math) is the next
major project, and is meant to be **co-designed with the user**, not chosen
unilaterally. Longer-horizon: the audio and video pillars (each its own runtime,
sharing the core graph/`.gfi`/undo/sub-patch machinery), designed but not built.
