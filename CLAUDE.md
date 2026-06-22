# goofi-pipe — Frontend Rewrite

You are starting fresh in `~/projects/goofi-next/goofi-pipe`. The backend
of goofi-pipe was rewritten in the previous session (iceoryx2 transport,
zero-copy publish path, ~25 commits on the `dev` branch). The backend is
**done and not your concern** beyond reading its public surface. Your job
is to **replace the entire dearpygui frontend with a browser-based UI**.

This brief is self-contained — read it end-to-end before touching code.

---

## 1. What goofi-pipe is

A real-time node-based data-processing platform aimed at biosignals
(EEG, ECG, audio, video). Users build patches in a node-graph GUI:
nodes are processes that ingest, transform, or emit `Data` objects;
edges (links) carry data between nodes' output and input slots.

Each node runs in its own OS process; the **manager** orchestrates them,
maintains the graph, persists patches as `.gfi` YAML files, and bridges
the in-flight data to whatever frontend is attached.

The current frontend is a single-process dearpygui app embedded in the
manager. **You are replacing it.**

---

## 2. Background: backend refactor (already shipped)

Read `git log --oneline main..dev` for the full record. Highlights:

- **`src/goofi/transport.py`** — unified `Publisher` / `Subscriber` /
  `Listener` / `Notifier` / `WaitSet` API backed by iceoryx2 (cross-
  process shared memory) and a thread variant (intra-process group).
  Factory functions: `open_publisher(name, *, in_process, latest_wins)`
  and `open_subscriber(...)`.
- **`src/goofi/codec.py`** — binary `Data` codec with a fixed 12-byte
  header, msgpack meta, and a dtype-specific body (raw numpy bytes for
  ARRAY). Functions: `prepare_encode(d) -> (size, meta_bytes)`,
  `encode_data_into(d, view, meta_bytes=...)`, `encode_data(d) -> bytes`,
  `decode_data(buf) -> Data`. Read the docstring at the top — the wire
  format is exact and you will need to mirror it in JavaScript.
- **Zero-copy publish path**: producers write numpy arrays straight into
  iceoryx2 SHM slices via `Publisher.loan(size)` → `_IpcLoan.buffer`
  memoryview → `.send()`. One memcpy heap → SHM.
- **`Manager` runs in its own process**; nodes spawn as their own
  processes (or share a process group via the `common.process_group`
  parameter). The manager owns `NodeContainer` (name → `NodeRef`) and
  the `_links` list.
- **`NodeRef`** is the manager-side proxy. It owns a ctrl pub/notifier
  (to the node) and a status sub/listener (from the node). When the
  node dirties its state, the node pushes a `STATE_UPDATE` and the
  `NodeRef.serialized_state` field updates. `wait_for_state()` blocks
  on a `threading.Event` until the first state arrives.
- **`NodeRef.set_data_handler(slot, callback)`** registers a per-output-
  slot data callback. A single pump thread per NodeRef drains a shared
  `WaitSet` over all subscribed slot listeners and invokes the callback
  with decoded `Data`. **This is the hook the new bridge uses to ship
  data to the browser.**
- Ctrl+C now works (iceoryx2 `SignalHandlingMode.Disabled`).

**Do not** touch `transport.py`, `codec.py`, or the iceoryx2 setup unless
absolutely necessary. Minor manager refactors are fine.

---

## 3. What you are deleting

```
src/goofi/gui/
    window.py         (1273 LOC)  — dearpygui main window, node editor, dockspace
    data_viewer.py    ( 670 LOC)  — viewer types: array, image, trajectory, topomap, string, table
    events.py         ( 369 LOC)  — keyboard / paste handlers
    __init__.py
src/goofi/assets/     — dearpygui font / theme assets
```

All of these go. They are not features to migrate; they are an
implementation to replace. Every feature they ship must be re-built in
the new frontend (see §6 for the full feature inventory).

Anything in `src/goofi/manager.py` that calls into `goofi.gui` will need
to be re-pointed at the new bridge. Specifically:

- `Manager.__init__` instantiates `Window(self)` in non-headless mode
  (the dearpygui main loop blocks the main thread).
- `Manager.add_node` / `remove_node` / `add_link` / `remove_link` /
  `save` notify the GUI via `Window().add_node(...)` etc.

Replace these calls with broadcasts to the new bridge (see §5).

---

## 4. The goal

> A **browser-based frontend** for goofi-pipe that:
>
> 1. covers every feature of the current dearpygui frontend (see §6),
> 2. supports **zoom and pan in the node editor** (this was missing /
>    broken in dearpygui — the headline UX win),
> 3. renders **large data efficiently** in many simultaneous viewers
>    (HD video, kHz-rate EEG, etc.) without stuttering,
> 4. is clean, modern, and not visually cluttered or overlapped, and
> 5. is **the only frontend** — dearpygui is removed entirely.

The user explicitly said: *"this is a greenfield redesign + reimplementation
of the frontend into the current goofi-pipe framework. goofi3 was an overly
ambitious prototype... in the current approach we are now only working on
the frontend while keeping the backend intact."*

Treat goofi3's frontend (`~/projects/goofi-next/goofi3/frontend/`) as a
**reference for principles and stack, not as code to copy**. Read it
for ideas. Build your own.

---

## 5. Architecture: bridge + browser

The backend stays Python and process-per-node. To talk to a browser
you'll introduce a **bridge layer** inside the manager process that
exposes the manager's API and live data via HTTP + WebSocket.

```
   ┌─────────────── browser ───────────────┐
   │  SvelteKit SPA                        │
   │  - node editor (Svelte Flow)          │
   │  - parameter panels                   │
   │  - data viewers (uPlot, canvas, ...)  │
   │  - WebSocket clients                  │
   └───────────────┬────────────────────────┘
                   │ HTTP (static assets) + WS
                   │
   ┌───────────────▼────────────────────────┐
   │  goofi.bridge  (new — Python)          │
   │  - aiohttp / FastAPI server            │
   │  - serves /static/* (built SPA)        │
   │  - WS /control  (JSON RPC + events)    │
   │  - WS /data/<node>/<slot>  (binary)    │
   │  - subscribes to NodeRef state +       │
   │    data handlers to fan out            │
   └───────────────┬────────────────────────┘
                   │ Python method calls
                   ▼
            Manager + NodeRefs (unchanged)
                   │ iceoryx2 / thread transport
                   ▼
             Node processes (unchanged)
```

### Backend bridge (new code, `src/goofi/bridge/`)

Suggested module shape:

```
src/goofi/bridge/
    __init__.py
    server.py        # HTTP + WS server (aiohttp recommended; pure-Python, async)
    control.py       # /control endpoint — RPC dispatch + state broadcast
    data.py          # /data endpoint — binary streaming per slot
    schemas.py       # request/response shapes (pydantic or dataclasses)
```

- Lives in the manager process. The manager bootstraps it when
  `--headless` is *off* (the previous dearpygui code path).
- `--headless` keeps doing what it does now: run the manager without
  any UI.
- Add `--port N` (default 8000) and `--bind HOST` (default 127.0.0.1).
  Print the URL on startup; tell the user "Open <url> in your browser".
- The HTTP server runs in a daemon thread (`aiohttp.web.run_app` in an
  asyncio loop on its own thread). The manager's main thread stays free
  for `post_init`'s sleep loop and KeyboardInterrupt handling.

#### Control plane — `WS /control`

JSON messages. Bidirectional. **Client → server** commands:

| op | payload | semantics |
|---|---|---|
| `list_nodes` | `{}` | reply with all node types/categories + their schema (input slots, output slots, params) |
| `list_graph` | `{}` | reply with current nodes (name, type, category, position, params, error) + links |
| `add_node` | `{type, category, name?, params?, pos?}` | calls `Manager.add_node(...)` |
| `remove_node` | `{name}` | calls `Manager.remove_node(name)` |
| `add_link` | `{node_out, node_in, slot_out, slot_in}` | calls `Manager.add_link(...)` |
| `remove_link` | `{...}` | inverse |
| `update_param` | `{node, group, name, value}` | calls `NodeRef.update_param(...)` |
| `set_node_pos` | `{name, pos: [x,y]}` | updates `NodeRef.gui_kwargs["pos"]` |
| `save` | `{path?, overwrite?}` | calls `Manager.save(...)` |
| `load` | `{path}` | calls `Manager.load(...)` |
| `subscribe_data` | `{node, slot}` | opens a `/data/<node>/<slot>` stream for this client |
| `unsubscribe_data` | `{node, slot}` | closes it |

**Server → client** events:

| event | payload | when |
|---|---|---|
| `state_update` | `{node, state}` | on every `STATE_UPDATE` from the node |
| `node_added` | `{name, type, category, params, pos}` | after `Manager.add_node` succeeds (any source — including programmatic, e.g. patch load) |
| `node_removed` | `{name}` | after `remove_node` |
| `link_added` | `{node_out, node_in, slot_out, slot_in}` | after `add_link` |
| `link_removed` | `{...}` | inverse |
| `error` | `{node, error}` | on `PROCESSING_ERROR` from a node |
| `manager_shutdown` | `{}` | on terminate |

Use `Manager.set_message_handler` plumbing for state + error; for the
add/remove events, add a small notification hook in the Manager (just
fire a callback the bridge registered).

#### Data plane — `WS /data/<node>/<slot>`

- One WS per (node, slot) pair the client is currently viewing.
- Server pushes **binary frames** in the GOOF wire format defined by
  `codec.py`. The browser decodes with a TypeScript port of the same
  format (small — ~80 lines).
- Backend implementation: `NodeRef.set_data_handler(slot, callback)`
  where the callback forwards the encoded bytes to the WS. Throttle if
  the WS send buffer backs up (drop oldest — match the iceoryx2
  latest-wins semantics).
- Frontend: subscribe only while a viewer is mounted and visible
  (IntersectionObserver helps).

### Frontend bundle (new code, `frontend/`)

```
frontend/
    package.json            # SvelteKit project
    svelte.config.js
    vite.config.ts
    src/
        routes/
            +layout.svelte
            +page.svelte         # main editor view
        lib/
            api/                 # WS clients (control + data)
            codec/               # GOOF decoder (TS port of codec.py)
            editor/              # node graph (Svelte Flow integration)
            viewers/             # ArrayViewer, ImageViewer, etc.
            params/              # param widgets
            stores/              # graph state, selection, etc.
        app.html
        app.css
```

- **Build target:** static SPA. `npm run build` produces `frontend/build/`
  which the bridge serves. In dev, run `npm run dev` (Vite dev server)
  and have it proxy `/control` and `/data` to the running bridge.
- **TypeScript strict, no `any` in app code.** Codec layer can use
  `unknown`-narrowing helpers.

---

## 6. Feature inventory — what the new frontend must do

Sourced from `src/goofi/gui/window.py`, `data_viewer.py`, `events.py`.
The new frontend is feature-complete when each of these works:

### 6.1 Node editor (the canvas)

- [ ] Render every node in the graph, positioned by `gui_kwargs["pos"]`.
- [ ] **Zoom (mouse wheel, pinch) — must work smoothly**. Pan (right-
      click drag / space+drag). Reset view (keyboard shortcut).
- [ ] Drag a node to reposition; new position broadcast as `set_node_pos`.
- [ ] Select node(s): click, shift-click, marquee.
- [ ] Multi-select drag.
- [ ] Connect output slot → input slot by dragging from one pin to
      another. Visual feedback while dragging. Prevent invalid links
      (slot dtype must match the consumer side or be coercible — the
      backend doesn't enforce, so add a soft client-side check).
- [ ] Delete selected nodes / links via Delete key.
- [ ] Per-node category color (currently a palette in
      `window.py:NODE_CAT_COLORS` — pick a fresh palette but keep the
      "one color per category" idea).
- [ ] Error state visual: if `NodeRef.last_error` is set, the node
      shows a red border / icon.
- [ ] Minimap (optional but nice — `react-flow`/`svelte-flow` ships one).

### 6.2 Add-node menu

- [ ] Searchable (typeahead) list of all node types, grouped by
      category (analysis, array, inputs, misc, outputs, signal).
- [ ] Show docstrings on hover.
- [ ] Insert at cursor position.
- [ ] Backend source: `list_nodes` in `node_helpers.py` returns the
      registered node classes. The bridge should serialize each as
      `{type, category, doc, input_slots, output_slots, params}`.

### 6.3 Parameter panel

- [ ] When a node is selected, show its parameter groups in a side
      panel.
- [ ] Param types to render (see `src/goofi/params.py`):
      - `FloatParam(value, min, max)` → number input + slider
      - `IntParam(value, min, max)` → integer input + slider
      - `BoolParam(value, trigger=False)` → toggle (trigger params are
        click-once buttons)
      - `StringParam(value, options=[...]?)` → text input OR dropdown
        if `options` is set
- [ ] Show docstring on hover.
- [ ] Edits fire `update_param`.
- [ ] State updates from the backend (`state_update` events) reflect
      in the panel.

### 6.4 Data viewers (per output slot)

The dearpygui implementation has 6 viewer types cycled via Ctrl+click.
The new frontend must cover:

- [ ] **ArrayViewer** — line plot. 1D and 2D (channels × samples) arrays.
      Log-scale toggle per axis. Auto-scaling with shrinking.
      **Use uPlot or similar — must handle ~1 kHz × N channels smoothly.**
- [ ] **ImageViewer** — RGB/RGBA images. HD frames (1920×1080×3).
      Use a `<canvas>` with `putImageData` or WebGL for very large frames.
- [ ] **TrajectoryViewer** — 2D paths (xy plots over time).
- [ ] **TopomapViewer** — EEG topographic maps (channels arranged on a
      head). The dearpygui version pulls electrode layouts; keep that
      same data path, render via SVG or canvas.
- [ ] **StringViewer** — for `Data` of dtype STRING. A scrolling text
      area.
- [ ] **TableViewer** — for `Data` of dtype TABLE (dict of nested Data).
      A nested expandable tree.
- [ ] Each viewer type is cyclable by clicking a "switch viewer" button
      on the slot's viewer container.
- [ ] **High-dim fallback**: if no viewer can render the array
      (`ndim > 3` or weird shape), show a text summary (shape, dtype,
      stats). The dearpygui version does this in `data_viewer.py:130`.
- [ ] Viewers must **only consume data while visible**. Use
      IntersectionObserver to subscribe / unsubscribe.

### 6.5 Patch persistence

- [ ] **Save** — file picker (browser File System Access API on
      Chrome / `<a download>` fallback elsewhere) writing the .gfi
      YAML produced by `Manager.save`. Backend already does this; the
      frontend just triggers it and downloads the result.
- [ ] **Load** — similarly. Triggers `Manager.load`.
- [ ] **Unsaved-changes indicator** — title bar shows a dot when
      `Manager.unsaved_changes` is True.
- [ ] **Examples menu** — list of patches in `examples/` (the manager
      already has `get_example_patch`); expose it via `list_examples`.

### 6.6 Copy / paste

- [ ] Ctrl+C: serialize selected nodes + their internal links to JSON
      (the current `copy_nodes` in `events.py` produces a payload —
      mirror that schema for compatibility, or design your own and
      version it).
- [ ] Ctrl+V: deserialize and create nodes at cursor.
- [ ] Use the system clipboard (`navigator.clipboard`).

### 6.7 Keyboard shortcuts

Match the dearpygui set where reasonable:

- `Ctrl+S` save · `Ctrl+O` load · `Ctrl+Z`/`Ctrl+Y` (out of scope —
  no undo in the old GUI, don't add it here either)
- `Delete` / `Backspace` remove selection
- `Ctrl+C` / `Ctrl+V` copy/paste
- `Ctrl+A` select all
- `Ctrl++` / `Ctrl+-` zoom
- `F` fit view to graph
- `Space+drag` pan (or right-click-drag)

### 6.8 Metadata inspector

When a single node is selected and a viewer is active, show the
incoming `Data.meta` dict as formatted text in a side panel.
(`window.py:metadata_view` in the old code.)

### 6.9 Error display

A panel listing nodes with errors, with their stack traces.
Click → focus that node in the canvas.

---

## 7. Performance requirements

This is the hardest non-feature requirement. The browser must stay
responsive under realistic loads:

- **Stress scenario**: a patch with 10+ viewers active simultaneously,
  including:
  - 1 HD video stream at 30 fps (~180 MB/s of raw frame data, ~10 MB/s
    after browser-side downscale)
  - 4 EEG buffers at 1 kHz × 32 channels (modest)
  - 4 PSD viewers updating at 30 Hz
- **Targets**:
  - UI stays at 60 fps (no janky pans, no dropped frames during drag)
  - Each viewer renders at its incoming data rate (or downsampled
    intelligently if rate > 30 Hz)
  - Memory does not grow unboundedly — bound per-viewer history,
    release Data when out of scope
  - WebSocket data backpressure: if the browser can't keep up, drop
    older frames on the backend side (the iceoryx2 layer already does
    this — surface that semantics to the WS layer)
- **Techniques**:
  - uPlot for line plots (fastest pure-JS time series)
  - `<canvas>` with `requestAnimationFrame` for images; for HD video
    consider `OffscreenCanvas` + worker
  - Pause data delivery for viewers off-screen (IntersectionObserver)
  - Bundle multiple slot updates into one WS message if rate is very
    high (optional optimization)

---

## 8. Tech stack — recommendations

These match goofi3's choices and are well-suited:

| layer | pick |
|---|---|
| frontend framework | **Svelte 5** + **SvelteKit** (static adapter) |
| build | Vite |
| typing | TypeScript strict |
| node graph | **Svelte Flow** (`@xyflow/svelte`) — zoom, pan, edges, minimap built in |
| time-series plot | **uPlot** |
| general plotting | canvas + custom (avoid heavy chart libs for hot paths) |
| WS / fetch | native |
| msgpack (codec meta) | `@msgpack/msgpack` |
| backend bridge | **aiohttp** (async, websockets built in; lower magic than FastAPI for this use) |

Justify deviations in your commit messages if you pick differently.

---

## 9. Validation

You have screenshot / browser-automation tools available. Use them:

- **Playwright** — install it, drive Chromium headless, take screenshots
  of every screen and every state. Verify visually that nothing
  overlaps, that the node editor zooms correctly, that the param panel
  doesn't clip, etc.
- For each viewer type, write a test that loads a known patch, waits
  for data, takes a screenshot, and compares it to a baseline (image
  diff with a tolerance — `pixelmatch` or similar).
- For performance: open the stress patch, run for 30 seconds, dump
  Chromium DevTools Performance trace, assert no frames > 50 ms.
- The user said: *"Make sure to pay close attention to z layering."*
  Test panel/menu/popover overlaps explicitly — screenshot with every
  panel open, with the add-node menu over the canvas, with the param
  panel over a viewer, etc.

Set up an `e2e/` directory:

```
e2e/
    playwright.config.ts
    fixtures/         # known patches
    tests/
        editor.spec.ts
        viewers.spec.ts
        params.spec.ts
        stress.spec.ts
```

---

## 10. Getting started

Read these in order:

1. **`src/goofi/manager.py`** — `Manager`, `NodeContainer`. Skim it.
2. **`src/goofi/node_helpers.py`** — `NodeRef`, especially `_messaging_loop`
   and `set_data_handler`. This is what the bridge hooks into.
3. **`src/goofi/codec.py`** — wire format. Port the encoder/decoder
   to TypeScript for the browser.
4. **`src/goofi/params.py`** — param classes, including `serialize`.
5. **`src/goofi/data.py`** — `Data` shape; meta conventions.
6. **`src/goofi/gui/window.py`** — what to replace. Don't deep-dive; use
   it as a feature reference (§6).
7. **`~/projects/goofi-next/goofi3/frontend/`** — reference architecture.
   Read `src/lib` and `src/routes`. **Don't copy code.**
8. **`test.gfi`** in the repo root — the stress patch the user has been
   testing with (Oscillator + PSD + 8 Buffers + VideoStream). Use it as
   your reference workload.

To run the existing backend:

```bash
cd ~/projects/goofi-next/goofi-pipe
uv run goofi-pipe --headless test.gfi           # no UI, runs the patch
uv run goofi-pipe --headless --duration 5 test.gfi   # auto-stop
```

The venv is at `.venv/` already. `uv pip install -e ".[dev]"` if you
need to refresh.

If `/dev/shm/iox2_*` accumulates after a crash:

```bash
.venv/bin/python -c "import iceoryx2 as i; i.Node.try_cleanup_dead_nodes(i.ServiceType.Ipc, i.config.global_config())"
```

---

## 11. Repo layout (after your work)

```
goofi-pipe/
    src/goofi/
        bridge/                  ← NEW: HTTP + WS server module
            __init__.py
            server.py
            control.py
            data.py
        manager.py               ← lightly tweaked to launch bridge
        node.py                  ← unchanged
        node_helpers.py          ← unchanged
        transport.py             ← unchanged
        codec.py                 ← unchanged
        nodes/                   ← unchanged
        gui/                     ← DELETED entirely
        assets/                  ← DELETED (or moved if anything is reused)
    frontend/                    ← NEW: SvelteKit SPA
        package.json
        src/
            routes/
            lib/
        build/                   ← gitignored; served by bridge
    e2e/                         ← NEW: Playwright tests
    examples/                    ← unchanged (.gfi patch fixtures)
    tests/                       ← unchanged (Python tests)
    pyproject.toml               ← add aiohttp dependency
```

---

## 12. Workflow + git

- You are on branch **`dev`**, which already has the backend refactor
  (commit `3acfe3c`).
- Make a new branch `feat/frontend` off `dev` for your work.
- Commit frequently, small focused commits. The previous session's
  log style is the model.
- Don't force-push without authorization.
- When you have a working MVP (canvas + one viewer + load patch),
  open a draft PR against `dev` so the user can preview.

---

## 13. Out of scope

- Reworking the Python tests
- Touching `transport.py`, `codec.py`, the iceoryx2 setup
- Reintroducing zmq, dearpygui, or any old IPC
- Authentication on the WS endpoints (single-user local app for now)
- Mobile / touch optimization (desktop browser only)
- Theming / dark-mode toggle (pick one, do it well)
- Multi-instance (one manager ↔ one browser tab is enough)

If you discover you need to break something on this list, surface it to
the user before doing it.

---

## 14. Goal condition

You are done when **all** of the following hold:

1. `uv run goofi-pipe` (no flags) launches the manager, prints the URL,
   and opening that URL in a browser shows the editor.
2. Every feature in §6 works (verified by Playwright tests in `e2e/`).
3. The stress patch (§7) runs for 60 s with 10+ visible viewers, the
   browser stays at ≥ 55 fps median, no JS console errors, no Python
   tracebacks.
4. `git grep -i dearpygui src/goofi/` returns nothing. `src/goofi/gui/`
   does not exist. `pyproject.toml` no longer lists dearpygui.
5. Playwright screenshot tests pass for every viewer type, the
   add-node menu, the param panel, save/load, and the multi-select
   marquee. No visual regressions, no z-order glitches.
6. The 128 existing Python tests (`pytest tests/`) still pass.
7. The user can `git log` and see a clean, readable history of focused
   commits — not one mega-commit.

When you think you're done, screenshot the editor, the param panel,
two viewer types side-by-side, and the add-node menu, and put them in a
brief summary message for the user.

---

## 15. P2P viewer-data plane + node-side thalamus (DESIGN — not yet built)

**Full spec:** [`docs/p2p-data-thalamus-spec.md`](docs/p2p-data-thalamus-spec.md) —
authoritative, self-contained, implementable end-to-end. This section is the
30-second summary; the spec is the source of truth.

**Problem.** The `/data` plane ships the *full* `Data` every frame with zero
reduction: `bridge/data.py:on_frame` re-encodes the whole decoded array. Measured:
a 44.1 kHz / 60 s mono buffer is **~10.6 MB/frame** → **~318 MB/s** at 30 Hz into a
~1000 px plot that needs ~2k points (a ~1300× oversend). Worse, the path is **not
P2P**: `/data/<node>/<slot>` is served by the bridge *inside the manager process*
(`server.py`: "Lives in the manager process"), whose `NodeRef._data_pump` opens its
own iceoryx2 subscriber and **decodes every full frame in the manager** before
re-encoding — three codec passes + a full SHM copy of unreduced data. This is a
dearpygui-era hook the browser bridge inherited. node↔node data (iceoryx2) and
logs (`node_log.py` SSE) are *already* P2P; only the viewer-data path routes
through the manager.

**Design.** Move the viewer-data path **peer-to-peer**, mirroring `node_log.py`:

- Each **node-host process** hosts one tiny binary-WebSocket server
  (`src/goofi/node_data.py`, NEW — stdlib hand-rolled RFC6455), advertises its
  **port** as `data_port` over `STATE_UPDATE` exactly like the log endpoint. The
  browser discovers it from the control plane, composes the URL from
  `location.hostname`, and connects **directly** to the node.
- A **dedicated per-process reducer thread** does the reduction off the processing
  thread: `_processing_loop` only does an O(1) `node_data.offer(node, slot, data)` —
  a **private snapshot** (array + meta copy on the node thread, so it's race-free vs.
  in-place mutators like `LatentRotator`) into a latest-wins mailbox. The reducer
  thread runs `reduce_for_view` (`src/goofi/node_reduce.py`, NEW) + `encode_data` and
  fans the bytes to per-connection mailboxes. The node's tick — and its node↔node
  output rate — is **not** slowed by reduce/encode. The manager leaves the data path
  entirely (`bridge/data.py` + `NodeRef.set_data_handler`/`_data_pump` deleted).
- Reduction is **per-axis and viewer-defined**: `ViewSpec = { axes: [{axis, max,
  method}], version }`. Each viewer declares which axes to reduce, how far, and the
  method — `envelope` (min/max per bin; waveforms, never stride), `area` (block-mean
  downscale; images), `subsample` (linspace gather; channels, trajectory paths).
  `reduce_for_view` composes the per-axis reductions; unlisted axes pass through.
  **Fail-open** (any error → input unreduced).
- **Meta inspector is unaffected by reduction.** One stream per slot, but the reduced
  frame carries `meta['reduced'][axis] = {orig_len, method, orig_coord?}`; body
  coord arrays co-reduce to satisfy `Data.__post_init__` (`data.py:104`), and
  `MetadataPanel` is reduction-aware — it reconstructs and shows the **true original
  meta**. Reduction is purely a viewer-rendering concern.
- The **frontend thalamus** (`thalamus.svelte.ts`, `dataStream.svelte.ts`) folds all
  live viewer-consumers per `(node,slot)` into ONE `ViewSpec` (per-axis largest-max;
  method: `envelope`>`area`>`subsample`), sent **inband** on the per-node WS. One
  stream per slot; expand/collapse + IntersectionObserver add/remove consumers.
- **Host scope = the frontend's.** `Manager.__init__` exports `GOOFI_BIND_HOST`
  (default `0.0.0.0`), inherited by node processes; `node_log` + `node_data` bind it,
  so logs and viewers share the manager's reachability (LAN-reachable when `--bind`
  is). No auth, CORS `*` — single-user/trusted-LAN scope, same as the manager bridge.

**Untouched (HARD):** node↔node iceoryx2 transport, `codec.py`, the zero-copy SHM
publish path. The reduced browser frame is a *separate* encode of a *separate*
reduced `Data` snapshot.

**Remaining accepted limitation:** `viewer_count` does not cascade upstream, so
viewing a purely input-triggered leaf whose upstream is idle shows nothing (sources
free-run, so this matches `test.gfi`). (The earlier tick-cadence and 127.0.0.1
trade-offs are now *resolved* by the reducer thread and the shared host scope.)

**STATUS 2026-06-22 — relationship to the shipped viewer-adapters plane.** This
spec was written (2026-06-16) *before* the viewer-adapters data plane that now
ships on `frontend`. The two are **alternative** reductions of the same problem,
not additive:

- **Shipped today (viewer adapters):** the bridge decodes each slot once and
  re-encodes per viewer *kind* with a **dtype** downcast at the boundary
  (`bridge/adapters.py`: image→uint8, line/etc→float16) over
  `/data/<node>/<slot>/<kind>`. The manager **stays** in the data path; reduction
  is bit-depth only (~2–4×), full resolution still crosses the wire.
- **This spec (P2P thalamus):** node-side per-axis reduction to the viewer's actual
  *capacity* (~1300× for a kHz buffer; HD→viewer-pixels), **P2P**, manager removed
  from the data path. It is strictly more powerful and directly removes the
  measured perf ceiling (manager decode→re-encode + float32). §9.1 of the spec
  **deletes `bridge/data.py`** — i.e. adopting the thalamus **supersedes** the
  adapters plane (they cannot coexist).

So the thalamus is the recommended **next major project** (a clean 10-step plan,
§11), not a rebase of the current code. When undertaken, fold the per-kind adapter
*intent* into the per-axis `ViewSpec` and delete `adapters.py` + the `/kind` route.
Until then the adapters plane is the shipped reduction. The design branch this came
from (`design/p2p-thalamus`) has been deleted now that the spec lives here.
