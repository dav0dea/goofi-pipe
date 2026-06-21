# goofi-pipe — Major Fixes Implementation Plan (2026-06-20)

> Backing analysis: `docs/analysis/2026-06-20-deep-analysis.md` (sections A–L).
> Cite that report for full evidence/repro; this plan is the execution spec.

## Scope (confirmed with user)

- **In scope:** (1) the **performance** unlock to reach "tens of HD streams" —
  *full depth*, including the browser Worker + OffscreenCanvas/WebGL rewrite; (2) the
  **critical correctness bugs** that break core workflows; (3) the **security** default.
- **Out of scope (deferred, tracked in the report):** §K systemic signal-node deepcopy
  sweep, §L I/O-node terminate()/blocking sweep, and the unfinished-feature work
  (TableViewer recursive tree, Examples menu, history.lastError toast, fps/HUD,
  sub-patch viewer state). These are listed in "Deferred" at the end.

## Conventions

- **TDD (Iron Law, house style):** no production code without a failing test first.
  Seams already exist: `FakeControl` (`frontend/src/lib/test/fakeControl.ts`), pure
  executors, pytest for backend, vitest for FE units, Playwright `e2e/` for integration.
- **Test commands:**
  - Backend: `rm -f /dev/shm/iox2_*; .venv/bin/python -m pytest tests/ -p no:cacheprovider -q`
  - FE units: `cd frontend && npm test` (vitest) · typecheck `npm run check`
  - e2e (build first): `(cd frontend && npm run build); .venv/bin/python -m pytest e2e/ -q`
    (e2e is gitignored — use plain `grep`/`ls`, not `git`).
- **Branch:** create `fix/perf-and-critical` off the current `feat/undo-redo` head.
  One focused commit per sub-task; commit message style per the repo log. Do not push
  without authorization.
- **No-regression gates per phase:** the named new tests pass; `pytest tests/` stays
  green (currently **925 passed**); `npm run check` clean; the stress patch (`test.gfi`)
  still runs.
- **New perf harness (addresses the §7/test-coverage gap):** add a throughput
  micro-benchmark + an e2e fps assertion (Phase 2 and Phase 7). This is the objective
  acceptance test for the whole effort.

## Acceptance (whole effort)

Open the stress patch with ≥10 visible viewers incl. several HD video streams for 60 s:
browser median **≥55 fps**, no JS console errors, no Python tracebacks, memory bounded.
Manager-side per-frame work ≈ 0 (no decode/re-encode); HD paint off the main thread.

---

## Phase 1 — Small, safe correctness + security (land first)

Low-risk, high-signal; unblocks the rest and is independently shippable.

### 1.1 — Security: default bind to loopback (report F1)
- **Files:** `src/goofi/bridge/server.py` (`BridgeServer.__init__` host default `:154`,
  `start_bridge` `:297`), `src/goofi/manager.py` (`bridge_host` default `:133`, `--bind`
  default `:1576`).
- **Change:** default host `"0.0.0.0"` → `"127.0.0.1"` everywhere. Keep `--bind` so a
  user can opt into `0.0.0.0`; when a non-loopback host is chosen, print a clear
  one-line warning ("UI exposed on the network with NO authentication; expressions
  execute arbitrary Python — bind 127.0.0.1 unless you intend this").
- **Test:** `tests/test_bridge*`/`test_fsbrowse` style — assert the default
  `Manager(...)` / arg parser yields `127.0.0.1`. (No socket needed.)
- **Risk:** trivial. Remote-access users must now pass `--bind 0.0.0.0`.

### 1.2 — StringViewer XSS (report B14)
- **File:** `frontend/src/lib/viewers/StringViewer.svelte:14,18`.
- **Change:** sanitize before `{@html}`. Prefer adding `dompurify` and
  `DOMPurify.sanitize(marked.parse(text))`; or configure `marked` to disable raw inline
  HTML. (Check `min-release-age`/cooldown policy when adding the dep.)
- **Test:** vitest unit on a sanitize helper: `<img src=x onerror=alert(1)>` →
  stripped; normal markdown preserved. (Component can't mount in vitest — extract the
  sanitize+parse into a pure `renderMarkdown(text): string` helper and test that.)
- **Risk:** low.

### 1.3 — Node expressions iterated un-snapshotted (report H4)
- **File:** `src/goofi/node.py:747`.
- **Change:** `for key, engine in list(self._expressions.items()):` (match `:636,:786`).
- **Test:** pytest — drive a node whose `_expressions` is mutated (SET_EXPRESSION)
  concurrently with a processing tick; assert no `RuntimeError`. If hard to race
  deterministically, a unit test that calls the drain helper while mutating the dict and
  asserts it doesn't raise. (Pairs with Phase 4.)
- **Risk:** trivial; one-liner.

### 1.4 — Undo/redo re-entrancy guard (report B13 + record-during-replay B-fh4)
- **File:** `frontend/src/lib/stores/history.svelte.ts` (`undo()` `:315`, `redo()` `:332`).
- **Change:** add a private `#busy`/`isReplaying` flag. At the top of `undo()`/`redo()`:
  `if (this.#busy) return; this.#busy = true; try { … } finally { this.#busy = false }`.
  This closes the "held Ctrl+Z double-replays + double-pops" window (two awaits before
  `pop()`). For B-fh4 (user action during in-flight replay silently dropped): acceptable
  for MVP since replays are short, but optionally surface it — document the limitation in
  the spec; do not expand scope here.
- **Test:** `frontend/src/lib/stores/history.test.ts` — call `undo()` twice without
  awaiting the first; assert only one action moved to the redo stack and the executor's
  inverse ran once (via `FakeControl.recordedCalls()`). Add an e2e holding the key if
  cheap.
- **Risk:** low; pure store change, covered by existing history tests.

**Phase 1 done-when:** the four tests pass; `pytest`/`npm run check` green.

---

## Phase 2 — Backend raw-byte forwarding (THE perf unlock)

Eliminates the manager-side decode→re-encode (measured 2.57 ms/frame uint8, 10.3 ms
float32 → ~3-stream GIL ceiling). The pump already holds the wire bytes; forward them.

### 2.1 — Raw data-handler path (report A1)
- **Files:** `src/goofi/node_helpers.py` (`set_data_handler` `:413`, `_data_pump`
  `:450-486`, `_data_handlers` tuple `:247`), `src/goofi/bridge/data.py`
  (`on_frame` `:160-171`).
- **Change:**
  1. `NodeRef.set_data_handler(slot, callback, *, raw: bool = False)`. Store the `raw`
     flag in the `_data_handlers[slot]` tuple (now `(sub, listener, callback, raw)`).
  2. In `_data_pump` (`:471-482`): when `raw` is set, **skip `decode_data`** and call
     `cb(self, slot_name, buf)` with the raw GOOF `bytes` from `take_latest()`. When not
     raw, keep the current decode path (other potential consumers unaffected).
  3. In `bridge/data.py` `_SlotMux`: register with `raw=True`; `on_frame(noderef, slot,
     buf)` becomes `_mux.dispatch(buf)` — **no `prepare_encode`/`encode_data_into`/
     `bytes()`**. `buf` is already a heap `bytes` (copied out of SHM by `take_latest`),
     safe to fan out by reference.
- **Why correct:** producers `encode_data_into` the exact GOOF frame into the SHM loan;
  `take_latest()` returns those bytes; the browser decodes with its own TS codec. The
  manager never needs the decoded `Data` for forwarding. The ExpressionEngine uses its
  **own** subscribers (not `set_data_handler`), so it is unaffected.
- **Cleanup:** fix the now-true `data.py` module docstring ("forwards verbatim").
- **Tests:**
  - `tests/test_datahub_mux.py` (extend): a node publishes a known `Data`; a raw handler
    receives bytes that `decode_data()` round-trips to the original. Assert **no**
    re-encode occurred (e.g. the forwarded bytes are *identical object/content* to
    `take_latest`, and a spy confirms `encode_data_into` is not called on the forward
    path).
  - `tests/test_node_helpers.py`: `set_data_handler(raw=True)` delivers raw bytes;
    `raw=False` still delivers decoded `Data` (back-comat).
- **Risk:** medium — touches the hottest backend path. Mitigated by the round-trip test
  and keeping the decoded path intact for `raw=False`.

### 2.2 — Don't block the asyncio loop on viewer disconnect (report B2)
- **File:** `src/goofi/bridge/data.py:174-188` (the `finally` teardown).
- **Change:** keep the mux bookkeeping (`mux.remove`, `_muxes.pop`) under `self._lock`,
  but run the **blocking** `ref.set_data_handler(slot, None)` (WaitSet detach + iceoryx2
  subscriber close + UNREGISTER_SUBSCRIBER IPC) **outside** the lock via
  `await loop.run_in_executor(None, ref.set_data_handler, slot, None)`. Same for the
  connect-time `set_data_handler(slot, on_frame)` if it shows up in profiling.
- **Also fix the listener leak (report B4):** in `node_helpers.set_data_handler`'s
  unregister branch (`:426-431`), after detaching, call `prev[1].close()` on the
  IpcListener (guarded) — currently only the subscriber is closed (expression.py closes
  both; mirror it).
- **Test:** pytest — rapid subscribe/unsubscribe cycles on a slot don't leak listeners
  (assert a bounded count) and don't deadlock; an async test that a second slot's send
  is not starved during a teardown (harder — may settle for the unit-level leak test +
  manual profiling note).
- **Risk:** medium (async ordering). Keep teardown idempotent.

### 2.3 — (Optional, profile-gated) zero-copy take (report A4)
- Add `IpcSubscriber.take_latest_view()` returning a held `Sample` + `memoryview`;
  forward the memoryview and release the Sample after the WS frame is queued. Removes the
  remaining SHM→bytes copy. **Defer unless** the Phase-2 benchmark shows the take-copy is
  still a bottleneck after 2.1. `transport.py` is otherwise out of scope (CLAUDE.md §13).

### 2.4 — Perf harness (new; acceptance for the backend unlock)
- Add `tests/test_dataplane_perf.py` (marked slow / opt-in): publish N HD frames through
  a raw handler vs the old decode→re-encode path; assert raw is ≥5× faster and allocates
  no ndarray. This both verifies 2.1 and fills the §7 "no perf test" gap.

**Phase 2 done-when:** raw forwarding verified by round-trip + no-re-encode tests; the
perf harness shows the manager per-frame cost collapse; `pytest` green; `test.gfi` runs.

---

## Phase 3 — uint8 image convention (perf unlock #2; 4× wire/bandwidth)

Images are emitted `float32/255` everywhere (~25 MB/HD frame); the viewer just re-clamps
to uint8. Emitting uint8 cuts every image's wire bytes 4× (746→187 MB/s/stream).

### 3.1 — Switch image producers to uint8 (report A0)
- **Producers (emit uint8 instead of float32/255):** `src/goofi/nodes/inputs/
  videostream.py:105`, `loadfile.py:150,152` (image branch), `imagegeneration.py:192`,
  `misc/edgedetector.py:62`.
- **Consumer audit (must handle uint8 RGB, [0,255]):** the image-consuming/transforming
  nodes — `misc/{colorenhancer,hsvtorgb,rgbtohsv,hologram,edgedetector}.py`,
  `analysis/{facelandmarker,facialexpression,bodyposeestimation,poseestimation,
  img2txt,audiotagging?}` — grep `nodes/` for `dim2`/image-shaped inputs and any code
  doing `* 255` / assuming `[0,1]`. Add a tiny shared helper (e.g. in `convenience.py`)
  `as_float01(img)` / `as_uint8(img)` and call it at the point a node genuinely needs
  floats; otherwise pass uint8 through.
- **Viewer:** `ImageViewer` already branches on `isU8` vs `isFloat` (`:78-83`) — no
  change needed; verify the uint8 fast path.
- **Tests:** pytest per touched producer — output dtype is `uint8`, shape unchanged,
  values match the pre-change `(float*255).round().clip` within ±1. e2e: load a patch
  with VideoStream→ImageViewer, assert it still renders (screenshot non-blank) and the
  decoded frame dtype is `|u1` (via `window.goofi` frame summary).
- **Risk:** medium-wide — the consumer audit is the real work. Keep it bounded to nodes
  that actually take an image input; when unsure, cast-to-float in that consumer (safe).
- **Note:** this is the one item touching `nodes/` — justified because it's the
  cheapest large perf win and is a *convention*, not the §K correctness sweep.

**Phase 3 done-when:** producers emit uint8; audited consumers green; VideoStream→
ImageViewer e2e renders; per-stream wire bytes drop ~4× (verify via frame size).

---

## Phase 4 — Node-runtime thread safety (report H1/H2/B3/I2)

The per-node WaitSet and ExpressionEngine are mutated from the messaging thread + engine
while the processing thread uses them, with no lock — can crash a node's loop or stop it.

### 4.1 — Lock the WaitSets (reports H1 + B3)
- **Files:** `src/goofi/node.py` (`self._waitset` attach/detach `:660,667,395-396`, wait
  `:727`) and `src/goofi/node_helpers.py` (`_data_waitset` attach/detach `:426,439`, the
  pump `wait()` `:465`). Underlying `WaitSet` internals: `transport.py:494-552`.
- **Change:** give each owner a dedicated `threading.Lock` guarding **all** WaitSet
  attach/detach/wait for that WaitSet, OR (preferred, less lock-contention) marshal all
  mutations onto the owning loop: queue subscribe/unsubscribe/attach requests and apply
  them at the top of the loop before `wait()`. Apply the same pattern to both WaitSets.
  The engine callbacks (`on_listener_added/removed`) must go through the same path.
- **Test:** pytest stress — wire/unwire a link (and set/clear an expression) repeatedly
  while a node processes; assert no `RuntimeError: dict/list changed size`, no lost
  listeners (the node keeps producing). Run forked/repeated (the harness flakes on
  fork+iceoryx2 — compare pass rates, clean `/dev/shm/iox2_*`).
- **Risk:** medium — concurrency. Prefer the marshal-onto-loop design (no new lock-order
  hazards). Keep the lock scope tight.

### 4.2 — Serialize ExpressionEngine evaluation (report H2)
- **File:** `src/goofi/node.py` (`_apply_expression` from both threads `:407,638,749,789`).
- **Change:** a per-node (or per-engine) lock around evaluate/mutate so the same engine
  can't run concurrently from the messaging and processing threads. Combine with 4.1's
  marshalling if that already serializes it.
- **Test:** pytest — concurrent SET_EXPRESSION + processing eval doesn't corrupt the
  engine's subscription bookkeeping (assert stable ref set / no exception).

### 4.3 — Drain non-triggering input slots (report I2)
- **File:** `src/goofi/node.py:646-660,738-766` (the `if slot.trigger_process` gate).
- **Change:** **decouple draining from triggering** — attach a listener and drain
  `slot.data` for ALL subscribed inputs; gate only *whether a fire triggers process()* on
  `trigger_process`. Today non-triggering inputs are never read → `slot.data` stays None.
- **Test:** pytest — a node with one triggering + one non-triggering input, fed both,
  sees real data on the non-triggering slot in `process()`.
- **Risk:** medium — changes core trigger semantics; verify existing node tests pass
  (some nodes may rely on the current behavior — check, adjust).

**Phase 4 done-when:** the race stress tests pass; non-triggering inputs deliver data;
full `pytest` green (watch for nodes depending on old trigger semantics).

---

## Phase 5 — Load + editor correctness (reports B1, B8, B9)

### 5.1 — Load no longer breaks live state/error forwarding (report B1)
- **File:** `src/goofi/bridge/control.py` (`_replace_graph` `:484-496`, `_wire_node_status`
  early-return `:431-434`, discard sites `on_node_removed :368`/`on_node_renamed :380`).
- **Change:** in the `graph_replaced` flow, **clear the hub's wiring state** before/after
  the destructive teardown: `self._wired_nodes.clear()` so reloaded nodes (which reuse
  display names) get their STATE_UPDATE/PROCESSING_ERROR handlers re-registered. (Or
  route the teardown removals through `on_node_removed` so the discard fires per node.)
- **Test:** `tests/test_control_ops.py` — load patch A, then load patch B that reuses
  names; assert each reloaded node's NodeRef has a state/error handler registered (or
  that a subsequent state_update is broadcast). e2e: load → trigger a param echo → assert
  the param panel updates; cause a node error → assert it appears.
- **Risk:** low-medium; verify no double-wiring.

### 5.2 — Preserve marquee/box selection across flowNodes rebuild (report B8)
- **File:** `frontend/src/lib/panels/NodeEditorPanel.svelte` (nodes effect `:227-300`,
  edges effect `:302-374`, `selectedNodeNames` `:385`).
- **Change:** the rebuild sets `selected` from `sel.nodes(panelId)` only, dropping Svelte
  Flow's live marquee `selected`. Either (a) **sync** Flow's `onselectionchange` into the
  selection store so the store is authoritative (then rebuild preserves it), or (b)
  preserve the prior `selected` flag: read current `flowNodes`/`flowEdges` selected state
  and OR it into the rebuilt nodes by id. Prefer (a) — it also fixes the union hack in
  `selectedNodeNames`.
- **Test:** e2e — marquee-select 3 nodes, trigger a graph change (move a 4th node / a
  state update), assert the 3 stay selected. (vitest can't mount the editor; rely on e2e.)
- **Risk:** medium — selection sync interactions; verify group/copy/duplicate still read
  the right set.

### 5.3 — Paste in flow coordinates (report B9)
- **File:** `frontend/src/lib/panels/NodeEditorPanel.svelte:743-758` (`pasteClipboard`).
- **Change:** replace the `[innerWidth/4, innerHeight/4]` screen anchor with a flow-space
  anchor via Svelte Flow's `screenToFlowPosition` at the cursor (or viewport center). The
  per-node offsets in `clipToSpecs` then land relative to a correct anchor.
- **Test:** e2e — pan/zoom the editor, paste, assert pasted nodes appear in the visible
  viewport (positions within the current flow bounds). (Duplicate already uses a relative
  offset and is fine.)
- **Risk:** low.

**Phase 5 done-when:** Load e2e shows live updates after reload; marquee survives a
rebuild; paste lands in view; `pytest`/e2e green.

---

## Phase 6 — Browser data plane in a Web Worker (perf unlock #3)

Move WS receive + decode + latest-wins coalescing OFF the main thread. The single choke
point is `subscribeFrames` (`frames.ts`); reimplement it behind a worker, preserving its
signature so **no viewer changes** are needed here.

### 6.1 — Coalesce BEFORE decode (report A11)
- **Today:** `data.ts:38` decodes every WS message eagerly; `frames.ts` coalesces *after*
  decode (so kHz EEG decodes ~940 wasted frames/s).
- **Change:** stash the raw `ArrayBuffer` per (node,slot) as `pendingRaw`; decode **only**
  the buffer that survives to the flush tick. (In the worker design below this is natural:
  the worker holds latest raw per slot and decodes on its tick.)

### 6.2 — Worker owns WS + decode (report A13)
- **New file:** `frontend/src/lib/api/dataWorker.ts` (a module Worker). It:
  - opens one WS per (node,slot) (same URL/reconnect/4000-terminal logic as `data.ts`),
  - keeps `latestRaw` per slot, on a ~display-rate tick decodes the survivor with the
    existing `decode.ts` (imported into the worker),
  - `postMessage({node, slot, frame}, [transferList])` transferring the decoded
    `values.buffer` (zero-copy) to the main thread.
- **Rewrite `frontend/src/lib/api/data.ts` + `frames.ts`** as a thin **main-thread proxy**
  over the worker: `subscribeFrames(node, slot, cb)` keeps its signature; it posts
  `{op:'sub', node, slot}` to the worker, refcounts consumers, routes `onmessage` frames
  to consumers, and maintains the `latestFrame(node,slot)` cache for `query.frameSummary`
  / the agent surface. `subscribeData` (raw, every-frame) can remain for any non-viewer
  consumer or be removed if unused (grep first).
- **Preserve:** the agent automation surface (`latestFrame`, `query.frameSummary`) and
  IntersectionObserver gating in `ViewerFeed` (unchanged — it still calls `subscribeFrames`).
- **Tests:** vitest — the worker decode/coalesce logic is pure-ish; extract the
  coalesce+decode core into a testable function and unit-test latest-wins + decode. e2e:
  viewers still receive frames; `window.goofi.query.frameSummary` still returns the
  current frame (worker round-trip intact). Add an e2e that a kHz stream decodes at ≈
  display rate, not source rate (assert via a frame-count probe).
- **Risk:** **high** — this restructures the data plane. Keep the `subscribeFrames` API
  byte-identical so the viewer layer and agent surface are untouched. Land behind the
  existing tests; verify reconnect + terminal-close still work through the worker.

### 6.3 — Zero-copy typed array for survivors (report A12)
- **File:** `frontend/src/lib/codec/decode.ts:170` (`readTypedArray` `.slice()`).
- **Change:** for the (now ~60/s) flushed survivors, avoid the per-frame full copy where
  alignment allows: construct the TypedArray as a view; fall back to an aligned copy only
  when the body byteOffset isn't element-aligned (the GOOF header+meta makes f4/f8 often
  misaligned, so a single aligned copy per survivor is acceptable — the win is doing it
  ~60×/s not 600–1000×/s). In the worker, the decoded buffer is transferred to main, so
  no main-thread copy remains regardless.
- **Test:** vitest — `decode.ts` round-trips all dtypes (incl. misaligned offset) — this
  also fills the "codec TS decoder untested" gap (report test-coverage).

**Phase 6 done-when:** data WS + decode run in a worker; `subscribeFrames`/`latestFrame`
unchanged for callers; kHz streams decode at display rate; codec unit tests pass; e2e
viewers render with no console errors.

---

## Phase 7 — Browser render off the main thread (perf unlock #4)

### 7.1 — ImageViewer via WebGL (reports A3 + A14 + A18)
- **File:** `frontend/src/lib/viewers/ImageViewer.svelte` (replace the per-pixel JS loop
  `:50-121` + `putImageData`).
- **Change:** create a WebGL2 context on the canvas; upload the frame's typed array with
  `texImage2D`/`texSubImage2D` (RGB or R8 for grayscale — **no JS RGBA expansion**); a
  fragment shader does dtype→[0,1], colormap LUT (as a 256×1 texture), value-range, and
  the GPU does the scale to the element box. **Downsample to the CSS box** by sizing the
  drawing buffer to the element's on-screen pixels (ResizeObserver-cached), not the full
  data dims (~70× fewer pixels for an inline HD viewer). Keep a tiny 2D-canvas fallback if
  WebGL is unavailable.
- **Test:** e2e — image viewer renders a known frame; screenshot is non-blank and matches
  a tolerance baseline (pixelmatch — also fills the "viewers never pixel-asserted" gap).
  Unit-test the colormap LUT/scale shader inputs where extractable.
- **Risk:** medium-high — WebGL plumbing; the fallback bounds the blast radius.

### 7.2 — Global rAF paint scheduler with a per-frame budget (report A16)
- **File:** replace the per-slot rAF in `frames.ts`/the worker proxy (`:62-80`) with ONE
  main-thread scheduler: a dirty-viewer queue, paint visible+large first, stop once a
  per-frame budget (~8–10 ms, leaving room for Svelte Flow) is exhausted; defer the rest
  to next frame (latest-wins makes deferral free). Viewers register a `paint()` callback
  with the scheduler instead of repainting in their own effect.
- **Test:** e2e perf — open the stress patch (≥10 viewers incl. HD), run 30–60 s, assert
  median frame time ≤ ~16 ms (no frames > 50 ms) via a `requestAnimationFrame` probe or a
  DevTools trace. **This is the headline acceptance test.**
- **Risk:** medium — scheduler correctness (don't starve a viewer); prioritize fairness.

### 7.3 — ArrayViewer min/max-decimate to canvas width (report A15)
- **File:** `frontend/src/lib/viewers/ArrayViewer.svelte` (`pushData` `:303-352`).
- **Change:** when samples ≫ canvas pixel width, reduce each channel to ~2 points/column
  via min/max-per-bucket before `setData` (preserves spikes; standard scope technique).
  Cache the pixel width via ResizeObserver. Also fix the settings-change full-rebuild
  thrash (report ArrayViewer:358 — only `makePlot` on distr-affecting changes).
- **Test:** vitest on the decimation helper (pure): N samples → ≤ 2×width points,
  preserving per-bucket min/max. e2e: 1 kHz buffer renders smoothly.
- **Risk:** low-medium.

### 7.4 — (Stretch) OffscreenCanvas paint in the worker
- After 7.1–7.3, optionally move the WebGL paint itself into the worker via
  `transferControlToOffscreen()` per image viewer (worker routes frames straight to GPU,
  nothing crosses back). Only if the main-thread `texImage2D` cost still shows in the
  Phase-7 trace. Gated on evidence.

**Phase 7 done-when:** HD ImageViewer is WebGL (no per-pixel JS); one scheduler bounds
per-frame paint; line plots decimate; the stress-patch e2e holds ≥55 fps median.

---

## Suggested order & checkpoints

1. **Phase 1** (safe wins) → commit, ship.
2. **Phase 2** (raw forwarding) + **2.4 perf harness** → the backend unlock; benchmark
   before/after.
3. **Phase 3** (uint8) → 4× wire reduction; re-benchmark.
4. **Phase 4** (thread safety) → stability under graph edits while streaming.
5. **Phase 5** (Load/editor) → core-workflow correctness.
6. **Phase 6** (worker) → main-thread decode gone; verify agent surface intact.
7. **Phase 7** (WebGL + scheduler) → main-thread paint gone; run the acceptance e2e.

Each phase is independently shippable; after Phases 2–3 the manager ceiling is lifted
even before the browser rewrite, so perf can be validated incrementally.

## Deferred (tracked in the analysis report; not in this plan)

- §K systemic signal-node **deepcopy-meta** sweep (PSD/Padding/Reduce/Join/… + the
  `sliding_window` cluster, Hilbert sfreq, Normalization int-truncation, broken
  Binarize/Avalanches). One repeatable fix pattern; do as a focused follow-up.
- §L I/O-node **terminate()-hygiene + thread-blocking + unlocked-callback** sweep
  (sockets/zmq-ctx/`/dev/shm`/MIDI-ports/threads; ZeroMQIn/MidiOut blocking; AudioStream/
  OSCIn races).
- Unfinished features: `history.lastError` toast, sub-patch viewer-state persistence,
  TableViewer recursive tree, Examples menu wiring, data recording/export, fps/health HUD,
  crashed-node restart, edge reconnect, Firefox panel-drag (I5), FsBrowser modal guard (I6),
  PlacementPreview wrong-editor (I4).
- Dead-code/abstraction cleanups (§C): `flat_view`, the `233` triplication, unused
  exports, etc.
