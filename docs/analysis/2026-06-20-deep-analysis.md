# goofi-pipe — Deep Analysis Pass (2026-06-20)

Analysis window: **16:25 → 18:25 EDT** (2h). Branch: `feat/undo-redo`.
Method: parallel subsystem workflows with adversarial per-finding verification,
plus firsthand reads of the highest-impact hot paths.

Severity legend: 🔴 critical · 🟠 high · 🟡 medium · ⚪ low

---

## Executive summary

The frontend rewrite is mature and the Python baseline is green (**925 passed, 6
skipped**). The architecture (process-per-node + SHM + browser-side decode) is sound.
But **the stated scale ("tens of HD streams at ease") is not met as built** — measured
ceiling is **~5–13 HD streams** — and the cause is implementation waste on the two
un-shardable chokepoints (the manager GIL and the browser main thread), not a design
flaw. There is also a cluster of real correctness bugs and a security-exposure default.

**Top priorities (in order):**

| # | Fix | Why | Refs |
|---|-----|-----|------|
| 0 | **Emit video as uint8, not float32** (`videostream.py:105` does `astype(f32)/255`) | HD frame is **~25 MB not 6 MB** → every data-path cost ×4; the viewer just re-clamps to uint8. Trivial change = **4× headroom instantly** | A0 |
| 1 | **Forward raw GOOF bytes through the manager** (skip decode→re-encode) | Measured 2.57 ms/frame (uint8) → ~0; lifts the GIL ceiling. *The* unlock. | A1, A4–A6 |
| 2 | **Move browser data WS + HD paint off the main thread** (Worker + OffscreenCanvas/WebGL; coalesce-before-decode; drop per-frame `.slice()`) | The other hard wall; 20 HD viewers = 10–29× over the 16 ms budget today | A3, A11–A14 |
| 3 | **One global rAF paint scheduler with a per-frame budget** | Bounds main-thread time regardless of viewer count | A16 |
| 4 | **Load silently breaks live state+error forwarding** — clear `_wired_nodes` on graph_replaced | Core Load workflow: params/errors stop updating until manager restart | B1 |
| 5 | **Guard undo/redo against re-entrancy** | Holding Ctrl+Z double-replays + desyncs both stacks | B13 |
| 6 | **Viewer disconnect stalls the whole data-plane loop** — run the blocking teardown off-loop | Every off-screen unsubscribe freezes all other viewers | B2 |
| 7 | **Default bind to `127.0.0.1`** (today `0.0.0.0` + no auth + expression `exec` = LAN RCE) | Security | F1 |
| 8 | **Marquee selection wiped on graph rebuild**; **paste ignores pan/zoom** | Core editor UX | B8, B9 |
| 9 | **Lock `node.py self._waitset` + serialize expression eval** (two threads mutate it lock-free) | Can crash a node's processing loop / silently stop it processing | H1, H2, H4 |
| 10 | **`sliding_window` off-by-one corrupts windowed results** | Silent data-correctness bug in a signal-processing helper | H3 |
| 11 | **Surface `history.lastError`** (failed-undo toast) + **finish sub-patch viewer-state persistence** + **TableViewer recursive tree** + **Examples menu wiring** | Unfinished features that leave dead-ends | G, D1, D2 |

Counts: **~130 confirmed bugs** across six adversarial waves (find→refute; ~75 false
positives dropped), a full performance bottleneck list (A0–A18, empirically validated:
~2.57 ms/frame uint8 / ~10.3 ms/frame for real float32 video → a **~3 HD-stream GIL
ceiling** today), ~20 dead-code/abstraction items, ~20 unfinished/missing-feature gaps,
a systemic signal-node correctness cluster (§K), and a user-facing I/O-node debt cluster
(§L: terminate-leaks, thread-blocking, unlocked races). Detail below.

---

## A. Performance at target scale (tens of HD streams + audio + biosignals)

### A0. 🟠 Video is emitted as float32 — 4× the necessary data (firsthand-measured)

`videostream.py:105` returns `frame.astype("float32") / 255.0`, so an HD frame on the
wire is **~24.9 MB**, not the 6.2 MB a uint8 frame would be. The browser `ImageViewer`
then scales floats back with `Math.round(v*255)` — i.e. the float32 carries **no extra
information**; it's immediately re-quantized to 8-bit for display. Net effect: every
copy/decode/encode/WS-send in the data path moves **4× more bytes than needed**.

**Measured impact:** the decode→re-encode path on a real float32 HD frame is **10.3
ms/frame → ceiling ~97 fps ≈ ~3 HD streams** before the manager GIL saturates (vs ~13
for uint8). Per-stream wire bandwidth at 30 fps is **746 MB/s**; ten streams ≈ **7.5
GB/s**.

**It's a systemic convention, not a one-off:** every image producer emits float32/255
— `videostream.py:105`, `loadfile.py:150,152`, `imagegeneration.py:192`,
`edgedetector.py:62`. So *all* image data on the wire is 4× oversized. (Audio-as-float32
is fine — that's the standard.) This is the cheapest high-impact perf fix in the report:
adopt a **uint8 image transport convention** and convert to float only in the nodes that
need it. Stacks multiplicatively with fix #1 (raw forwarding) — together, ~3 → well past
the target.

### A1. 🔴 Bridge decode→re-encode per frame (firsthand-verified)

**The single biggest scalability ceiling.** The data path forwards frames that
are *already in GOOF wire format* in SHM, but decodes and re-encodes them:

- `node_helpers.py:471-482` — the per-NodeRef `_data_pump` does
  `buf = sub.take_latest()` (raw GOOF bytes from SHM) → `data = decode_data(buf)`
  (allocates a fresh ~6 MB ndarray, memcpy out of SHM) → `cb(self, slot, data)`.
- `bridge/data.py:160-165` — the bridge callback receives the **decoded `Data`**
  and rebuilds the identical frame: `prepare_encode(data)` (re-packs msgpack meta
  *every frame*) + `encode_data_into(...)` (memcpy ~6 MB into a fresh `bytearray`)
  + `bytes(buf)` (**another** full ~6 MB copy into immutable bytes).

So per HD frame, **in the manager process under the GIL**: 1 decode copy + 1
msgpack re-pack + 1 encode copy + 1 `bytes()` copy = **≥3 full-frame memcpies +
a redundant msgpack round-trip**, for bytes it could have forwarded verbatim.

The `data.py` module docstring even **claims the opposite** — *"forwards every
encoded Data frame verbatim … no transcoding happens server-side."* That is
false as written.

At ~20 streams × 30 fps × 6 MB that is multiple GB/s of avoidable memcpy plus
20 GIL-bound decode+encode pumps contending for one interpreter lock. This alone
will prevent the stated scale.

**Measured (real codec, 1080p RGB, this machine):** the per-frame
`decode_data → prepare_encode → encode_data_into → bytes(buf)` path costs
**2.57 ms/frame → ceiling ~389 fps single-thread under the GIL ≈ ~13 HD streams
@30fps** — and that excludes the WS send, the SHM `take` copy, and all other node
work, so the real ceiling is lower (≈5–8). decode-only is 0.29 ms; the redundant
re-encode + `bytes()` (including a 6 MB `bytearray` zero-fill every frame) is ~2.3 ms
of the 2.57. The proposed raw-forward path is ~0 ms. This is direct confirmation, not
estimate.

**Fix:** add a raw-bytes handler path so the bridge forwards `buf` (the SHM GOOF
bytes) directly — eliminates decode, re-encode, and meta re-pack entirely for the
forwarding case. The pump already holds `buf`; pass it through (e.g.
`set_data_handler(slot, cb, raw=True)` → `cb(self, slot, buf)`), or a dedicated
`set_raw_data_handler`. Decoding is only needed by consumers that want a `Data`
(expression engine uses its own subscribers, not this handler), so the bridge
never needs it.

### A2. 🟡 `bytes(buf)` extra copy in the bridge hot path (firsthand-verified)

`bridge/data.py:165` — `_mux.dispatch(bytes(buf))` copies the freshly-encoded
`bytearray` into immutable `bytes`. aiohttp's `send_bytes` accepts
`bytes | bytearray | memoryview`, and `buf` is never mutated after encode, so the
`bytes()` is a pure extra full-frame copy. (Subsumed by A1's fix, but trivially
removable on its own.)

### A3. 🟠 HD ImageViewer paints on the main thread (firsthand-verified)

`viewers/ImageViewer.svelte:50-127` — `paint()` runs a **~2.07 M-iteration JS
loop** per HD frame (1920×1080) to expand the array to RGBA, then
`ctx.putImageData` (≈8 MB) — all on the **main thread**. No OffscreenCanvas, no
worker pool, no WebGL texture upload (CLAUDE.md §7 explicitly recommended these).
rAF coalescing caps each viewer to display rate, but 20 viewers still demand
hundreds of full-frame RGBA expansions + putImageData per second on one thread;
60 fps is not holdable. ImageData reuse across frames (good) does not address the
per-pixel JS loop or the upload.

**Fix:** WebGL path (upload the RGB typed array straight as a texture, let the GPU
do RGBA + scaling — no JS per-pixel loop), or `createImageBitmap` + OffscreenCanvas
in a worker pool. Either removes HD pixel work from the main thread.

### A. VERDICT on the target scale

**NO — not as currently built.** Realistic ceiling today is **~5–8 HD streams**;
"tens" is ~6–10× beyond it. But the failures are **implementation waste, not a
fundamental architecture limit** (process-per-node + SHM + raw-byte WS forwarding +
browser-side decode is sound). The same 6 MB HD frame currently travels through
**7+ full-frame copies and 2 full decode/encode round-trips**, bottlenecked on the
two un-shardable serialization points: the **manager GIL** and the **browser main
thread**.

Per-frame accounting (one 1080p RGB frame, 6.22 MB), copies marked **waste** are
removable:

| # | copy / transcode | where | waste? |
|---|---|---|---|
| 1 | heap→SHM | producer `encode_data_into` | needed |
| 2 | SHM→`bytes` | `transport.py:_bytes` | reducible→memoryview |
| 3 | `bytes`→ndarray | `decode_data` (manager) | **waste** |
| 4 | meta msgpack **un**pack | `decode_data` | **waste** |
| 5 | meta msgpack **re**pack | `prepare_encode` | **waste** |
| 6 | ndarray→bytearray | `encode_data_into` (bridge) | **waste** |
| 7 | bytearray→`bytes` | `bytes(buf)` | **waste** |
| 8 | →WS transport buf | aiohttp `send_bytes` | reducible→0-copy |
| 9 | WS→JS ArrayBuffer | browser net | needed |
| 10 | `buffer.slice()` | `decode.ts:170` (main thread) | **waste** |
| 11 | per-pixel JS→RGBA | `ImageViewer.paint` (main thread) | **waste→GPU** |
| 12 | `putImageData` | `ImageViewer` (main thread) | reducible→WebGL |

Copies 3–7 are a **closed loop**: the producer wrote the frame with
`encode_data_into`; the manager decodes it and re-encodes it with the *same*
encoder, producing byte-identical output. At 20×30 fps that's **~11 GB/s memcpy +
~3.7 GB/s allocation, all under one GIL** — saturates a core before a frame reaches
the network. With N pump threads contending on the GIL for the msgpack/slice work,
they **do not parallelize**: ~600 fps × ~3 ms GIL-held ≈ **1.8 s of GIL work per
wall-second — physically impossible**; latest-wins masks it as silent drops → video
stutters to single digits.

### A — full bottleneck list (corroborated by perf pass)

**Backend / manager (GIL-bound):**
- 🔴 **A1** decode→re-encode on the forward path — `node_helpers.py:471-482` +
  `data.py:160-165`. *The single biggest limiter.*
- 🔴 **A4** double full-frame copy in `transport.py` take path (SHM→bytes→bytes) —
  `transport.py:229-247` + `codec.py:248`. Fix: `take_latest_view()` returning a
  held Sample + memoryview.
- 🔴 **A5** N decode-pump threads contend on one GIL — `node_helpers.py:442-448`
  (one thread per NodeRef). Fix falls out of A1 (pump work → ~0); consider one
  shared WaitSet across NodeRefs.
- 🟠 **A6** per-frame msgpack re-pack of large meta (PSD freq axis >64 KB, no
  caching) — `codec.py:_pack_meta` via `data.py:162`. Disappears under A1.
- 🟠 **A7** single asyncio loop on one daemon thread serializes ALL WS sends —
  `server.py:179`, `data.py:76`. Next ceiling after A1. Fix: shard across a loop
  pool (≈1/core), pin each mux to a loop.
- 🟡 **A8** one WS message per Data frame — no batching of high-rate small streams
  (4× 1 kHz EEG = thousands of tiny msgs/s) — `data.py:64-76`. Fix: per-loop flush
  tick (~60 Hz) concatenating latest-per-slot into one length-prefixed message.
- 🟡 **A9** WaitSet wake does O(listeners) linear guard scan — `transport.py:539-552`.
  Fix: index `_ipc_guards` by attachment id.
- 🟡 **A10** iceoryx2 64 KiB→8 MiB PowerOfTwo grow-storm at startup for every video
  node — `transport.py:198-210`. Fix: per-slot `initial_max_slice_len` hint.

**Browser receive / decode (main-thread-bound):**
- 🔴 **A11** eager decode on every WS message *before* coalescing — `data.ts:38`,
  `frames.ts:62-80`. kHz EEG/audio: ~94% of decodes wasted. Fix: move latest-wins
  coalescing **upstream of decode** (stash raw ArrayBuffer per slot, decode only
  the survivor at flush).
- 🔴 **A12** `readTypedArray` `.slice()` copies the whole body into a fresh
  ArrayBuffer every frame — `decode.ts:170`. ~3.6 GB/s redundant main-thread memcpy
  + GC stalls. Guards a buffer-reuse hazard **browsers don't have** (each WS message
  owns its buffer). Fix: zero-copy `new Float32Array(buf, offset, count)` views
  (handle element alignment; trivial once only flushed survivors are decoded).
- 🔴 **A13** ALL decode runs on the main thread — no Worker/OffscreenCanvas/
  transferable anywhere in `src/lib` (confirmed). Fix: data WS + decode in a Web
  Worker, `postMessage` decoded TypedArrays via transferable list.

**Viewer render (main-thread-bound):**
- 🔴 **A3** (above) HD ImageViewer pixel loop + putImageData on main thread —
  `ImageViewer.svelte:50-121`. 20 viewers ≈ 160–460 ms/frame = 10–29× over budget.
- 🟠 **A14** ImageViewer renders at full **data** resolution (1920×1080) into a
  ~200×150 CSS box — ~70× oversampling — `ImageViewer.svelte:67-72`. Fix:
  downsample to the viewer's CSS box (ResizeObserver-cached) before paint.
- 🟠 **A15** ArrayViewer feeds full-resolution data to `uPlot.setData` every frame,
  no min/max-per-pixel decimation — `ArrayViewer.svelte:303-352`. ~5–30× redundant
  stroke work for 1 kHz buffers. Fix: min/max-per-column downsample to canvas width.
- 🟠 **A16** no shared paint scheduler — 20 viewers fire 20 independent rAFs that
  pile into one frame — `frames.ts:62-80`. Fix: one global rAF scheduler with a
  per-frame paint budget (~8–10 ms), paint visible+large first, defer the rest
  (latest-wins makes deferral free).
- 🟡 **A17** Topomap `data.fill(0)` of full rect + O(nTotal×pixels) sweep every
  frame on main thread — `TopomapViewer.svelte:142-184`. Fix: throttle to ~10–15 Hz;
  only clear/​write inside-circle pixels.
- 🟡 **A18** ImageViewer uses `Number(src[i])` + `scale()` indirection in the hot
  pixel loop even for plain u8 RGB — `ImageViewer.svelte:92-119`. Fix: branch once
  on dtype, tight u8 fast path.

### Minimal change set to unlock the target (dependency order)

1. **Forward raw bytes through the manager** (A1/A4/A5/A6) — `set_data_handler(…, raw=True)`
   passing `buf` straight to `on_frame` → `_mux.dispatch(buf)`, zero transcoding.
   Converts the limiter from impossible to trivial; ~90% of the win.
2. **Move browser data-plane + HD paint off the main thread** (A11/A12/A13/A3/A14) —
   Web Worker owns WS + coalesce-before-decode + OffscreenCanvas/WebGL paint;
   transferable hand-off; downsample to CSS box.
3. **One global rAF scheduler with a paint budget** (A16).
4. **Shard the bridge data plane across loops + min/max-decimate line plots**
   (A7/A8/A15) — only effective after #1 frees the GIL.

| segment | now (per HD frame) | after | reduction |
|---|---|---|---|
| manager full-frame copies | 6 | 1 (0 w/ memoryview) | ~83–100% |
| manager msgpack passes | 2 | 0 | 100% |
| manager GIL @20×30fps | ~1.8 s/wall-s (impossible) | ~0 | unblocks |
| browser main-thread copies | 2 | 0 | 100% |
| main-thread paint / HD viewer | ~10–20 ms | <0.5 ms | ~20–40× |
| realistic HD-stream ceiling | **~5** | **~20–30+** | target met |

---

## B. Bugs (verified)

> Populated from the adversarial bug-hunt pass (find → refute). Only `confirmed`
> findings are listed; false positives dropped.

### Firsthand-verified (read directly)

- 🟡 **B-fh1 — int64/uint64 arrays break viewers that do raw arithmetic.**
  `codec/decode.ts:183-196` returns `BigInt64Array`/`BigUint64Array` for `i8`/`u8`
  numpy dtypes, cast to `ArrayLike<number>`. `ImageViewer` is safe (it wraps every
  read in `Number(...)`), but any viewer/summary doing `v[i] - lo`, `v[i] * s`, etc.
  directly on the values will throw *"Cannot mix BigInt and other types"* at
  runtime for int64 data (e.g. a node emitting int64 timestamps). Fix: convert
  i8/u8 to `Float64Array` at decode (lossy beyond 2^53 but correct for plotting),
  or guarantee every consumer coerces with `Number()`.

- 🟡 **B-fh2 — `data.py` docstring is actively false.** `bridge/data.py:1-14` says
  it "forwards every encoded Data frame verbatim … no transcoding happens
  server-side." The code decodes (pump) then re-encodes (bridge). Misleading to
  anyone reasoning about the hot path. (See A1.) Fix the comment when fixing A1.

- ⚪ **B-fh3 — `decodeData(offset, length)` params are dead.** `codec/decode.ts:67`
  accepts `offset`/`length`, only ever called with defaults (one frame per WS
  message). Harmless, but would become load-bearing if WS batching (A8) lands —
  keep or document intent.

### From the adversarial bug-hunt pass (find → refute; only confirmed shown)

71 findings raised, 29 refuted as false positives, 18 confirmed + 1 uncertain by
the verifier, 23 left unverified by rate-limiting (re-verification in §E; two
firsthand-confirmed below).

**🟠 High**
- **B1 — Load silently breaks live state + error forwarding for the whole patch.**
  `bridge/control.py:431-474,484-496`. `_replace_graph` removes old nodes with
  `notify_gui=False`, so `on_node_removed` never fires and `ControlHub._wired_nodes`
  is never pruned. `manager.load()` re-adds nodes under the **same reused display
  names**; `on_node_added → _wire_node_status` then early-returns (`if name in
  _wired_nodes: return`), so the new NodeRefs **never get STATE_UPDATE/
  PROCESSING_ERROR handlers**. After any Load/load_text, param reflection and error
  events stop reaching the browser for every reused-name node — and a page reload
  doesn't fix it (only a manager restart). Fix: `_wired_nodes.clear()` in the
  graph_replaced flow (or route removals through `on_node_removed`).
- **B13 — undo/redo not re-entrancy-guarded (firsthand-confirmed; finder rated
  critical).** `stores/history.svelte.ts:315-347`. `undo()` reads the top action,
  `await`s `restoreNavContext` and the inverse replay, and only `pop()`s after.
  Holding **Ctrl+Z** (OS key-repeat) fires overlapping `undo()` calls that read the
  *same* top action, replay its inverse twice (duplicate RPCs), then double-`pop()`
  — desyncing both stacks from graph state. Fix: an in-flight/`busy` guard at the
  top of `undo()`/`redo()`.

**🟡 Medium**
- **B2 — viewer disconnect stalls the entire data-plane loop.** `bridge/data.py:180-187`.
  The teardown holds `async with self._lock` while calling the **blocking**
  `ref.set_data_handler(slot, None)` (WaitSet detach + iceoryx2 subscriber close +
  UNREGISTER_SUBSCRIBER IPC send) directly on the asyncio loop thread. Every
  off-screen unsubscribe (IntersectionObserver, frequent) freezes all other viewers'
  sends + control traffic for the duration. Fix: run the blocking call in an executor
  outside the lock.
- **B3 — data-pump WaitSet attach/detach races the pump's `wait()`.**
  `node_helpers.py:439,465` + `transport.py:494-552`. `set_data_handler` mutates
  `_data_waitset` under `_data_handlers_lock`, but the pump calls `wait()` **without**
  that lock and the WaitSet has no internal lock — concurrent mutate+iterate of
  `_ipc_guards`/`_ipc_listeners`/`_ipc_ws` can raise *"dict changed size during
  iteration"* / AttributeError. Triggered by subscribing/unsubscribing while frames
  flow (the hot path). Fix: lock the WaitSet internals, or snapshot membership under
  the handler lock.
- **B7 — ExpressionEngine masks decode errors + feeds stale data.**
  `expression.py:187,271-274`. A fetch-time `decode_data` failure records
  `last_error` but returns the previous (stale) Data; then `evaluate()` unconditionally
  clears `last_error=None` on success — erasing the decode error and silently computing
  on stale data. Fix: don't clear `last_error` if a fetch-time error occurred this eval.
- **B14 — StringViewer XSS via `marked.parse` + `{@html}` with no sanitization.**
  `viewers/StringViewer.svelte:14,18`. Markdown mode injects unsanitized HTML; goofi
  nodes routinely emit attacker-influenced text (transcripts, web/API text, LLM
  output). `<img onerror>`/`<script>` in node output runs JS in the app origin. Fix:
  DOMPurify, or disable raw HTML in marked.
- **B-fh4 — user edits during an in-flight undo/redo are dropped from history
  (firsthand-confirmed).** `history.svelte.ts:285-287`. `record()` early-returns while
  `suspendDepth>0`; `suspend` stays up for the whole async replay, so a node drag/param
  edit during that window is silently not recorded. Narrow race; real.

**🟢 Low** (selected — full list in the report data)
- **B4 — iceoryx2 listener leak per subscribe/unsubscribe cycle.** `node_helpers.py:426-431`
  detaches+closes the subscriber but never `prev[1].close()` on the IpcListener
  (expression.py does close both — local inconsistency).
- **B16 — `dropPanelOnTabBar` off-by-one** when the dragged panel's source tab sits
  left of the drop index. `workspace.svelte.ts:403-422`.
- **B17 — slider `editing` flag has no `pointercancel` coverage** — can stick `true`
  and permanently freeze a param field against backend echoes. `ParamField.svelte:244-245`.
- **B-uncertain — `ThreadSubscriber.close()` drops the shared channel with no refcount**
  (transport.py:435-437) — split-brain for in-process multi-consumer fan-out;
  demonstrated by the finder, verifier marked uncertain. (transport.py is nominally
  out-of-scope, but real.)
- `_msgpack_default` hard-fails on a raw `ndarray` in meta (latent footgun;
  codec.py:87-92) · `decodeTable` ignores `consumed` vs `value_len` (silent desync;
  decode.ts:141-145) · codec.py docstring dtype-tag off-by-one (B-fh2) · ArrayViewer
  full plot rebuild on any settings change (ArrayViewer.svelte:358-376) · dead
  `modalEl` in ExpressionModal.

### Firsthand-verified backend audit (refuted my own hypotheses)
- `_rename` (manager.py:520-545) is **sound** — boundary interface entries store the
  *local* name (`:594`), resolved on demand via `_member_display` (`:1005`), so they
  don't go stale across group/expand renames.
- strict-mirror `update_param`/`set_node_pos` (`:1094-1165`) are **well-defended**;
  deepcopy is applied consistently and `set_node_pos` assigns fresh dicts to dodge the
  `_node_record` aliasing it documents. (One niche: `update_param` writes a bare scalar
  into a shared member's *saved* def record even if the param had an expression-bound
  `{value,expression}` shape — could drop the binding in the saved file. Low.)

---

## C. Dead code / inelegant abstractions / duplication

> Frontend editor+viewers covered; backend + frontend-stores/workspace finders
> were rate-limited (re-run in §E).

- 🟡 **C1 — magic `233` (and 36/24/144) triplicated** across `app.css:47-50`,
  `nodeMetrics.ts:13-19`, `snap.ts:22-23`, each with a "must match" comment. Drift
  hazard. Fix: one source of truth, push to CSS vars at startup; `snap` imports `NODE`.
- 🟡 **C2 — undo-recording binding wrapper duplicated 3×**
  (`SlotViewer.svelte:23-46`, `ViewerPanel.svelte:43-72`, `agent/commands.ts:112-118`)
  — the `ViewBinding` abstraction stops one layer short. Fix: a `recordingBinding`
  decorator in `viewBinding.ts`.
- 🟡 **C6 — output-port Y positions computed in JS** (`GoofiNode.svelte:57-64`)
  mirror the CSS slot-stack layout (inputs use pure CSS) — same drift hazard as C1.
- 🟡 **C7 — `MENU_W=212` duplicated** JS-const vs CSS in `ViewerSettingsMenu.svelte:21,128`.
- ⚪ **C3** dead `hasSettings` export (`settingsSchema.ts:113-115`) · **C4**
  `defaultSettings` over-exported (`:101-105`) · **C5** stale docstring on
  `setInlineFullView` (`inlineView.svelte.ts:43-52`) · **C8** duplicated `requestSlotClick`
  seed object · **C9** unreachable "No settings" branch · dead `decodeData(offset,length)`
  params · dead `modalEl` in ExpressionModal.

---

## D. Unfinished & missing features (UX/workflow completeness)

> §6-vs-code audit covered; started-never-finished / missing-UX / test-coverage
> finders were rate-limited (re-run in §E). The §6 audit confirms the **large
> majority of CLAUDE.md §6 is fully implemented.**

**Unfinished / partial (dead-end or confusing UX):**
- 🟠 **D1 — TableViewer is a flat one-level grid, not the §6.4 recursive tree.**
  `TableViewer.svelte:15-33` collapses nested TABLE→`{N fields}`, arrays→`array[h×w]`.
  A data type the tool emits is effectively un-viewable. Effort: medium.
- 🟠 **D2 — Examples menu backend exists but is unwired.** `control.py:343`,
  `fsbrowse.py:74`, `graph.svelte.ts:658` `listExamples()` all present, but **no
  component calls it** — users hand-navigate the full-FS browser to `examples/`. Small.
- 🟡 **D3 — Error display is a truncated floating chip, not a dockable panel.**
  `ErrorPanel.svelte` clamps to 3 lines; no `errors` panel type registered; full
  tracebacks only in Console. §6.9 partially met. Medium.
- 🟠 **D4 — ImageViewer HD path unfinished vs §7** (no OffscreenCanvas/WebGL/worker/
  downscale) — see A3/A13/A14. Large.
- ⚪ **D5 — Minimap removed** (commit 2041637) — §6.1 optional; deliberate. No overview
  for large patches.

**Confirmed done (per §6 audit):** render/zoom/pan/drag/marquee/multi-drag, dtype-aware
add-menu w/ doc-on-hover + insert-at-cursor, category colors, error border, all four
param types incl. trigger buttons + live echo, ArrayViewer (uPlot, logX/logY, autoscale),
Trajectory/Topomap/String/HighDimFallback, IntersectionObserver gating, Save/SaveAs/Load
+ unsaved dot, Ctrl+C/V versioned clipboard, full keyboard set, metadata inspector.

> Missing-UX inventory (recording/export of streams, connection/fps HUD for the
> high-throughput goal, edge reconnect, node alignment, pause/snapshot a viewer,
> crashed-node restart, onboarding/empty-state) and the started-never-finished sweep
> (`__inlineKind` probe, sub-patch-instance viewer state not persisted, undo
> highlight-pulse partial) are re-running in §E.

## E. Re-runs of rate-limited finders + open verification

First-pass rate limiting (from 4 concurrent heavy workflows) lost: 23 bug verifiers,
3 dead-code finders, 3 feature finders.

**Bug re-verification done: 13 confirmed, 10 refuted.** New confirmed bugs beyond
§B (B1/B13/B-fh4 already covered):

**🟡 Medium**
- **B-add_member — `add_member_node` mutates the shared definition before the
  sibling-spawn loop with no rollback.** `manager.py:707-732`. A sibling spawn that
  raises mid-loop (`SubPatchTooDeep` from a longer-prefixed sibling name that wasn't
  pre-checked, or any spawn failure) leaves the definition + a subset of siblings
  desynced **permanently** (corrupt save output, broken sibling consistency). The
  sibling-name budget IS pre-checked in `make_shared`/`group_into_subpatch`, but not
  here. Fix: pre-check sibling names + wrap the loop in rollback.
- **B-leak2 — `add_node` leaks the spawned subprocess on a name collision.**
  `manager.py:355-363`. `_spawn_node` (real OS subprocess) runs before
  `self.nodes.add_node(name, ref, force_name=True)`, which raises `KeyError` on a
  duplicate explicit name. The `KeyError` propagates with the process already spawned,
  never registered, never `terminate()`d, and NodeRef has no finalizer → orphan
  process + iceoryx2 services. Reachable via the bridge `add_node` op (forwards the
  client name with no uniqueness pre-check). Fix: try/except around the insert,
  terminate the ref on failure.
- **B8 — marquee/box selection wiped on flowNodes rebuild** (firsthand + verified).
  `NodeEditorPanel.svelte:227-300`. Rebuild sets `selected` from `sel.nodes(panelId)`
  only; marquee selection (Svelte Flow `n.selected`, never synced to the store) is
  lost on any rebuild trigger (node add/remove/move, `node_moved` echo, sub-patch
  enter/exit). Fix: sync Flow selection into the store, or preserve `selected` across
  rebuild.
- **B9 — paste anchors at screen pixels treated as flow coords** (firsthand +
  verified). `NodeEditorPanel.svelte:743-758`. `[innerWidth/4, innerHeight/4]` ignores
  pan/zoom; pasted nodes land at a fixed graph position, often off-screen. Fix:
  `screenToFlowPosition` at the cursor/viewport center.
- **B20 — Ctrl+S / Ctrl+O bypass the modal/input guard** (firsthand + verified).
  `AppShell.svelte:128-139`. undo/redo correctly gate on `ui().modalOpen` + targetTag;
  save/load don't — they fire while typing in the ExpressionModal or a param field.
  Fix: apply the same guard.

**🟢 Low** (confirmed)
- **B1-redundant** — destructive load re-broadcasts N `node_added` + M `link_added`
  on top of the authoritative `graph_replaced` (idempotent on the client; pure WS/
  churn waste). `control.py:263-288`.
- **B6 — `log_endpoint` (node-private runtime field) round-trips into the saved
  `.gfi`.** `manager.py:1167-1176` (`_node_record` pops `output_subscribers` but not
  `log_endpoint`). Stale ephemeral URL persisted; should be stripped.
- **B10 — `SubpatchZoomExit` exit `$effect` re-runs on node add/move, not only zoom**
  (`SubpatchZoomExit.svelte:39-53`) — reads `store.nodes`/bounds, so a node move while
  zoomed-out near threshold can pop you out of a sub-patch without zooming.
- **B15 — multi-node drag records N separate `set_node_pos` undo entries**, not one
  atomic step (`NodeEditorPanel.svelte:593-601`, no transaction wrapper) — undoing a
  group move takes N Ctrl+Z.
- **B17b — `restoreNavContext` writes selection/active-panel for panels that no longer
  exist** (`navContext.ts:57-64`) — the `enteredPath` loop guards existence, the
  selection/active writes don't → stale selection-map entries.
- **B21 — Splitter px→fraction uses full container width incl. the fixed 6px gutters**
  as denominator (`Splitter.svelte:24-36`) — slight drift between drag delta and
  resulting fraction, accumulating across N-child splits.

**Refuted (good news — not bugs):** #19 debounced-layout clobber (hydrate's own effect
cancels the stale timer), #11 drag-persist-on-click (Svelte Flow doesn't fire dragStop
on a click), #16 compound redo name-propagation (backend reuses the display name), #18
load_patch redo layout, #4/#5 gui_kwargs aliasing (no harmful mutation), #7
mark_unsaved on raise, #12 AddNodeMenu key index, #22 literalFor non-finite, #23
modalOpen boolean.

### Test baseline

`pytest tests/` → **925 passed, 6 skipped** (70 s). Suite has grown far past the
"128" in the brief; baseline is green.

### Backend dead code / abstractions (re-run)

- 🟢 **`_splice_endpoint(dir)` arg is dead** — never read; 4 call sites pass
  discarded "out"/"in" (`control.py:136`).
- 🟢 **`flat_view` is superseded scaffolding** kept alive only by its own tests —
  the recursive expander (`read_graph` + `_expand_doc`) landed; `flat_view` has zero
  prod callers (`patch_format.py:68`). Delete it + 2 tests; docstring still advertises
  it as THE projector.
- 🟢 **Unused `**gui_kwargs`** on `remove_node`/`add_link`/`remove_link` (would
  swallow a typo'd kwarg) — `manager.py:385,439,496`.
- 🟢 **Unused imports** `Node` (manager.py:35), `Optional` (control.py:23).
- 🟢 Backend runtime dead code: `NodeRef.serialization_pending` (node_helpers:263),
  `NodeProcessRegistry.all()` (:622), `Node.data_path` (node.py:1079), `Param.copy()`
  (params.py:50), unreachable SHUTDOWN elif (node_helpers:525).
- 🟡 **`{value, expression, …}` param-dict parsing replicated 3×** in `params.py`
  (258-303, 365-398) · strict-mirror sibling skeleton duplicated in
  `update_param`/`set_node_pos` (manager.py:1102,1140) · `add_extra_attributes`
  monkey-patch justified by an obsolete Python<3.10 constraint (params.py:16) ·
  namespaced member-name f-string hand-rolled at 8 sites despite `_member_display`.
- 🟡 **Two near-identical WS reconnect/backoff state machines** (`control.ts` and
  `data.ts`) — candidate for a shared helper.

### Frontend stores/workspace dead code (re-run)

- 🟠 **`history.lastError` is set but never surfaced** (`history.svelte.ts:263,324,341`)
  — the undo spec's atomic-or-nothing **"toast on failed undo" (§6.3/§10) is
  unfinished**: a rejected undo/redo silently does nothing with **no user feedback**.
  This is the most user-visible of the dead-code items — really an unfinished feature.
- 🟢 `GraphStore.isOutputConnected/isInputConnected` unused (graph.svelte.ts:775,780)
  · `SelectionStore.forgetPanel/clearNodes/clearEdges/selectEdges` unused ·
  `ControlClient.close()`/`connected` getter unused · `BaseParam`/`UnknownParam`
  exported types unused · `pathToArray`/`arrayToPath` duplicated verbatim in
  `navContext.ts` and `NodeEditorPanel.svelte` · orphaned doc comment on
  `SelectionStore.forgetAll` · stale undo-phase narrative comments.

## G. Feature gaps (re-run: started-unfinished, missing UX, test coverage)

### Started but unfinished
- 🟡 **Sub-patch instance viewer state not persisted** (firsthand-confirmed,
  `control.py:232-244` accept-and-ignore) — viewer kind/settings on a collapsed
  sub-patch's output slots are dropped on reload.
- 🟡 **Failed-undo toast unfinished** (`history.lastError` never rendered — see above).
- 🟢 **Undo highlight pulse missing** — spec §5.2 step 6 ("~600 ms CSS class on the
  changed node/param") not implemented; only selection-restore highlighting exists.
- 🟢 **`load_patch` redo layout = null** (`graph.svelte.ts:_recordLoadPatch` hardcodes
  `afterLayout: null`) — redo re-hydrates from YAML, losing the exact post-load layout.
- 🟢 **`__inlineKind` e2e probe referenced but never defined** (`test_undo_viewer.py:9`,
  null-guarded) — the viewer-kind undo e2e silently no-ops its core assertion.
- 🟡 **`make_unique` undo re-attach is best-effort** (graphExecutors.ts:255-267) —
  can't restore the exact prior shared definition (spec §11 deferral).
- 🟢 **`restoreNavContext` writes `subpatchPath` directly** instead of driving
  `enterInstance`/`exitToDepth` with the validity fallback the spec §5.3 describes.

### Missing for complete/intuitive UX (biosignal node-graph at scale)
- 🟠 **Data recording / export of a slot's stream to disk** — *no* record/export op in
  the bridge. For a biosignal tool this is a core workflow gap. Large.
- 🟠 **No data-rate / FPS / health HUD** — for a tool whose headline goal is "tens of
  HD streams," there is no surfaced throughput, per-viewer fps, or backpressure/drop
  indicator. Only a binary "connected" badge (`TopBar.svelte:46`). Medium.
- 🟠 **No crashed-node restart from the UI** — `remove_node`+`add_node` are the only
  lifecycle ops; a crashed node can't be respawned without rebuilding it. Medium.
- 🟡 **Edge editing is one-way** — Svelte Flow `onreconnect`/`edgesReconnectable` not
  wired; you delete + redraw a link rather than drag its endpoint. No hover dtype.
- 🟡 **No node align/distribute/auto-layout** (only drag-snap to neighbors exists).
- 🟡 **Viewer ergonomics**: no pause/freeze, frame snapshot, fullscreen, or per-channel
  toggle.
- 🟢 **No empty-state onboarding** (blank canvas, no "press Tab / load an example" hint).
- 🟢 **No global search** to find/jump to an existing node in a large patch (the
  typeahead only searches node *types to add*).

### Test-coverage gaps (untested ⇒ under-polished risk)
- 🟠 **No data-plane throughput/perf/stress test exists** — `grep -i
  'stress|throughput|fps|performance'` over `e2e/` returns zero. The §7 headline
  requirement has **no automated coverage**; `test_data_multiplex.py` only checks 2
  subscribers each get ≥N frames.
- 🟠 **Codec TS decoder (`decode.ts`) has no unit test** — the load-bearing wire-format
  port is unverified against Python output (CLAUDE.md §10 called for a round-trip test).
- 🟡 **Viewer rendering never pixel-asserted** — `test_viewers.py` screenshots but only
  asserts dropdown values + no-console-errors; CLAUDE.md §9's pixelmatch baselines were
  not built. Topomap/Trajectory have no real-data e2e at all.
- 🟡 Copy/paste (`clipboard.ts`), param value round-trip, expression round-trip
  (frontend), metadata-inspector content, and HighDimFallback are untested.

## H. Second-wave findings (lightly-covered areas + completeness critic)

23 confirmed, 13 refuted, 1 uncertain. Net-new beyond §A–G:

**🟠 High**
- **H1 — `node.py self._waitset` mutated lock-free while the processing thread
  `wait()`s on it.** `node.py:660,667,395-396,727`. The messaging thread (SUBSCRIBE/
  UNSUBSCRIBE_INPUT on link wire/unwire) **and** the expression engine attach/detach
  listeners on the same WaitSet the processing thread blocks in — no lock. Can raise
  *"list/dict changed size during iteration"* **crashing the node's processing loop**
  (node goes silent but stays "alive"), or drop a just-attached listener. This is the
  **more exposed twin of B3** — node.py's `_waitset` has *zero* synchronization (the
  data-pump one at least snapshots). Triggered by ordinary wiring + expression edits
  while a node runs. Fix: lock all `_waitset` attach/detach/wait, or marshal mutations
  onto the processing thread.
- **H2 — the same `ExpressionEngine.evaluate()` runs concurrently from the messaging
  and processing threads**, corrupting its subscription bookkeeping. `node.py:749,789,
  407,638`. `_apply_expression` is called from both threads with no per-engine lock.
  Fix: serialize expression eval/mutation under a per-node lock.
- **H3 — `sliding_window` channelwise `n_windows` off-by-one silently corrupts
  results.** `convenience.py:80`. A **data-correctness** bug in a signal-processing
  helper — windowed outputs are subtly wrong, no error. Fix: derive `n_windows` from
  the same iterator that generates the windows.

**🟡 Medium**
- **H4 — `node.py:747` iterates `self._expressions` un-snapshotted** (the other two
  sites use `list(...)`); a SET_EXPRESSION mid-iteration raises `RuntimeError` that
  **permanently kills the processing thread** — the node stays "alive" but never
  processes again. One-line fix: `list(self._expressions.items())`.
- **H5 — `autotrigger` is silently disabled whenever a triggering input is wired**
  (`node.py:723,862-863`) — behavior regression vs the old runtime; a node meant to
  free-run stops free-running once you wire any input.
- **H6 — `ViewerFeed` IntersectionObserver keeps tiny zoomed-out inline viewers fully
  subscribed** (`ViewerFeed.svelte:29`) — no minimum-rendered-size gate, so at scale
  every inline viewer streams even when zoomed out to dots (compounds the perf budget).
  Fix: gate on `intersectionRect` area / min on-screen size.

**🟢 Low — reconnect/resilience cluster** (real for a long-running tool)
- **H7 — in-flight RPCs reject on a control-WS drop and nothing retries them; mutations
  are silently lost** on a transient reconnect (`control.ts:305-314`). No user feedback.
- **H8 — active data subscriptions don't re-open after a node remove/re-add during a
  reconnect → viewers go permanently blank** (`data.ts:51-64,94-110`); and a data WS
  **reconnect-loops forever against a shut-down backend** (`manager_shutdown` never
  surfaced to the data plane).
- **H9 — same-session control reconnect runs full `_replaceSnapshot`, wiping all
  per-node viewer UI state** (collapse/kind/settings) mid-session
  (`graph.svelte.ts:70-102`); related: `subpatch_changed` leaks `ui().expanded`
  entries (`:196-199`), and a reconnect snapshot that differs from local state isn't
  reconciled (missed-while-disconnected nodes aren't cleaned up).

**🟢 Low — leaks / robustness**
- **H10 — `transport` data service caps subscribers at 16** (`DEFAULT_MAX_SUBSCRIBERS`,
  `transport.py:36,141`) — a heavily fanned-out slot + multiple browser viewers fails
  `open_or_create` on the 17th consumer. Raise/configure for the data plane.
- **H11 — `Manager.terminate()` kills group-host processes before terminating member
  NodeRefs** (`manager.py:1456-1461`), so members' `_teardown_endpoints` never runs →
  iceoryx2 publisher leak until dead-node reaping.
- **H12 — `Splitter` and `Panel` corner-drag leak window pointer listeners on mid-drag
  unmount** (`Splitter.svelte:43`, `Panel.svelte:185`) — no `onDestroy` teardown.
- **H13 — `logStream` EventSource has no `onerror`** (`logStream.svelte.ts:95`) →
  dead-endpoint reconnect storm, no failure surface; `node_log` SSE handler blocks up
  to `KEEPALIVE_S` after client disconnect, and silently drops records under a slow
  consumer without a gap marker.
- **H14 — `parseClipboard` accepts version-tagged payloads without validating per-node
  shape** (`clipboard.ts:54`) — malformed paste can inject bad node specs.

*(Refuted in Wave 2: 13 — e.g. several speculative reconnect/ordering claims that the
code already handles.)*

## I. Third-wave findings (z-order, numerical correctness, components, node triggers)

16 confirmed, 3 refuted, 1 uncertain.

**🟠 High / 🟡 Medium — correctness**
- **I1 — `sliding_window` is a *cluster* of correctness bugs** (`convenience.py:74-99`),
  not just H3: (a) `n_windows` (line 80) ≠ the actual loop range → channelwise
  `reshape` throws or **silently mis-groups windows×channels**; (b) `times` and
  `results` end up **different lengths** → window onsets misaligned with values; (c)
  the exclusive upper bound **drops the final valid window**; (d) channelwise branch
  **crashes on empty/ragged results**. Used by many signal nodes → results are subtly
  or overtly wrong. Fix: enumerate windows once with inclusive bound
  `n - window_size + 1`, derive `times`/`n_windows`/grouping from that same range.
  **Empirically reproduced** (index math): N=100,W=10,S=10 → 9 windows computed but
  `n_windows=10`, so `reshape(10,-1)` of 9×nch results **raises `ValueError`**; the
  final valid window (start=N−W) is **always dropped**; `times` (9) ≠ claimed 10.
- **I2 — non-triggering input slots (`trigger_process=False`) are never drained**
  (`node.py:646-660,738-766`) — their listener isn't attached/read, so `slot.data`
  stays `None` forever. Any node that consumes a "data-only, don't-trigger" input gets
  `None` for it. Fix: drain ALL subscribed inputs; gate *triggering* (not draining) on
  `trigger_process`.
- **I3 — `to_data` mutates the caller-supplied `meta` dict in place** (`data.py:148-149`)
  — injects `shape`/`channels` into and aliases the caller's dict, so a node reusing a
  meta dict across frames gets cross-frame contamination. Fix: copy meta before
  mutating.
- **I4 — `PlacementPreview` commit listeners are global** (`PlacementPreview.svelte:69-83`)
  — with multiple node-editor panels open, a pending placement **commits into the wrong
  editor**. Fix: scope the commit to the originating editor's root element.
- **I5 — panel/tab drag never starts in Firefox** (`PanelHeader.svelte:62-69`,
  `WorkspaceTabs.svelte:92`) — `dragstart` omits `dataTransfer.setData`, which Firefox
  requires to initiate a drag. The whole drag-to-rearrange-panels workflow is broken in
  FF. Fix: `e.dataTransfer.setData(token, id)` + `effectAllowed='move'`.
- **I6 — FsBrowser doesn't set `ui().modalOpen`** (`FsBrowser.svelte:86-92`) — global
  undo/redo (and the keybinding guard) **fire on the graph behind the open Save/Load
  dialog** once focus lands on a file `<button>` (not covered by the INPUT/TEXTAREA
  typing guard). A visually-modal dialog that isn't behaviorally modal. Fix: set
  `modalOpen` on mount (mirror ExpressionModal); same for AddNodeMenu.

**🟢 Low — z-order / UX** (the user's "pay close attention to z layering")
- **I7 — `InspectorOverlay` (z 50) occludes its own toggle button** — once open, the
  breadcrumb is the only escape; a reachable control is covered (`InspectorOverlay.svelte:99-114`).
- **I8 — Tab-to-open the add-node menu can render off-screen** — `openMenuAtCursor`
  (`NodeEditorPanel.svelte:760-763`) sets raw cursor pos with no viewport clamp (the
  other open paths clamp). 
- **I9 — `ErrorPanel` popover (`--z-chip` 60) renders *under* an open context/add menu**
  (z 100/110) — `ErrorPanel.svelte:44-72`.
- **I10 — FsBrowser uses raw `z-index:1000/1001`** outside the documented scale and is
  **not portaled** (relies on the magic number to escape stacking contexts) —
  `FsBrowser.svelte:181,196,96-174`. Works today (mounted at app root) but violates the
  single-source z-scale.
- **I11 — no-op tab reorder / rename-to-same-name records a spurious undo entry**
  (`workspace.svelte.ts:424-432,291-302`).
- **I12 — a triggering input whose only pending frame fails to decode silently drops
  the trigger** for that tick (`node.py:757-766`).

*(Refuted in Wave 3: 3.)*

## J. Fourth-wave findings (viewer math + backend nodes — nominally "done")

14 confirmed, 4 refuted, 1 uncertain. The node-implementation items are in the
nominally-done backend, listed for awareness.

**Viewer math**
- 🟡 **J1 — duplicate electrode coords blank the entire topomap.** `eegLayout.ts:45-47,
  60-61,73-74` gives T7=T3, T5=P7, T6=P8 identical coords; a cap listing both an alias
  and its classic name → identical TPS matrix rows → **singular matrix** → whole
  topomap renders "layout failed" (not just one electrode). Reproduced end-to-end by
  the verifier. Fix: dedup channels by position (average colliding values) or add a
  ridge term.
- 🟡 **J2 — NaN channel value silently paints the topomap as a flat min-color disc**
  (`TopomapViewer.svelte:173-183`) — no finiteness check; looks like valid data. Fix:
  detect non-finite field → draw a "non-finite" message.
- 🟡 **J3 — topomap anisotropically distorted on non-square viewers**
  (`TopomapViewer.svelte:194-195`) — electrodes fall outside the disc. Fix: square
  drawing region `side=min(w,h)`.
- ⚪ **J4 — ArrayViewer log-X uses 1-based index vs linear 0-based** (off-by-one in
  cursor/sample readout) · **makeLUT size≤1 div-by-zero** (latent) · **ImageViewer
  grayscale NaN → LUT index 0** (silent).

**Codec encode (edge cases; core invariant is sound — see §F)**
- ⚪ **J5 — `data_byte_size` ignores a reused `meta_bytes`** so a caller passing a stale
  `meta_bytes` to `encode_data_into` can disagree with the loan size (latent SHM
  over/underflow; not hit in current callers). `codec.py:77-79,116-138`.
- ⚪ **J6 — TABLE key > 65535 UTF-8 bytes**: `_body_size` counts it but `_write_body`
  raises mid-buffer (`codec.py:106,172-174`). Validate early.

**Backend node implementations (nominally done — for awareness)**
- 🟠 **J7 — `Transpose` drops all input metadata** (returns empty meta) —
  `nodes/array/transpose.py:37`. Channels/sfreq lost downstream.
- 🟡 **J8 — `Normalization` forces output dtype to input dtype → integer inputs get
  normalized floats truncated to int** (`normalization.py:81-89`). Silent corruption.
- 🟡 **J9 — `Buffer` crashes `None += list`** when channel names appear after a
  name-less first tick (`buffer.py:72-74`); and the name-buffer/seconds length can
  desync from the buffered axis (`:89-97`).
- ⚪ **J10 — `Reduce`/`Resample`/`Buffer`/`Transpose` mutate the input `Data`'s meta/
  array in place** (`reduce.py:49-57`, …) — sloppy given multi-consumer fan-out;
  **`Resample` also truncates fractional `sfreq`**, corrupting the resample ratio
  (`resample.py:55-57`).

---

## K. Fifth-wave: signal/array/analysis node correctness sweep

**35 confirmed, 4 refuted** across a sample of DSP/array/feature nodes. The backend is
nominally "done," but the signal-processing layer has substantial correctness debt that
**silently corrupts scientific results**. Grouped by theme:

**🔴 Systemic — in-place mutation of input `Data` meta/array.** The dominant pattern:
a node does `meta = data.meta.copy()` (shallow) then mutates the **shared** nested
`channels` dict, or reassigns `data.data` in place — corrupting the node's own cached
input on the next non-data re-trigger (self-trigger / expression autoeval) and any
fan-out consumer. **`fft.py` does it correctly with `deepcopy`; these don't:** `psd.py:119`,
`padding.py:78`, `reduce.py:49` (+ Select/Reshape), `join.py:34,45`, `powerbandeeg.py:59,64`,
`autocorrelation.py:73`, `connectivity.py:69,96`, `smooth.py:39` (+ Threshold/Delay
republish input meta by reference). **One fix pattern** (deepcopy meta, never reassign
input arrays) resolves the whole cluster.

**🟠 High — outright broken / crashes**
- **Two nodes are dead from NumPy ≥1.24 removed aliases** (env has numpy 1.26.4):
  `binarize.py:51` uses `np.int` via edgeofpy → crashes every call (also reshapes
  1D→(1,N) but returns original meta → `Data()` crash, `:44`); `avalanches.py:51` calls
  **`np.float(...)` directly** → `AttributeError` when it has avalanches to size. (A repo
  grep found only these two — the breakage is contained, not widespread.)
- **`correlation.py:53` crashes on all 2D inputs** (stale `dim1` in reused meta violates
  the Data shape contract); also reduces the **wrong axis** for non-negative axis (`:42`).
- **`staticbaseline.py:110` quantile method ignores the baseline window entirely** —
  normalizes against the current frame (the baseline is meaningless); 2D uses only the
  first accumulated frame (`:81`); NaN for single-sample (`:124`); mixes time units (`:80`).
- **`powerbandeeg.py:59` mutates shared input meta (`del dim1/dim0`) → crashes next tick.**

**🟡 Medium — wrong numbers (silent)**
- **`hilbert.py:40` — `inst_frequency` is in cycles/sample, not Hz** (missing `×sfreq`);
  off by a factor of `sfreq` for every signal.
- **`psd.py:104` 'fft' method returns magnitude `|FFT|`, not power, and isn't a density**
  — mislabeled as PSD.
- **`ifft.py:34` zero-pads the wrong axis** (compares axis-0 lengths) → breaks for 2D /
  unequal-length spectra.
- **`frequencyshift.py:40` flattens N-D input to 1D**, mixing all channels.
- **`math.py:47` crashes / silently truncates on integer-dtype inputs** (cf. the
  Normalization int-truncation J8 — integer-dtype handling is a recurring gap).
- **`welfordsztransform.py:49` normalizes per (channel, sample-position)** (contradicts
  its per-channel docstring); outliers permanently inflate the running std (`:62`).
- **`filter.py:106` bandpass has no Nyquist clamp** → `ValueError` when `f_high ≥ sfreq/2`.
- **`delay.py:35` blocks the processing thread with `time.sleep`**, stalling all of the
  node's inputs.
- **`histogram.py:75` KDE vs bins return different scales** under the same label; constant
  input crashes KDE. **`threshold.py:47` nan_reset crashes on multi-element inputs.**
  **`join.py:34` stack-with-one-None grows the array a dimension every tick.**
  **`fractality.py:112` box-counting returns NaN/unreliable dimension.**

**⚪ Low** — `math.py:73` rescale div-by-zero (equal min/max), `select.py:106` by_index
ignores order/drops dupes, `powerband.py:56` relative-power div-by-zero on flat PSD,
`connectivity.py:230` non-standard imag-coherence denominator, `operation.py:56` crash
when meta lacks `channels`, `filter.py:91` buffer extended every tick.

> This is a *sample* (≈25 of ~145 nodes). The systemic in-place-mutation and
> sfreq/dtype/2D-shape patterns likely recur in unaudited nodes — a focused
> deepcopy-meta + dtype-safety + 2D-shape sweep across `nodes/` is warranted.

## L. Sixth-wave: user-facing I/O node correctness

**30 confirmed, 11 refuted.** The hardware/protocol I/O nodes share three systemic
defects (same classes seen in the runtime):

**🔴 Systemic — no/incomplete `terminate()` → resource leaks** (orphaned across shutdown
*and* live re-config; some block restart):
- **`sharedmemout.py` leaks its `/dev/shm` segment** (no terminate) → orphaned entry can
  block the next run. **`zeromqin.py:22`/`zeromqout.py:25` leak sockets + never terminate
  the zmq context. `oscout.py:50` leaks the broadcast socket. `midiout.py`/`midiccout.py`
  never close MIDI ports** (and `midiccout.py:81` references an undefined `self.goofi_port`).
  **`lslclient.py:173` leaks a discover thread; `eegrecording.py:85` streams regardless of
  consumers.** A `terminate()`-hygiene sweep across `nodes/outputs/` + stream inputs is
  warranted.

**🔴 Systemic — blocking the processing thread** (node becomes unresponsive to terminate/
param updates and stalls all its inputs): **`zeromqin.py:36` `recv_pyobj()` blocks
forever**, **`midiout.py` `time.sleep`s the full note duration**, `serialstream.py:59`
blocking read, plus `delay.py:35` (§K). These should use non-blocking/timeout reads or a
worker thread.

**🟠 Systemic — cross-thread races without locks** (callback/server thread vs processing
thread): **`audiostream.py:45`** (callback appends to `self.buffer` unlocked → lost audio
samples + uncapped O(n²) growth) and **`oscin.py:155`** (OSC server thread shares
`self.messages` unlocked → lost/dup messages; also restarts the whole UDP server on every
param echo, dropping in-flight messages, `:193`). Same class as the node `_waitset` races
(H1).

**🟠 High — data integrity**
- **`audioout.py:74` inserts a transition ramp on *every* block** → continuous click/glitch
  + sample-rate drift (duplicates `transition_samples` each call).
- **`writecsvsafe.py:62` mutates the shared annotation `Data` in place via `pop()`**,
  corrupting it for forwarded/repeated use; its RangeIndex collides across appended chunks
  (`:155`); non-daemon write thread joined with 5 s timeout silently drops queued rows (`:112`).
- **`writecsv.py:105` `default_mode` pads short columns with the last value** — silently
  fabricates/duplicates samples in saved data.

**🟡 Medium / ⚪ Low**
- `audiostream.py` reports the *configured* sfreq not the device's negotiated rate (`:38`)
  and doesn't pin channel count (mono-mean blends unrelated channels, `:64`).
- `oscillator.py:43` **crashes when the frequency input has >1 element**; its pulse output
  is one sample wide and can be missed (`:74`).
- `lslout.py:39` recreates the outlet on any channel-count change, dropping consumers.
- `serialstream.py:91` resampling drops/dups samples at chunk boundaries.
- `setmeta.py:35` bool cast is always True for non-empty strings (`'false'`→True);
  `constanttable.py:60` uses deprecated `np.fromstring`; `oscout.py:39` global
  `last_messages` never reset.

## Coverage & method note

Six adversarial bug-hunt waves (find → independent refute) over: bridge, manager,
node runtime, transport/codec, all frontend subsystems, viewer math, EEG/topomap
interpolation, shared signal helpers, and ~50 of the ~145 node implementations (signal/
array/analysis + user-facing I/O) — plus a dedicated performance pass, a dead-code/
abstraction pass, a feature-inventory pass, a security pass, and a completeness critic.
~210 raw findings; **~130 confirmed**, ~75 refuted as false positives. Several claims were **empirically reproduced** (decode/
re-encode cost incl. the float32 video amplifier, `sliding_window` off-by-one, codec
size/write invariant, `to_data` meta mutation, normalization int-truncation) and the
Python suite was run green (925 passed). The codebase is **well-engineered**; the issues
concentrate in (1) the data-path performance architecture vs the stated scale (incl. the
float32-image convention), (2) threading discipline around the per-node WaitSet/
expression engine, (3) reconnect/modal/selection edge cases in the young undo +
workspace layers, and (4) a systemic in-place-meta-mutation + dtype/2D-shape pattern in
the signal-processing nodes.

## F. Security / robustness

- 🟠 **F1 — default bind is `0.0.0.0`, not the `127.0.0.1` CLAUDE.md §5 specifies**
  (`server.py:154,297`, `manager.py:133`, `--bind` default `manager.py:1576`). The
  WS endpoints have **no auth** (intentional per §13 — but that decision assumes
  *localhost-only*). On `0.0.0.0`, anyone on the LAN can reach `/control` and:
  - **bind a param expression** → the ExpressionEngine **compiles and `exec`s
    arbitrary Python** in the manager process = **remote code execution**;
  - **`save`/`load` arbitrary absolute paths** (write/read anywhere the process can);
  - **browse the entire filesystem** via the un-jailed `fsbrowse` (`fsbrowse.py`,
    no path jail by design);
  - add/remove nodes, change the running patch.
  The no-auth design is defensible *only* bound to loopback. **Fix:** default
  `--bind 127.0.0.1` (as §5 intended); require explicit opt-in for `0.0.0.0` and
  print a clear "exposed to the network, no auth" warning when it's used.
- ⚪ **F2 — `decode_data` is robust** against truncated/oversized/malicious frames
  (memoryview slicing clamps; bad lengths raise rather than OOB or over-allocate;
  the pump catches decode exceptions). Minor: a deeply-nested TABLE can hit Python's
  recursion limit (caught). No action needed beyond awareness.
- ⚪ **F3 — `StringViewer` XSS** (see B14) is the browser-side analogue: node-sourced
  text rendered as unsanitized HTML.
- ✅ **Codec is robust (empirically validated).** `data_byte_size == encoded length ==
  loan-buffer size` and round-trip is exact across contiguous/non-contiguous/0-dim/
  empty/uint8-image/multibyte-string/nested-table inputs — no encode size-write
  mismatch that could over/underflow the zero-copy SHM loan. Decode safely rejects
  truncated/oversized frames. The codec layer is **not** a source of bugs.
