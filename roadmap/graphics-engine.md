# Graphics engine

The third engine — shaders on the GPU, in the style of a TouchDesigner TOP chain — living as a
**peer of the signal and audio engines inside one graph**. First designed 2026-08-09 as a fused
compute engine over a `Field` dtype; redesigned with the owner on 2026-09-06 as a smaller first
step, and this file was rewritten to it. What the first design decided and still stands is kept
below under "kept from the first design". **The first step is BUILT as of 2026-09-06** — the
engine, the `.wgsl` contract, uploads, references, `Feedback`, the tap, the uint8 hop and all
thirteen nodes. What is left is phase 2 and the Open list. The working spec is
`docs/superpowers/specs/2026-09-06-graphics-engine-design.md`.

The seam this engine assumes is `multi-engine-graph.md`. The audio engine, `audio-engine.md`, is
the template for a scheduled engine and is followed here wherever the two are the same thing.

## What it is for

Generative visuals modulated by everything else in the patch: a chain of shader nodes, each a
texture in and a texture out, driven by biosignals through uploads and references. The first step
is 2D and pointwise. 3D, geometry and ray marching are later phases and must not be shaped for
before they arrive.

## Locked decisions

- **A `.wgsl` file is a graphics node.** Its type name is the file stem. The first comment block
  is the header: it opens with `goofi` and carries the probe schema every other node speaks, as
  JSON — `doc`, `tags`, `inputs`, `params`, and `feedback`. The engine adds the one output, `out`.
  The engine generates the prelude and appends it AFTER the file's text, so a naga error names the
  file's own line: `time`, `resolution`, `samp`, `p` (the params as one uniform struct, one scalar
  per param), one `texture_2d<f32>` per input, and the full-screen stages that call
  `shade(uv) -> vec4f`. No Rust SDK, no prebuild, no toolchain to author: a text editor is the whole
  requirement. A shipped node is a file in `node-bundles/graphics/`, embedded as a `.py` is. The
  alternatives were an `.rs` file against a graphics SDK, which needs cargo to ship a string, and a
  `Shader` node with the WGSL as a text param, which cannot carry named params because a manifest is
  per type; the second may still arrive as a convenience over the same compile path, and is out for
  now by the owner's ruling.
- **`SlotType::Texture`, wire name `TEXTURE`**, the audio rule copied — LANDED 2026-09-06: a
  texture output feeds a texture input in the engine, or an ARRAY input through the tap, and
  nothing but a texture feeds a texture input. Audio and a texture are now ONE property,
  `SlotType::is_engine_local` — a kind that never crosses the wire — so `feeds` states the rule
  once and a third such kind needs no new arm. The two scanners' foreign-slot refusals became one
  `goofi_node::foreign_slot(intro, own)`, which names how a node of the kind it found IS written.
  An ARRAY input on a graphics node is an upload — the frame becomes a texture the shader samples
  under the input's name — as an ARRAY input on an audio node is a resampled port. An unwired
  texture input samples one shared 1x1 transparent black texture: present, never an error.
  `InTexture` and `OutTexture` are the boundary ports. `Data` stays f32; a texture never crosses
  the wire. The viewer kind a slot OPENS with is now a table too (`vocab::default_kind`,
  projected as `DEFAULT_KIND`): a texture draws as an image, an array and audio as a line. The
  suite's graphics-shaped skeleton was renamed `skelgfx`, because `graphics` is the real engine's.
- **`uv` is `(0, 0)` at the TOP-left — WGSL's own texture space — and so is every row order in the
  engine.** REVISED 2026-09-06, and the first design of this rule was wrong. It said `uv` was
  bottom-left, the OpenGL convention, with the vertex stage reconciling the two; the two cannot be
  reconciled in one value. WGSL samples a texture with `v = 0` at row 0, so a bottom-left `uv`
  makes `textureSample(input, samp, uv)` a vertical FLIP, and every node body would have to write
  `1.0 - uv.y` to undo it — the thing the rule existed to prevent. One convention instead: `uv`,
  the sampler, texture memory, an upload's rows and a readback's rows all put row 0 at the top, so
  a pass-through body is a copy and nothing flips anywhere. The upload of the suite's gradient
  fixture is what found it; a chain of uniform colours cannot see a flip, which is why the first
  four scenario steps were green over it.
- **The PROCESS owns one device and one compile thread; the engine owns one render thread** —
  LANDED 2026-09-06, and the "one device per engine" of the first design is REVISED. Two reasons,
  both measured. A Vulkan instance opened and closed per engine unloads the driver under a sibling
  engine still inside it. And the driver crashes inside its own shader compiler whenever one
  thread builds a pipeline while another encodes, submits or frees on the same device — so every
  operation on that device takes one process-wide gate: a compile, a tick, a birth, a teardown,
  and every drop of a class or an instance that owns a pipeline. Eight concurrent engines crashed
  four runs in five before the gate and none in twenty after; one engine doing thirty-two times
  the work never crashed at all, which is what named concurrency rather than volume. In
  production there is one graph and one engine, so the gate is uncontended except against the
  compile thread, which is exactly what it is for. No
  window. The clock is a constructor choice: `Clock::External` for the suite, driven by
  `render(frames)`; `Clock::Timer` at 60 Hz for the CLI. A stage renders in a tick only when its
  output has a reader — a subscriber on its data service, or a same-engine consumer that is
  demanded — so a node nobody reads costs nothing. That is TouchDesigner's cook model. The runtime
  is behind a mutex the tick thread and the engine share; the render thread never takes the graph
  lock, and an op waits at most one frame.
- **A plan is replaced whole, never edited.** `Input::Stage` is an index into the plan's own
  stage list, so dropping one entry renames every entry after it; and a stage that outlives the
  node it was built for draws the departed node's pipeline into the state a restart just made at
  the same uid. A node leaving therefore drops the WHOLE plan, and the settle that ends the batch
  builds the next one. Between the two, the engine draws nothing.
- **Settle compiles a plan**: Kahn over texture edges, ties by uid, a `feedback` node ignoring its
  in-edges and running first on its producer's previous frame, a loop with no feedback node
  excluded and named — the audio rule. Resolution is settled state: the universal group `output`
  holds `width` and `height`, 0 following the first wired texture input on that axis, a generator
  with none 512. Every texture is `Rgba16Float`.
- **The control half is shared with audio**, lifted into `goofi-control` — LANDED 2026-09-06,
  before the engine that needs it: the per-node thread on its door, `Desired`, the subscriptions,
  evaluation on arrival, pulses, refresh, reports and bells are one implementation. What an
  ARRIVAL becomes and what a tap publishes is the engine's, behind `Half` — four methods, of which
  two are defaulted. `spawn` takes a FACTORY that builds the half on the control thread, so a half
  may hold what does not cross one: audio's cpal stream and MIDI connection are built there and
  never move. Audio's own additions (the clock's rate, what drives it, a plugin editor's writes)
  are an `AudioShared` beside the generic one, and the scalar readers a plan needs are the control
  crate's. A second copy of that thread is the drift the design principles forbid.
- **One tap serves every reader of an output**: a readback to a `[H, W, 4]` f32 frame published on
  the derived name while anyone subscribes, and nothing while nobody does. The bandwidth fix lands
  on the `/data` socket, not here, and it is engine-agnostic — LANDED 2026-09-06, before the engine.
  `ViewSpec` carries `depth`, `plan` folds it to `u8` only when EVERY admitted viewer accepts it,
  `reduce::quantize_u8` quantizes the already-reduced frame, `codec::encode_u8` writes the `|u1`
  body the browser decoder now parses, and the image viewer uploads it as `R8`/`RGB8`/`RGBA8`. A
  colour frame (3 or 4 channels) spans `[0, 1]`, the convention a viewer clamps to anyway; anything
  narrower spans its own finite range, carried as `meta.reduced.depth = {lo, hi}` so the viewer maps
  a texel back before its own range logic applies. `Data` stays f32 everywhere inside the graph,
  so the frame the INSPECTOR and an agent's query read is mapped back and reported as `float32`:
  a texel is not a value, and every reader of the one folded stream that is not the image viewer
  would otherwise summarize 0..255. The `u8_image` golden case pins the bytes in both directions
  — the legacy Python encoder does write a `|u1` body for a uint8 array, so this path has real
  cross-language parity rather than only a Rust-side assertion.
- **The engine registers only where a GPU adapter answers, hardware or software.** Where none
  does, there is no graphics engine and no graphics type in the catalog — the demo's rule for
  audio. The suite requires an adapter and fails naming the package (`mesa-vulkan-drivers`), never
  skips. CI's Linux runner installs lavapipe; Windows has WARP; macOS Metal on the runner is
  unverified.
- **A `.wgsl` node's tier is `shader`**, the fourth `Isolation` — LANDED 2026-09-06. The way back
  from the byte a shared cell holds is a SEARCH of `Isolation::ALL`, not a match: the wildcard arm
  that was there read the newly added fourth tier as the third, and said nothing.
- **The node set was agreed with the owner** (2026-09-06): `Constant`, `Ramp`, `Noise`, `Shape`,
  `Level`, `Transform`, `Blur`, `Composite`, `Displace`, `Lookup`, `Threshold`, `Feedback`,
  `ArrayIn`. `Shader` is out for now. A camera is a Python signal node feeding `ArrayIn`, later.

### What the engine core landed (2026-09-06)

`goofi-graphics` is the crate: one `Gpu` (adapter, device, queue, the shared sampler, the 1x1
transparent texture every unwired input samples, and the bind-group-layout cache), `shader` (header
parse, prelude generation, naga validation, uniform packing), `scan` (one file to one class, and
the thread every pipeline is built on), `plan` (Kahn, demand, resolution inheritance), `runtime`
(the tick and the readback) and `half` (what an arrival becomes, over `goofi-control`).

Three decisions were taken while building it, and each is a rule the code now states once:

- **A compile never blocks an op.** A driver takes its time over a pipeline, so the class holds a
  cell and the compile thread fills it; the plan picks it up at the tick after. A pipeline that
  just arrived sets `replan` and rings the waker, because a compile changes nothing until a settle
  plans for it.
- **The universal group is the ENGINE's, not literally `common`.** Graphics adds `output`
  (`width`, `height`) where signal adds `common`. The palette's page-order contract used to name
  `common` by hand; it now reads `universal_decls` per engine, so the law is "the author's pages in
  declared order, then the engine's own" and a fourth engine needs no new arm.
- **A shipped name may now belong to two engines.** `Constant` and `Level` name a signal and a
  graphics node, an audio and a graphics node. That is the designed behaviour — `node add` refuses
  the bare name and lists both candidates — and the cost is that the callers of those two names
  qualify. Nothing was renamed to dodge it.

`session status` grew a `graphics` block beside `audio`: the clock, the adapter and backend that
answered, and the frame, stage and tick-time counters. It is null where no adapter answered.

Shipped so far: `Constant` and `Level`. The suite's session is `goofi-tests/tests/graphics.rs`,
under the external clock, and it proves the palette row, a readback of a real render, the `output`
resize and the size a wired input inherits, an HDR value past 1 surviving a chain, an unwired input
as transparent black rather than a fault, a `.wgsl` that does not compile as a greyed row carrying
naga's own line number in the FILE's numbering, an authored node loaded and reloaded through
`library refresh`, a node nobody reads costing no work until a reader arrives, a restart as a
rebirth on a new generation with the corpse's service gone silent, and a remove that leaves the
rest standing.

### What the node set and the audit landed (2026-09-06)

All thirteen ship. `ArrayIn` is the door in from the rest of the patch, `Feedback` the one node a
loop closes through, and the other eleven are sources and filters. The scenario draws a frame from
EVERY shipped type and checks the node stands with no error: a palette row says naga read the
file, and only a frame says this device built the pipeline behind it.

The audit of the engine core found seven defects, and each is now a rule rather than a patch:

- The plan is replaced whole (above), which is also what a render thread and an op batch can agree
  on without a second owner.
- An unwired ARRAY input reads transparent black FROM THE SETTLED VIEW, not from an unlink event.
  It kept sampling its last frame for ever, and `Half::unwired` was the second owner that made it
  possible.
- The reports are drained under ONE lock. A clone followed by a clear lost whatever a control
  thread pushed between them.
- A texture wire is a plan edge, so it no longer rings the consumer's door — the rule the audio
  engine already had.
- The render clock is its own argument to `fresh_graph`. It was read off the audio clock, so a
  demo — which asks for no audio engine — would have got a graphics engine nothing drove: every
  node green, and every node dead.
- An idle tick returns before it submits. It was submitting an empty command buffer and blocking
  on it at 60 Hz on every machine with a GPU.
- A loop head keeps the axis it asked for instead of falling back to 512 on both.

The duplication with the audio engine went into `goofi-control` rather than being noted: `Faults`
answers what moved between two settles, `Shared::drain` hands the reports over, and `Handle` holds
what it last sent so `send_if_changed` is the only door. Each of those was a copy in two engines,
and the drain copy had already drifted into the defect above.

### What the program audit landed (2026-09-06)

Three finders over the whole program. Every finding they raised is fixed, and four of them are
rules the code now states once:

- **A pass cannot read the texture it writes**, so a node wired to its OWN output is out of the
  plan and faulted, feedback node or not. One that tried made the tick's whole command buffer
  invalid, so EVERY stage went silent while every viewer held its last frame and no node showed an
  error. A feedback loop needs a second stage for the frame to age in.
- **An upload past the device limit is no upload.** `Upload::of` refuses a frame wider or taller
  than `MAX_SIZE`, because a texture the device cannot make takes the same whole tick down. A
  ten-second EEG buffer at a kilohertz is 10000 wide, so this was one wire away.
- **A reducer's SUBSCRIPTION follows demand.** It keeps its cache for the node's life, as before,
  but lets the feed go after a second with nobody asking. An attached subscriber is what a
  scheduled engine reads as "somebody wants frames", so a reducer that never detached kept a GPU
  node rendering at 60 Hz for ever after one glance — which is the "an idle patch is free" claim,
  false in the real app and true only in a suite whose probes detach.
- **The `output` size is settled state, so it takes no reference or expression.** Graphics is the
  first engine with a universal group, and a binding on one was accepted and then silently dead.
  The engine now states the verdict on the whole group at every settle, so the node carries the
  refusal in words and it clears itself when the reference goes.

Two smaller ones: the compile thread gives its finished job back BEHIND the gate, because the job
can hold the last handle on a pipeline whose class was already dropped; and `Isolation::language`
lost the wildcard that answered `python` for a `.wgsl`.

### The presentation window (2026-09-06)

`Window` is a `.wgsl` node like any other, with `"window": true` in its header — the flag `feedback`
already showed the shape of. It shows its input on the machine's own screen and still has the one
output every graphics node has, so a viewer or another node reads the same texture.

- **The window host left the audio engine for `goofi-window`.** An audio plugin's editor and a
  graphics node's window are one screen, and a process has one main thread to give them. Neither
  engine owns it now.
- **The window is the FRAME's size**, set by the universal `output` group, so nothing scales and no
  platform needs a scaler. The title is the node's own name, because a patch may open several.
- **A screen is a READER.** `Stage::read()` is the one answer to "does anything read this stage" —
  a subscriber on its data service, or a window — and it drives both the demand walk and whether a
  readback is made at all. Without the second half the window opened and stayed black.
- **The frame is the one the tap already reads back.** A wgpu swapchain would be better and is not
  reachable: Vulkan wants an Xlib `Display*` or an xcb connection, our X11 host is x11rb's pure-Rust
  connection, and neither `x11-dl` nor `as-raw-xcb-connection` is in the offline registry. So the
  window is fed the readback, converted to RGBA and blitted — `PutImage` in bands on X11,
  `SetDIBitsToDevice` on Win32, an `NSBitmapImageRep` on macOS. The buffer handed to a screen is
  RGBA, row 0 top; the two platforms that want BGRA share one swizzle. Revisit when a connection
  pointer can be had.
- **A screen is slower than a clock, so the presenter is latest-wins with ONE job outstanding.** A
  job per frame on the window thread's unbounded queue starved every other job on it — an op among
  them — because the clock always posted the next frame before the screen had finished the last.
  The in-flight flag clears AFTER the draw, which is what lets the queue empty between two frames.
- **An op never waits on a render.** The engine stopped taking the runtime lock: it appends a
  `runtime::Cmd` to an inbox the tick drains. A tick is long — 165 ms at 512 square in a debug
  build — and a lock a render holds is a lock an op cannot have; std's mutex is not fair, so a
  clock that overran its period locked the graph out for good rather than merely being slow.

Proved by hand on this machine: a `Ramp` at 45 degrees into a `Window` at 480x270 puts a real X11
window on the desktop, and `xwd` reads black at the top-left and white at the bottom-right — the
gradient, the right way up. The suite proves the seam through the screenless host: a window asked
for, sized, fed, counted in `session status`, and closed with its node.

## Kept from the first design

- **The wire name is `graphics`**, not `video`: the engine's registered id, the first half of
  every `graphics:Name` type id.
- **The node body is a plain WGSL function that takes values**, so the fusion of a pointwise chain
  into one kernel stays possible without touching a node file. Fusion is not built and not
  scheduled; it is the answer if a measurement ever shows the per-node dispatch cost matters.
- **Cross-engine data follows the seam**: latest-wins by decree over the derived names; the tap is
  the crossing out, an ARRAY input the crossing in. No bridge node is needed beyond `ArrayIn`.
- **No Python tier, no second UI stack.**
- **Shadertoy compatibility is not a constraint.** naga's GLSL frontend was measured mis-hoisting
  loads out of `&&` guards; WGSL is the language.

## Phases

1. **BUILT 2026-09-06**: the engine, the `.wgsl` contract, uploads, references, `Feedback`, the
   tap and the uint8 hop, the thirteen nodes. Proof: `goofi-tests/tests/graphics.rs` — one session
   under the external clock, and a second one under the timer clock the binary runs.
2. **3D**: geometry, cameras, a raster pipeline, instancing. Its own spec.
3. **Fields**: ray-marched distance fields and fractals. The exactness tag per node
   (`Exact | Bound(k)`) must arrive with the first field node, never after.
4. **Sight**: vision and generative models as nodes behind the same seam.

## Open

- macOS Metal on the CI runner, and Windows WARP. Only Linux is proved, on lavapipe in CI and on
  an NVIDIA card by hand.
- The device gate serialises the tick against a compile. It was measured as necessary on one
  driver; whether every driver needs it is not known, and the cheap way to find out is to try
  another machine before making the gate narrower.
- The BROWSER has not drawn a graphics frame by hand yet. The binary has: a release `goofi` on an
  RTX 4090 through Vulkan holds 60 fps with one reader, at 512 square and at 1280x720 alike, and
  the worst tick of a run is the first, which allocates. It renders nothing while nothing reads,
  which is the demand rule holding outside the suite. A DEBUG binary manages 14 fps at 512 square,
  because the readback converts a million texels from f16 in an unoptimized loop; that is the
  build, not the design, and it is why a rate must never be read off `cargo run` alone.
- The readback waits on the render thread once per tick; a double-buffered readback is the lever
  if a tick misses 16 ms.
- The reducer's area kernel at 1080p; a GPU-side downscale needs the viewer's size to reach the
  engine, which the constraint algebra does not carry.
- A feedback chain restarts from black at a resize.
- A vector-typed param (a colour as one `vec4f`) needs a layouter and a `Param` kind that does
  not exist.

## Traps worth not rediscovering (wgpu 30, verified 2026-08-09 and 2026-09-06)

- **Push constants are gone**; they are *immediates* (`Features::IMMEDIATES`, `var<immediate>`),
  and `PipelineLayoutDescriptor` spells the range as `immediate_size`.
- **`request_adapter` and `request_device` answer a `Result`**, not an `Option`.
- **`multi_draw_indirect` is no longer feature-gated.**
- **Experimental features need an explicit token** (`ExperimentalFeatures::disabled()` in the
  device descriptor) or device creation fails.
- **Subgroups work, but you must omit `enable subgroups;`** (wgpu #5555).
- **Storage and uniform pointers may not be user-function parameters** — a node body takes values.
- **`copy_texture_to_buffer` rows align to 256 bytes** (`COPY_BYTES_PER_ROW_ALIGNMENT`); the
  readback unpads.
- **`device.poll` takes `PollType::Wait { submission_index, timeout }`**, a struct variant.
- **`InstanceDescriptor::new_without_display_handle()`**, taken by value, is what opens a device on
  a machine with no display; asking for a display handle refuses one on a server or a CI runner.
- **`on_uncaptured_error` takes an `Arc`**, `push_error_scope` answers a guard whose `.pop()` is
  the future, `get_mapped_range()` answers a `Result`, and `RenderPassDescriptor` needs
  `multiview_mask`.
- **An error scope is keyed by THREAD**, so a scope pushed on the compile thread cannot catch an
  error raised on the render thread.
