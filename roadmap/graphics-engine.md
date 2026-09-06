# Graphics engine

The third engine — shaders on the GPU, in the style of a TouchDesigner TOP chain — living as a
**peer of the signal and audio engines inside one graph**. First designed 2026-08-09 as a fused
compute engine over a `Field` dtype; redesigned with the owner on 2026-09-06 as a smaller first
step, and this file was rewritten to it. What the first design decided and still stands is kept
below under "kept from the first design". Not built; the working spec is
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
- **`SlotType::Texture`, wire name `TEXTURE`**, the audio rule copied: a texture output feeds a
  texture input in the engine, or an ARRAY input through the tap, and nothing but a texture feeds a
  texture input. An ARRAY input on a graphics node is an upload — the frame becomes a texture the
  shader samples under the input's name — as an ARRAY input on an audio node is a resampled port.
  An unwired texture input samples one shared 1×1 transparent black texture: present, never an
  error. `InTexture` and `OutTexture` are the boundary ports. `Data` stays f32; a texture never
  crosses the wire.
- **`uv` is `(0, 0)` at the bottom-left, the shader convention; a texture's row 0 is the top, the
  image convention.** The prelude's vertex stage is where the two meet, and nothing flips anywhere
  else: an uploaded image displays upright, and a readback's row 0 is the top.
- **The engine is scheduled, demand-driven, and owns one device and one render thread.** No
  window. The clock is a constructor choice: `Clock::External` for the suite, driven by
  `render(frames)`; `Clock::Timer` at 60 Hz for the CLI. A stage renders in a tick only when its
  output has a reader — a subscriber on its data service, or a same-engine consumer that is
  demanded — so a node nobody reads costs nothing. That is TouchDesigner's cook model. The runtime
  is behind a mutex the tick thread and the engine share; the render thread never takes the graph
  lock, and an op waits at most one frame.
- **Settle compiles a plan**: Kahn over texture edges, ties by uid, a `feedback` node ignoring its
  in-edges and running first on its producer's previous frame, a loop with no feedback node
  excluded and named — the audio rule. Resolution is settled state: the universal group `output`
  holds `width` and `height`, 0 following the first wired texture input on that axis, a generator
  with none 512. Every texture is `Rgba16Float`.
- **The control half is shared with audio**, lifted into `goofi-control`: the per-node thread on
  its door, `Desired`, evaluation on arrival, pulses, refresh, reports, bells. What an arrival
  becomes and what a tap publishes is the engine's, behind one trait (`Half`). A second copy of
  that thread is the drift the design principles forbid.
- **One tap serves every reader of an output**: a readback to a `[H, W, 4]` f32 frame published on
  the derived name while anyone subscribes, and nothing while nobody does. The bandwidth fix lands
  on the `/data` socket, not here: `ViewSpec` carries `depth`, the reducer quantizes to `u8` at
  encode when every viewer of the slot accepts it, the codec writes `|u1`, and the browser draws it.
  A colour frame quantizes over `[0, 1]`; a gray one over its own range, carried in
  `meta.reduced.depth`. Engine-agnostic, and it lands first.
- **The engine registers only where a GPU adapter answers, hardware or software.** Where none
  does, there is no graphics engine and no graphics type in the catalog — the demo's rule for
  audio. The suite requires an adapter and fails naming the package (`mesa-vulkan-drivers`), never
  skips. CI's Linux runner installs lavapipe; Windows has WARP; macOS Metal on the runner is
  unverified.
- **A `.wgsl` node's tier is `shader`**, the fourth `Isolation`.
- **The node set was agreed with the owner** (2026-09-06): `Constant`, `Ramp`, `Noise`, `Shape`,
  `Level`, `Transform`, `Blur`, `Composite`, `Displace`, `Lookup`, `Threshold`, `Feedback`,
  `ArrayIn`. `Shader` is out for now. A camera is a Python signal node feeding `ArrayIn`, later.

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

1. **This step**: the engine, the `.wgsl` contract, uploads, references, `Feedback`, the tap and the
   uint8 hop, the thirteen nodes. Proof: `goofi-tests/tests/graphics.rs`, one session under the
   external clock.
2. **3D**: geometry, cameras, a raster pipeline, instancing. Its own spec.
3. **Fields**: ray-marched distance fields and fractals. The exactness tag per node
   (`Exact | Bound(k)`) must arrive with the first field node, never after.
4. **Sight**: vision and generative models as nodes behind the same seam.

## Open

- macOS Metal on the CI runner.
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
