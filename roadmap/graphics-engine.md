# Graphics engine

A second node engine — Rust + wgpu, 60 FPS, 2D and 3D, compute shaders first-class — living as a
**peer of the signal engine inside one graph**. Designed 2026-08-09 with the user; not built.
The engine seam this assumes was specified later, against the audio plane: see
`multi-engine-graph.md`.

## The thesis

TouchDesigner's ceiling is three things it cannot remove without a rewrite: a single-threaded
cook, one unfused GPU dispatch with a full texture round-trip per operator, and a family split
(TOP/CHOP/SOP/MAT/DAT) whose conversion tax it now pays twice. Measured on the target machine: a
32-node pointwise chain at 1080p RGBA16F costs **1.50 ms unfused vs 0.085 ms fused — 17.5×**, and
the unfused chain moves more effective read+write traffic than the card's DRAM peak.

The four answers, in build order:

- **Compile the graph into fused kernels.** Maximal pointwise subgraphs become one WGSL dispatch;
  textures materialize only at fusion boundaries and preview taps.
- **One data model.** A single `Field` dtype — a domain (`Grid<N>` or `List`) carrying named typed
  attributes — replaces the family split outright. No EEG field, no audio field, no video field.
- **Signal metadata survives the crossing.** A signal array entering graphics keeps its shape and
  its positional axis coordinates, which ride the Field as dimension labels and survive a round
  trip back. This is the capability no competitor can copy without rebuilding their type system,
  and the reason this engine lives inside goofi rather than beside it.
- **Off-tick, multi-threaded execution** that never blocks the signal plane.

## Locked decisions

- **One graph, one document, two engines.** Same uid space, commands, history, CRDT doc, `.gfi`,
  palette, sub-patch scopes and panels. What differs is only WHERE a node executes.
- **The wire name is `graphics`**, not `video` — this engine is compute, geometry and distance
  fields, and video describes one output. It lands in `.gfi` files, so it is fixed now.
- **UX cost is proportional to compute cost.** Free reinterpretation is invisible; cheap GPU-side
  conversion is automatic but measured; a **clock crossing gets an explicit bridge node**.
- **Every graphics preview is a server-rendered image stream**, atlas-composited into one encoded
  stream per size class — not one stream per node.
- **A node's body is a composable WGSL fragment.** The compiler fuses; parameters live in a
  buffer. No new shader language: WGSL is the language, naga the IR, and the value is the graph
  compiler rather than new syntax.
- **The render thread never takes the graph mutex.** Lock-free queues drained at frame boundaries.
- **Link legality is enforced by DTYPE**, and engine homogeneity follows by construction: only a
  bridge node may declare a slot of a foreign engine's dtype, so a pure graphics node cannot be
  wired to a signal node at all.
- **No Python tier, ever. No second UI stack.** No second undo system, layout engine, workspace,
  panel registry or control protocol.
- **Shadertoy compatibility is dropped as an architectural constraint** — a clean engine beats
  GLSL import. (naga's GLSL frontend was also measured silently mis-hoisting loads out of `&&`
  guards, which is a correctness bug rather than a coverage gap.)

## Phases

**G1 — Spine.** The engine exists end to end, compute only. wgpu device and a 60 Hz render thread
off the graph tick; graphics nodes as ordinary `Graph` nodes; the `Field` dtype with `Grid`
domains and engine-managed residency; the fragment authoring contract and the accretor with
pointwise fusion; a compile worker with async pipeline swap; clock-crossing bridge nodes both
directions; the atlas preview path including the browser decoder; manager-side link validation.
About eight nodes. *Proof:* a fused 2D field chain at 60 FPS modulated by a live signal source,
with every intermediate visible.

**G2 — Stuff.** Geometry and the first raster pipeline. The `List` domain and named attributes;
point clouds; meshes; **GPU-driven instancing** with `multi_draw_indirect` and compute-written
counts; the raster pipeline (render passes, depth, scene cameras, materials); the free
`Grid → List` reinterpretation; and dimension labels becoming addressable BY NAME from inside
shader code. *Proof:* biosignal data as an instanced point cloud at 60 FPS with GPU-determined
counts. *Deferred to its own spec:* attribute lifetime and aliasing (Blender dropped anonymous
attributes over exactly this, citing VRAM leaks), and the async readback the UI needs to display
"how many instances?".

**G3 — Fields.** The accretor at a second domain type: distance fields where every CSG operator is
a pointwise node, so a whole CSG subtree fuses into one generated `fn map()`. A compute ray
marcher, volumetrics, fractals, marching cubes producing into G2's mesh representation. **The one
thing that must not be retrofitted:** an exactness tag (`Exact | Bound(k)`) per node — `smin`,
non-uniform scale and domain warps degrade a field to a lower bound and the march step must
shrink accordingly. One enum in the node contract, very painful to add afterwards.

**G4 — Sight.** Vision and generative models as nodes. Deferred; G1–G3 must not be shaped by it.
The only obligation is to leave the seam open — an ML node is an ordinary node in its own rate
domain behind a bridge. Ship the naive path first (`ort` with a CPU round trip, measured at
0.355 ms for a 512²×3 tensor). Highest known risk: StyleGAN's ONNX export has years-old open
problems with its custom CUDA ops — prototype the export before it enters a spec.

## Traps worth not rediscovering (wgpu 30, verified 2026-08-09)

- **Push constants are gone**; they are now *immediates* (`Features::IMMEDIATES`,
  `var<immediate>` in WGSL). `set_push_constants` does not exist.
- **`multi_draw_indirect` is no longer feature-gated** — GPU-driven instancing is baseline.
- **Experimental features need an explicit unsafe token** or device creation fails.
- **Subgroups work, but you must omit `enable subgroups;`** (wgpu #5555).
- **Storage and uniform pointers may not be user-function parameters** — naga does not implement
  the extension that would lift this, so generated node bodies must take values.
