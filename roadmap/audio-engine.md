# Audio engine

The second engine — a peer of the signal plane inside one graph. Designed with the user on
2026-08-24/25 against measurements, not instinct; every number below was taken on the target machine
and the harnesses are named where they still exist. Designed in full with the user on 2026-09-02, section by section: the node
contract is goofi's own, CLAP is out, a reload no longer crossfades, and params gained a
reference source (`param-sources.md`). Not built; proved, when it is, by
`goofi-tests/tests/audio.rs` — one session under the external clock, every action through the
op vocabulary, every probe through a `/data` viewer on an audio slot.

The seam that lets a second engine exist at all — the `Engine` trait, the settle point and what
the split deletes — is `multi-engine-graph.md`. What is here is audio.

## What it is for

Real-time generative audio through modular synthesis, modulated by everything else in the patch. It
is not a DAW and does not compete with one: what Ableton or Logic provide is not reinvented here.
The nearest neighbour is VCV Rack — a graph of modules with CV, where a plugin host is one module
among many — and it is a neighbour, not a template. The consequence that shapes the most decisions
below: a node reload is a discrete authoring event, and a click or a short gap at one is acceptable.

## What makes it different from the signal plane

The signal plane is latest-wins with no queue, one self-paced thread per node, and no scheduler. A
synth needs every sample in order, so none of that carries over. The audio engine is **synchronous,
centrally scheduled, and in-process**: one block of 64 frames at the device rate, every node visited
once in topological order, buffers passed node to node as slice indices into one arena. No iceoryx2
on the audio path. A late block is an audible artefact rather than a dropped frame.

The old Python implementation already paid for getting this wrong. `AudioOut` prepended a 100-sample
linear crossfade to every block (`transition_samples`) to hide the discontinuity that latest-wins
delivery produced. That is the artefact, papered over.

## Locked decisions

**Every audio node implements one goofi trait, and a plugin format is an adapter behind it.** A
shipped node and an authored node are the same thing: one `.rs` file — in `nodes_audio/` or in the
patch's `workspace/nodes_audio/` — built against the SDK crate at goofi's own version by the one
pipeline `node-sources.md` describes, and loaded as a `cdylib` through a `#[repr(C)]` vtable the
SDK emits, which the author never sees. The shipped folder is prebuilt at goofi's build time and
embedded, so running needs no toolchain. A VST3 plugin is an adapter that implements the same
trait. The plan compiler, the Kahn sort, the arena, the watchdog and the viewer
tap see ONE trait and no format enum. Audio I/O is the engine's own — a device is never a node's to
open — and it stands behind the same trait. Forking a shipped node is a copy.

**This replaces CLAP, which was locked on 2026-08-25 and reviewed twice (below).** What reversed
it: CLAP params are f64-only, so every string param, every `refresh` param and every host-resolved
resource rode a vendor extension — a goofi ABI wrapped in CLAP, growing with every modular need
CLAP has no answer for: inferred channel counts, spectral-bin ports, CV inputs, cross-domain
modulation. Under the goofi trait an audio node declares the same `ParamDecl` a signal node
declares, and the document holds one param vocabulary. The dependency that left with it was the
loosely maintained one — `clack-host`, `clack-plugin` and `clap-sys` — where `libloading`, `cpal`,
`rtrb` and the VST3 bindings were needed either way. The precedent the first review missed: REAPER
and Ardour ship their stock effects as plugins because a channel strip IS what a plugin format
models; every modular environment — VCV Rack, Max, Pure Data, Bitwig's Grid — owns its module
contract and hosts a plugin as one special module. Hosting a CLAP-only plugin is deferred, not
lost: an adapter behind the same trait, on which no builtin depends, if a CLAP-only plugin ever
matters — nearly every plugin that ships CLAP ships VST3. Exporting a goofi node to a DAW through
`clap-wrapper` is dropped as a goal.

**The trait is five methods, and the block is three fields.** Landed 2026-09-02 as
`backend/audio/goofi-audio-sdk`, proved in use by the skeleton engine's session before any engine
exists. `channels(ins, params, outs)` answers per-output channel counts at plan compile — the
engine passes the output count, so the default needs no hook into the manifest and the trait
stays object-safe; `prepare(rate)` allocates once on the control thread;
`process(&mut Block)` runs on the audio thread and allocates, locks and blocks on nothing;
`feedback()` marks the one kind of node a loop may close through; `save`/`load` carry state a
param cannot — a VST3 plugin's own, a recorded loop — and a node that returns nothing leaves
nothing behind. `Block { ins, outs, params }`: every block is exactly `BLOCK = 64` frames, the
engine renders whole blocks and its FIFO carries a callback's surplus, so a channel is
`&[f32; BLOCK]` — a fixed-size array the compiler bounds and vectorizes. The rate arrived at
`prepare`; a second copy in the block is a duplicate.

**A param is a `Port`.** `Block.params` is one `Port` per declared param, read through the same
`chan(c)` a signal port is. A constant, an expression or a foreign reference is a scalar: the
engine keeps the last value it wrote beside the param, loads the atomic once per block, and
refills the 64-float region only when the value moved. An audio reference costs nothing — the
param's region IS the producer's region, with the producer's channel count. `Float` and `Int`
arrive as their value, `Bool` as 0/1, a `Str` with `options` as the option's index, a free-text
`Str` as silence. Exactly one source holds per param, by `param-sources.md`; no node carries a
CV port, because every modulatable quantity is a param.

**A port carries a signal with no default; a param carries a value with a default.** That is the
whole split. `SlotType::Audio` is the fourth slot type, and the graph's one link door holds the
rule: an audio output feeds an audio input, or an ARRAY input through the engine's tap, and
nothing but audio feeds an audio input — so `Osc.out → Buffer` is a cable and `Oscillator.out →
Osc` is a `SignalIn`. NO new `Value` variant and NO new GOOF dtype, because audio buffers never
cross iceoryx2 or the wire; a signal node that declares an audio slot is greyed out with the
folder named. An unwired input reads one shared silent region, `channels = 1`,
`wired() == false`: present, silent, never an error. A `multi` input SUMS its wires at the jack.

**Every signal is audio-rate numbers in a standard range, and the engine treats them all the
same** — the VCV Rack stance with float ranges instead of volts. Bipolar signals live in
`[-1, 1]` and `1` is full scale at the output; unipolar signals — a gate, a velocity, an envelope
— in `[0, 1]`; a gate is HIGH at `>= 0.5` and a trigger is its rising edge; pitch is volts per
octave, zero at C4, unbounded, so transposition is an addition and a control-rate Hz value is one
expression, `log2(hz / 261.63)`. A gate arrives at a `Bool` param — constant for a manual trigger,
expression for a signal-plane detector, reference for `MidiIn.gate` — and there is no event type
anywhere. Polyphony is channels: a 4-channel gate referenced into an envelope is four voices.

**Channel counts are inferred, never configured.** `channels` is evaluated once per
node INSIDE the Kahn loop, over ports and referenced params — inputs are settled before a node is
visited, so there is no fixed point. The SDK default is `max(ins).max(1)` for every output; a panner, a mixdown or
`MidiIn` (whose count is its `voices` param) overrides it. A node reads a param by index, never
by string: `params!` is ONE list that is both the manifest's params and the `P::NAME` indices, so
the order has one owner. Coercion lives on the edge — `Same`,
`Broadcast`, `PadOrTruncate` — and mono to stereo is one region read twice. **The count is dynamic
per block, not per instance**: a node is prepared once for `MAX_CHANNELS = 16` and reads
`port.channels` each block, so wiring a stereo source into a mono chain changes the plan and
nothing else — no reinstantiation, no state reset. A layout tag (`Speakers`, `Bins`) is NOT
carried: a port is a count, and the tag arrives with the first node that needs one.

**The audio device belongs to the engine, not to a node.** The device callback is the clock; the
engine opens ONE cpal output stream, asks for a fixed 64-frame buffer, and carries whatever the
backend actually delivers through its own FIFO. Without a device the clock is external —
`AudioEngine::drive(frames)`, the concrete door the test harness reaches through `as_any_mut` —
and everything downstream is the same code. One output device per engine: the stream follows the
device the `AudioOut` nodes name, and a second `AudioOut` naming another device faults until they
agree. Device selection is an ordinary param — `Param::Str { options, refresh: true }` carrying
cpal's `DeviceId`, empty meaning the host default; an absent device faults the node at insert. No
machine-local sidecar.

**The audio thread owns one `Runtime`: a slab of instances, the plan, the arena, the atomics.**
The plan holds slab INDICES, so a topology edit is a new order and new regions over the same
instances — adding a cable resets nothing. Every routine change is a message over an `rtrb`
ring (`Insert | Remove | Plan | Grow`), each a pointer move, and every retired box or plan
returns on a second ring to be dropped off-thread. The plan crosses as an owned value, NOT
`ArcSwap`: it holds per-node staging and needs `&mut`. The runtime sits behind a mutex for
exactly one reason — a clock swap — which the callback `try_lock`s and the control thread takes
only across a device switch; a node add never costs a block. The arena is one `Vec<f32>`: a
region per output port and per scalar-sourced param, scratch for summing, one silent region; no
liveness-based reuse until a measurement asks. Landed 2026-09-02 as `backend/audio/goofi-audio`
under the external clock (`drive(frames)`, the device arrives with its own step): `channels` is
answered on the control thread by a TWIN from the same factory, so the box that processes never
leaves the audio thread; the SDK's `Port::chan` is the one owner of the channel rule — a
one-channel port is on every channel, a channel past a wider port's count is silence — so no
coercion enum exists and the engine's sum reads a part through the same door a node does; a
scalar region is refilled when it no longer holds its atomic's value, so a fresh arena needs no
memory beside it; a loop with no feedback node excludes only its members, and what the loop feeds
runs on silence at that jack; the lowest-uid `AudioOut` is heard until the device sums them; a
binding the graph did not ship (`live == false`) is never a plan edge; `Osc` takes `pitch` in volts
per octave, as the owner ruled, and Hz is a conversion; and the three shipped nodes are compiled
into the engine, one file each in the author's form, until the audio ABI moves them into
`nodes_audio/`.

**Topological order, Kahn, ties broken by `Uid`.** Not iteration order — `Graph::nodes` is an
`IndexMap` whose order moves across save, load, undo and paste. A loop closes only through a node
whose type answers `feedback() == true`: the sort ignores its in-edges, it runs first each block
on the regions the previous block left — one block of delay, ~1.3 ms, no cycle-breaking
heuristic. A cycle with no such node is excluded and named through the fault channel. `Feedback`
is an ordinary shipped node that answers `true` and copies input to output.

**One node, two halves, and the control half owns everything that touches the OS.** The engine's
main-thread side behind the multi-engine trait is the CONTROL half: params arrive as trait
propagation, health leaves through the drain queue, a sample file, a MIDI port or a device name
is resolved here and handed to the DSP half as a buffer. The DSP half is `process`, and it owns
arithmetic. The DSP object is `Send` and moves to the audio thread inside the plan.

**The control half is doorbell-driven, and the audio thread is not.** This amends
`multi-engine-graph.md`'s "a scheduled engine has no doorbells": an expression is Python and
cannot evaluate on the audio thread, so every boundary consumer of this engine is a control-half
thread, woken like a signal node is — by the producer ringing the consumer node's own doorbell.
It evaluates an expression or copies a referenced scalar on arrival and stores the result in the
param's atomic; the audio thread reads it at its next block start. Latest wins, no event list, no
queue. Landed 2026-09-02 as `control.rs`: a THREAD PER AUDIO NODE, parked on that node's door with
a 10 ms tick — the shape a signal node already has, and no wait set — and the ONE writer of the
node's atomics, constants included, so a dropped binding cannot race a settle's constant write.
Settle hands it its whole desired state when that changes; it diffs subscriptions by service name.
A binding with no stream variable re-evaluates every tick, since there is no run to evaluate
before. The signal engine now plans a sequence for any consumer that RINGS, skipping the Apply
phase for a foreign one — before, a signal producer was never told to ring an audio door. The
ring is latency, not delivery: the tick would drain the same subscribers within 10 ms.

**One tap serves every reader of an audio output.** Landed 2026-09-02 as a plain publish on the
derived name, keyed on the data service's subscriber count: the audio thread feeds every output's
ring after every block (a 64 KB ring per output, the newest block dropped when it is full), and
the node's control half drains it each tick and publishes one `[C, T]` frame with `sfreq = rate`
while anyone subscribes. A viewer, `node snapshot` and a signal consumer are all plain subscribers
on the name every other engine's output already carries — no `SlotFeed` arm in the reducer, no
bridge reaching into the engine, and no plan recompile when a tap opens or closes, which is why
this replaced the spec's reducer arm. Latest-wins by decree.

**An in-order signal→audio crossing is a BRIDGE node this engine owns.** `SignalIn` has a
`SlotType::Array` input; its control half is rung by the signal producer, pushes the frame's
samples into a per-node ring, and the DSP half resamples linearly from the frame's `sfreq` to the
device rate. Latency is one source frame; underrun holds, overrun drops the oldest. This is for
ORDERED samples — sonification, playback. A control value or a gate from the signal plane is a
reference into a param, never this. Landed 2026-09-02 with the crossing owned by the ENGINE, not
the node: any `Array` input of an audio node is delivered as an audio-rate port. The control half
resamples the frame linearly from its `sfreq` to the rate — no `sfreq` is one sample per sample,
so a control value is held — and enters it whole as one chunk headed by channel count and length;
the audio thread reads it one sample per sample and holds the last on underrun; a frame that does
not fit the one-second inbox is dropped whole. `SignalIn` is a copy, and a new channel count
re-plans through `dirty()`.

**MIDI is a node that emits signals, and no engine mechanism knows it exists.** `MidiIn` outputs
`gate`, `pitch` and `velocity`, each `voices` channels wide; its control half runs `midir` and
timestamps messages into a ring; its DSP half applies note-on/off at the sample position and
allocates voices round-robin. An envelope's `gate` REFERENCES `MidiIn.gate`. `MidiCC` — one CC
number per node, one output — is decided and deferred. `AudioIn` is the same shape over a cpal
input stream; a second device drifts, and correction waits for a measurement.

**A node reload is a discrete event, and it does not crossfade.** A reload is the graph's own
restart — remove plus insert with a fresh generation, state blob flushed and reloaded across it —
at a block boundary, and a click or a short gap there is accepted: a reload is an act of
authoring, and this engine is not a DAW. So is a device switch, which re-prepares every node
across the runtime lock with silence during it. Reinstantiate per node, never per graph.

**No plugin GUIs, and goofi draws every param itself.** goofi is a server that prints a URL and
never opens a window (principle 5), and its UI is a browser replica. A VST3 editor hands the host a
NATIVE window handle with no offscreen form, so `IPlugView` is never created and the parameter
list is the whole UI. **The cost is named, not hidden: a plugin whose value IS its editor is
degraded to a parameter list.** Skipping the editor also skips the platform event loop and timer
plumbing, which is where every reported hosting difficulty lives.

**VST3 hosting is one more implementor of the trait, over the MIT `vst3` bindings**, and it is
built after the goofi nodes. The COM lifecycle, the bus/param/state plumbing and the threading
rules are ours; the bindings (0.3, after the SDK went MIT in October 2025) are raw. Scanning runs
each bundle in a CHILD `goofi` process — the one binary is its own scanner — so a plugin that
crashes at load is refused and named, never taking the server down; results are cached by path,
mtime and size. The manifest is derived: buses to ports, parameters to `ParamDecl` — stepped with
≤ 64 steps a `Str` with the plugin's own value strings, stepped otherwise an `Int`, continuous a
normalized `Float` in `[0, 1]` with the plugin's display string in its doc; hidden, read-only and
bypass parameters omitted. MIDI is VST3's language, not goofi's: an instrument gets `gate`,
`pitch` and `velocity` params and the adapter emits note events from them, per channel, so a
16-channel gate referenced in is 16 voices. Bus arrangements are the menu `channels` selects from,
and an arrangement change reinstantiates the plugin — the one place that rule survives. A plugin
param under an audio reference is sampled once per block. The processor moves to the audio
thread inside the box under one `unsafe impl Send`, justified by the VST3 contract and commented
as the deviation it is. No usable VST3→CLAP wrapper exists (DISTRHO Ildaeil does not even bridge
params), which is one more reason the centre is a trait and not a format.

**Opaque per-node state lives in `workspace/.goofi/state/<uid_hex>/`**, written by `save` when a
node's box comes back from the runtime and only when `save` returned bytes, read by `load` at
insert. That one choice inherits the archive, the dirty fingerprint, the atomic load swap and undo
of a delete. A goofi node's blob never carries a param value, so the `.gfi` record is the one
authority by construction; a VST3 plugin serializes its params INTO its state, so its load is blob
first, then the param record on top.

**One node editor panel for every engine.** Engines are told apart by slot colour, and a frontend
branch on which engine a node belongs to is a defect.

**No Python on the audio clock, and not as a narrow exception either** — the narrow exception is the
signal plane behind the bridge. Every real-time neural audio model has already left Python because
the plugin market forced it; Google's own Magenta RealTime 2 ships a C++ inference library beside its
JAX library. `nam-rs` and `tract` cover the Rust side. PEP 703's stop-the-world collector pauses only
ATTACHED threads, so a pure-Rust audio thread is safe beside the in-process Python tier.

**wasmtime was evaluated at length and rejected.** It was measured at parity with native for scalar
DSP and it turns a memory error into a recoverable `Err`. Against that one gain: a second toolchain
target, 13.5 MB, a `static mut`-lives-in-linear-memory trap that silently shares state between two
instances of one module, a 160 ns per node page-walk tax for private memories that `Pooling` cannot
fix, a store-wide epoch deadline, an instance leak with a 10,000 hard ceiling, and a trap handler
unproven on a `SCHED_FIFO` thread across three platforms. Recorded so it is not re-proposed.

**The crates are `backend/audio/goofi-audio-sdk` and `goofi-audio`; the shipped nodes are
`nodes_audio/`.** The SDK carries BOTH sides of the boundary — the trait, the `#[repr(C)]`
vtable, `export_node!` and `Loaded` — so they cannot drift, and depends on `goofi-node` alone;
the pipeline that builds and loads is the shared `goofi-build`. The engine depends on nothing
above `goofi-transport`, so no iceoryx2 thread or tokio reaches the DSP path and an external block
callback can drive it. New dependencies: `cpal`, `rtrb`, `midir`, `vst3`. No DSP crate in the
engine or the shipped nodes.

## Authored nodes

An agent or a user writes ONE file, `workspace/nodes_audio/<Name>.rs` — `impl AudioNode`, in safe
Rust — and goofi generates the crate around it, builds it, and loads it while audio runs. The
`.gfi` carries the SOURCE because the workspace does; the artifact is a machine-local cache.
`node-sources.md` holds the folder rule and the pipeline, which is one for every engine; what is
here is the part that is audio's — the rules an audio thread forces.

- **The manifest crosses as data, never as a Rust struct.** `describe()` answers the same JSON
  declaration the Python probe reads from a Python node's class attributes, and the engine leaks
  it to a `&'static NodeManifest` through the probe's own `leak_manifest` — both moved into
  `goofi-node`, because they were shared vocabulary all along. Only the vtable crosses as code.
- **A version symbol is checked before anything else.** The SDK stamps the goofi version it was
  built against; the loader reads it first and refuses a mismatch with a message naming both. A
  stale artifact is a refusal, never a crash — the objection to a home-grown ABI, answered.
- **The build is `goofi-build`, shared with the signal plane, and it runs outside the graph
  lock.** `library refresh` runs it BEFORE taking the lock, then locks for the scan, the diff and
  the restarts. The DSP allowlist it declares — `fundsp`, `biquad`, `realfft`, `rustfft`, `libm` —
  is what an audio node may import. A `cargo` that is absent makes an authored node UNAVAILABLE
  with "needs cargo to build"; a shipped node never needs it.
- **Reload is `library refresh`, and nothing watches a file.** It builds, scans, diffs stamps and
  restarts every live instance of a changed type — how an edited Python node reaches the canvas
  today. A build that fails leaves the stamp unchanged: the instances keep running the old
  artifact, and the palette row turns UNAVAILABLE with rustc's output as the reason.
- **Open with `RTLD_NOW`.** `libloading`'s unix default is `RTLD_LAZY`, so the first call into a
  fresh node runs the PLT resolver ON THE AUDIO THREAD. Windows snaps imports at load and has no
  such asymmetry.
- **A per-generation unique filename is mandatory, not hygiene.** Rebuilding to a fixed path and
  re-loading returned the identical vtable pointer and the OLD behaviour, even with a new inode.
  Make it the universal path and Windows' file lock stops being a special case.
- **Never `dlclose`.** Live vtables, `&'static` data, TLS destructors (rust-lang/rust#59629, open
  since 2019) and `atexit` registrations all pin a library, and `dlclose(3)` only promises to try.
  Measured: 338 kB of address space per reload for a small node, 435 kB for a `fundsp` reverb —
  about 3,100 and 2,400 edits before a gibibyte. Sweep at next start, as `/dev/shm/iox2_*` is.
- **`#![forbid(unsafe_code)]` in the node template, plus the allowlist.** In safe Rust an
  out-of-bounds slice index is a panic, not a segfault. The lint does NOT reach dependencies —
  `fundsp + biquad + realfft + rustfft + libm` resolves 87 crates with 16,886 `unsafe` tokens, and
  `RUSTFLAGS="-Funsafe_code"` is a silent no-op because cargo passes `--cap-lints allow`. So the
  allowlist is a policy, stated as one, not a lint that reaches.
- **`catch_unwind` in the SDK's shim, never in the author's code.** The vtable entry is
  `extern "C"`, and since Rust 1.81.0 an escaping panic is a guaranteed abort. Cost when nothing
  panics: +0.17%, inside noise. A panicking block costs 4.4 µs, or 34 µs with `RUST_BACKTRACE=1`.
  Policy: catch once, zero that node's output, drop it from the plan, republish, and surface
  `NodeFault::Process`. Never retry in place — a node that panics panics 750 times a second.
- **A watchdog, not a per-node budget.** Stamp `Instant::now()` at callback entry; each node skips if
  the deadline is already gone; N consecutive skips disables it. `Instant::now()` is a vDSO call at
  ~20 ns, so instrumenting fifty nodes costs 0.19% of a core. The "too expensive to measure" belief
  is folklore.

Measured edit-to-audible, wall power: **~126 ms** for a small node (125 ms of it `cargo build`) and
**~241 ms** for a `fundsp` reverb. Everything goofi itself does is under a millisecond and the
audio thread takes the new plan inside one block. Measured on the CLAP prototype; nothing in the
number is CLAP's.

## Measured, so it is not re-argued

- A C-ABI vtable call over a native trait call: +8.3 ns at 0 events, +17.9 at 4, ~2.15 ns marginal
  per event — 0.13% of a block for a hundred nodes. Measured on CLAP's ABI, and the shape — an
  `extern "C"` call with a value list — is the same, so the number carries. The ABI cost is not an
  argument in either direction.
- Wasm versus native: parity for recursive scalar DSP (1.04×), 2.09× for a full 1024-point spectral
  node, 3.22× for a bare FFT — of which ~85% is the structural 128-bit SIMD ceiling. In budget terms
  a spectral node is 0.28% native and 0.60% all-wasm.
- Per-edge buffer copies never become measurable: suppressing them never made a block faster at any
  chain length. The cost of separate linear memories is page-walk pressure, not copying.
- `PyExprEvaluator::eval`: 509 ns for one scalar variable (≈245 bindings per core at 8 kHz), ~4.6 µs
  fixed overhead for any array variable regardless of size, 77 µs for a 1 MiB frame. Measured with
  a one-off harness that was not kept; the numbers stand as recorded.

## CLAP: adopted 2026-08-25, replaced 2026-09-02

Recorded so the format is neither re-proposed for the builtins nor written off. The reversal and
its reasons are in the locked decisions; what survives of the two reviews that preceded the lock:

- **CLAP is alive and deliberately boring.** 1.2.10 tagged 2026-07-13, eight additive 1.2.x
  releases across 2024–2026, no 2.0 branch; stewarded by the same u-he/Bitwig/Surge people since
  2022. Hosted by Bitwig, Reaper, FL Studio and Studio One; not by Live, Logic, Cubase or Pro Tools.
  FabFilter, u-he and the indie world ship it, and every one of them ships VST3 beside it.
- **The hard part of hosting a plugin is the part goofi does not do.** Every public difficulty
  report is GUI, timers and platform event loops; none is the audio path. REAPER's four-year tail is
  61 changelog lines of plugin-specific compatibility, which no framework absorbs. Skipping the
  editor is what keeps a format adapter cheap — VST3 now, CLAP if ever.
- **`clack-host` was the only high-level Rust CLAP host, and a one-person project** absent from
  CLAP's own README, which names only `nice-plug` and `clap-sys`. A CLAP adapter, if one is ever
  built, is written over `clap-sys`, reading `clap-validator`'s `src/plugin/` — an MIT,
  spec-authoritative host under the free-audio org — and the shipped precedents price it at weeks:
  Qtractor in about a month, ossia in eight days, each one developer over the raw C ABI.
- **What the first review got wrong, in one line:** it priced a goofi contract as "a fourth arm
  only goofi would version", and that contract was already the arm behind the I/O nodes in the CLAP
  design. One trait with format adapters is one contract; CLAP for the builtins was two, plus a
  vendor extension that would have grown into the first.

## Open questions

- **The `codes` mutex.** `PyExprEvaluator::eval` takes a process-global `Mutex` on every evaluation,
  from every node thread. The benchmark above is single-threaded and cannot see the contention. It
  wants to be read-mostly — the map is written only on compile and release, and `arc-swap` is already
  a dependency. Settle it with a threaded measurement before any high expression rate is promised;
  a reference bypasses it entirely.
- **Three avoidable costs in the evaluator**, none yet fixed: the Rust locals map is rebuilt every
  eval (143 ns for four vars), `PyModule::import(py, "numpy")` runs on every array conversion, and
  `PyBytes::new` copies the whole array although `ArrayStore` is `Arc<[u8]>` and its own doc claims a
  numpy view can alias it zero-copy.
- **Nothing has been measured with a real device callback.** Every number here is a synthetic loop.
  The interaction of the node host, RT priority and the control thread's graph lock is unmeasured,
  and the graph lock is the most plausible cause of a real xrun — a knob drag is a burst of
  `node param edit` RPCs, each taking it. A `--debug` door reporting xruns and block timing is how
  it is measured, by hand, on each platform.
- **Windows latency.** cpal's WASAPI backend is shared-mode only and its own source says the callback
  period is always `GetDevicePeriod()` whatever is requested, so `BufferSize::Fixed` is a lie there
  and the floor is ~10 ms. `IAudioClient3` reaches 2.66 ms and cpal does not use it. ASIO needs the
  Steinberg SDK, which went GPLv3-or-proprietary in 2025, so it is not shippable in one binary.
- **macOS signing**, which costs nothing today and arrives with the first notarized release. Apple's
  documented answer for a process that loads foreign code is
  `com.apple.security.cs.disable-library-validation`; Ardour, Surge, VCV Rack, ossia score, Pure
  Data and BespokeSynth all ship it, and it covers an authored `cdylib` and a VST3 bundle alike. A
  locally compiled node is ad-hoc signed by the linker and carries no quarantine attribute, so it
  loads under that entitlement.
- **A control-rate param referencing an AUDIO output** receives a `[C, T]` frame, and the bare
  reference rule wants one element — so it errors where a follower or a mean is what was meant.
  Whether the rule takes the last sample, or a reference into the signal plane needs a node in
  between, is open.
- **Whether `MAX_CHANNELS = 16` is right**, and what a spectral port does to it when `Bins` arrives.
- **Drift between two devices** for `AudioIn`: measured before any correction is built.
- **A canvas affordance for references** — `param-sources.md` holds it.
- **A CLAP adapter**, deferred: one more implementor of the trait, only if a CLAP-only plugin ever
  matters. Nothing leans on it.
- **Whether goofi should run as a plugin inside a DAW.** Deliberately not recorded as an item: the
  cross-engine modulation that motivates this engine needs the signal plane, which a stripped plugin
  build would not have, so it may be a different product. The constraint that keeps the door open
  is kept: the audio crate depends on nothing above `goofi-transport`, so an external block
  callback can drive it — which is also what a test needs.
