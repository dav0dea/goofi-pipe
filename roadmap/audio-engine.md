# Audio engine

The second engine — a peer of the signal plane inside one graph. Designed with the user on
2026-08-24/25 against measurements, not instinct; every number below was taken on the target machine
and the harnesses are named where they still exist. Revisited with the user on 2026-09-02: the node
contract is goofi's own, CLAP is out, and a reload no longer crossfades. Not built.

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
shipped node implements `AudioNode` and links in statically. An authored node implements the same
trait, is built against the SDK crate at goofi's own version, and loads as a `cdylib` through a
`#[repr(C)]` vtable the SDK emits — the author never sees it. A VST3 plugin is an adapter that
implements the same trait. The plan compiler, the Kahn sort, the arena, the watchdog and the viewer
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

**VST3 hosting is an adapter, and it is built after the goofi nodes.** The MIT-relicensed `vst3`
crate (0.3, after the SDK went MIT in October 2025) is raw unsafe COM bindings with only hobby
hosts on top — the COM lifecycle, bus/param/state plumbing and threading rules are ours to write.
Its params project onto goofi's `Param` for the document; its bus arrangements are the menu the
channel inference selects from. No usable VST3→CLAP wrapper exists (DISTRHO Ildaeil does not even
bridge params), which is one more reason the centre is a trait and not a format.

**One node, two halves.** An audio node is a `Kind::Leaf` in the one node map. The engine's own
main-thread side, behind the multi-engine trait, is the CONTROL half — params arrive as trait
propagation, health leaves through the drain queue, the `/data` scope tap is the `SlotFeed` ring
arm below — and the node's `process` is the DSP half, which owns nothing but arithmetic. **The
control half owns everything that touches the OS.** A sample file, a MIDI port, a device name: all
resolved on the control half, which hands the DSP half a buffer through the trait. The DSP object
is `Send` and moves to the audio thread inside the plan; nothing on the audio thread allocates,
locks or blocks. The OS-ownership claim is scoped to goofi-authored nodes: a VST3 plugin loads its
own samples on its own main thread, and no host can stop it.

**The audio device belongs to the engine, not to a node.** The device callback is the clock. An
input or output node is an arena region the engine fills or drains.

**The plan crosses to the audio thread as an owned value over an `rtrb` SPSC ring**, with a return
lane that bounces the retired plan back to a non-RT thread to drop. NOT `ArcSwap`: a DSP plan owns
per-node mutable state, `TypedFunc`-style calls need `&mut`, and the attempt does not compile
(E0596). `ArcSwap` stays for globals and params, which are read-only projections.

**Topological order, Kahn, ties broken by `Uid`.** Not iteration order — `Graph::nodes` is an
`IndexMap` whose order moves across save, load, undo and paste, and uids are restored verbatim by a
load. Cycles fall out of the sort: if `order.len()` is short, the remainder is the cycle, and those
nodes are excluded and named through the existing per-node error channel. No second DFS, no cycle
check in `add_link`.

**Channel counts are inferred, never configured.** One pure function per node type,
`channels(ins) -> per-output counts`, defaulting to `max(ins).max(1)`, evaluated once per node
INSIDE the Kahn loop — inputs are settled before a node is visited, so there is no fixed point and
no iteration. Coercion lives on the edge: `Same`, `Broadcast(1→n)`, `PadOrTruncate`. Mono to stereo
is one arena region used twice, and it is free. A layout tag rides beside the count (`Discrete`,
`Speakers(mask)`, `Bins(n)`) — 512 spectral bins are one `Bins(512)` port, not 512 channels, and
`max()` over bins means nothing. The codomain differs by origin: a goofi node answers
`channels(ins)` itself, so inference is free; a VST3 plugin offers its bus arrangements and the
adapter selects the best fit, with edge coercion bridging the remainder. An arrangement change
implies a restart of that node.

**`nd()` evaluates in the host and lands as a param value at the block boundary, at control rate.**
Measured: 2.15 ns per value marginal, so a thousand modulated params cost 0.33% of a block.
Bindings evaluate on ARRIVAL, so the eval rate is the source's rate and the audio thread only reads
the latest value — and for a cross-engine source, arrival is the pre-tick boundary drain, so the
eval rate is capped at the block rate.

**Audio-rate modulation is a CABLE, not a param.** Anything that must glide at audio rate takes a CV
input port and a slew node in front of it. This is the modular convention and it is why the engine
needs no host-imposed parameter ramp.

**An in-order signal→audio crossing is a BRIDGE node this engine owns.** A signal array that must
land as ordered samples — sonification, sample playback — enters through an audio-engine node with
a signal-dtype input slot, whose cross-engine edge derives a deeper subscriber buffer as part of
its service config (multi-engine-graph.md locks the convention). Modulation is not this: it
crosses as `nd()` at control rate, above.

**No host-side ramp in the first version.** Smoothing is the node's own. `globals.default_param_fade`
is therefore the value a node reads, not a ramp the host applies. Dense host ramping stays
affordable if it is ever wanted — 64 param values per block is 1.0% of a block for a hundred
simultaneously gliding params.

**A node reload is a discrete event, and it does not crossfade.** A reload replaces one node's DSP
instance in the plan at a block boundary, and a click or a short gap there is accepted: a reload is
an act of authoring, and this engine is not a DAW. Reinstantiate per node, never per graph. This
deletes the earlier ~20 ms two-instance crossfade and everything that gated it.

**`SlotType::Audio`.** Link legality (`out.kind != inp.kind`) then enforces plane homogeneity by
construction. NO new `Value` variant and NO new GOOF dtype tag — audio buffers never cross iceoryx2
or the wire, so the codec and its golden are untouched. The cost is `SlotType::name`/`from_name`,
`BOUNDARY_TYPES` 6 → 8 rows, the generated TS union, and one arm in `dtypeColor` plus a
`--dtype-audio` token.

**No plugin GUIs, and goofi draws every param itself.** goofi is a server that prints a URL and
never opens a window (principle 5), and its UI is a browser replica. A VST3 editor hands the host a
NATIVE window handle, with no offscreen or streamable form, so a plugin editor cannot reach the
browser. Parameters are drawn from the plugin's parameter list the way every goofi node's params
already are. **The cost is named, not hidden: a plugin whose value IS its editor — a dynamic EQ
curve, a wavetable display — is degraded to a parameter list.** Deferred option, not v1: a
companion editor process that opens a native window on the machine the audio runs on, which is
coherent because goofi is single-user and local by design. Skipping the editor also skips the
platform event loop and timer plumbing, which is where every reported hosting difficulty lives.

**Viewing an audio slot widens `SlotFeed` to two arms** — an iceoryx2 subscriber, or an in-process
ring the audio runtime fills. Everything downstream is already payload-free, and the `line` viewer
already asks for envelope reduction, so the frontend cost is zero.

**Opaque per-node state lives in `workspace/.goofi/state/<uid_hex>/`.** That one choice inherits the
archive, the dirty fingerprint, the atomic load swap, and undo of a delete — `capture_subtree_restore`
already restores a node at the same uid, so the directory was never removed. Bytes never cross a
channel and a large sampler never enters the document. `restart_node` is the destroyer, so the flush
goes inside it, before the rebirth, covering both call sites. For a goofi node the state blob never
carries a param value — the trait keeps them apart — so the `.gfi` param record is the one authority
by construction. A VST3 plugin serializes its params INTO its state, so its load has two
authorities for one value; the rule is: blob first, then goofi's param record flushed on top as
param values. The `.gfi` record is authoritative for params, the blob for everything else.

**The `.gfi` records nothing per node about the engine** — LANDED with multi-engine-graph.md's
step 4: `pillar_default` is gone, and the manifest carries `goofi: "<version>"`, read before the
version gate so a refusal can name the writer.

**Device selection is an ordinary param** — `Param::Str { options, refresh: true }` carrying cpal's
`DeviceId`, empty meaning the host default. A named device that is absent fails `setup()` into a red
node. No machine-local sidecar: a second file with a second lifetime breaks "the `.gfi` is the patch".

**One node editor panel for every engine.** This REVERSES the earlier decision in this file, which
said each engine gets its own editor. Engines are told apart by slot colour, and a frontend branch on
which engine a node belongs to is a defect.

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

## Authored nodes

An agent or a user writes a Rust node against the SDK crate — `impl AudioNode`, in safe Rust —
goofi compiles it to a `cdylib` and loads it while audio runs. See `node-sources.md` for how it is
discovered; the loading rules are here because they are audio's.

- **The manifest crosses as data, never as a Rust struct.** The `cdylib` answers with the same
  declaration the Python probe reads from a Python node's class attributes, and the engine leaks it
  to a `&'static NodeManifest` the way the probe does — one declaration schema for every node
  language. Only the `#[repr(C)]` vtable crosses as code.
- **A version symbol is checked before anything else.** The SDK stamps the goofi version it was
  built against; the loader reads it first and refuses a mismatch with a message naming both. A
  stale artifact is a refusal, never a crash — the objection to a home-grown ABI, answered.
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
- **`#![forbid(unsafe_code)]` in the node template, plus an allowlist of vetted DSP dependencies.**
  In safe Rust an out-of-bounds slice index is a panic, not a segfault. The lint does NOT reach
  dependencies — `fundsp + biquad + realfft + rustfft + libm` resolves 87 crates with 16,886 `unsafe`
  tokens, and `RUSTFLAGS="-Funsafe_code"` is a silent no-op because cargo passes `--cap-lints allow`.
  So the allowlist is a policy, stated as one, not a lint that reaches.
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
  a dependency. Settle it with a threaded measurement before any high binding rate is promised.
- **Three avoidable costs in the evaluator**, none yet fixed: the Rust locals map is rebuilt every
  eval (143 ns for four vars), `PyModule::import(py, "numpy")` runs on every array conversion, and
  `PyBytes::new` copies the whole array although `ArrayStore` is `Arc<[u8]>` and its own doc claims a
  numpy view can alias it zero-copy.
- **Nothing has been measured with a real device callback.** Every number here is a synthetic loop.
  The interaction of the node host, RT priority and the control thread's graph lock is unmeasured,
  and the graph lock is the most plausible cause of a real xrun — a knob drag is a burst of
  `node param edit` RPCs, each taking it.
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
- **MIDI** has not been designed. Its natural shape is `midir` on the control half feeding a ring the
  DSP half reads as timestamped events at the block boundary — the shape params already cross in.
- **A CLAP adapter**, deferred: one more implementor of the trait, only if a CLAP-only plugin ever
  matters. Not a v1 item, and nothing in v1 leans on it.
- **Whether goofi should run as a plugin inside a DAW.** Deliberately not recorded as an item: the
  cross-engine modulation that motivates this engine needs the signal plane, which a stripped plugin
  build would not have, so it may be a different product. One constraint is kept — the audio crate
  depends on `goofi-core`, `goofi-node` and `goofi-transport` and nothing above them (none carries
  iceoryx2 threads or tokio into the DSP path), so it can be driven by an external block callback,
  which is also what a test needs.
