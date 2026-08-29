# Audio engine

The second engine — a peer of the signal plane inside one graph. Designed with the user on
2026-08-24/25 against measurements, not instinct; every number below was taken on the target machine
and the harnesses are named where they still exist. Not built.

The seam that lets a second engine exist at all — the `Engine` trait, the settle point and what
the split deletes — is `multi-engine-graph.md`. What is here is audio.

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

**Every processing node is CLAP; audio I/O is the engine's own.** A goofi builtin, an agent-authored
node and a vendored plugin are the same kind of object, hosted through one path (`clack-host`). No
architectural split between goofi's own processing nodes and someone else's — goofi accepts CLAP's
parameter model rather than keeping a second one. Forking a shipped node is a copy rather than a
port, and `clap-wrapper` (CLAP→VST3/AUv2/AUv3/AAX/standalone; production-used by Six Sines and
Shortcircuit XT) exports a pure-DSP node to another DAW. The device and I/O nodes are NOT plugins —
a CLAP plugin never opens a device — and they are where goofi's full param model legitimately
survives, `refresh: true` included. The engine's internal dispatch is therefore
`Clap | Vst3 | Intrinsic`, recorded here so "one hosting path" is not re-litigated when the VST3
arm lands.

**VST3 hosting is a second subsystem, and no wrapper removes it.** No usable VST3→CLAP wrapper
exists (DISTRHO Ildaeil does not even bridge params), and the MIT-relicensed `vst3` crate (0.3,
after the SDK went MIT in October 2025) is raw unsafe COM bindings with only hobby hosts on top —
the COM lifecycle, bus/param/state plumbing and threading rules are ours to write. Build order: the
CLAP arm first and alone; the VST3 arm lands later behind the same dispatch, and `clap-wrapper`
carries goofi nodes OUT to VST3 DAWs in the meantime.

**One node, two halves.** An audio node is a `Kind::Leaf` in the one node map. The engine's own
main-thread side, behind the multi-engine trait, is the CONTROL half — params arrive as trait
propagation, health leaves through the drain queue, the `/data` scope tap is the `SlotFeed` ring
arm below — and the CLAP plugin is the DSP half. (This supersedes the earlier `NodeRuntime`
sentence: `NodeRuntime` is signal author machinery, signal-private after the split.) **The control half owns everything that touches the OS; the
kernel owns arithmetic.** A sample file, a MIDI port, a device name: all resolved on the control
half, which hands the kernel a buffer. This is CLAP's own `[main-thread]` / `[audio-thread]` split,
and `clack` encodes it in the type system — `PluginInstance` is `!Send`, `StartedPluginAudioProcessor`
is not. The OS-ownership claim is scoped to goofi-authored nodes: a vendored plugin loads its own
samples on its own main thread, and no host can stop it. For a goofi node the carrier is the vendor
extension below, because CLAP has no standard host→plugin resource channel.

**One goofi vendor extension carries what CLAP params cannot.** CLAP params are f64-only, so a
free-text value — a file path, a stream name — and a host-resolved resource handed over as a buffer
ride ONE vendor extension (CLAP sanctions these through `get_extension`). It is the surviving
remnant of a goofi ABI, deliberately confined to one extension: a node that uses it exports to
another DAW in degraded form, and a pure-DSP node exports whole.

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
`max()` over bins means nothing. The codomain differs by origin: a goofi-authored template declares
adaptable ports, so inference is free; a foreign plugin offers its `audio-ports-config` menu and the
function selects the best fit, with edge coercion bridging the remainder. A config change implies a
restart, which the crossfade-on-swap already covers.

**`nd()` evaluates in the host and lands as `clap_event_param_value` at control rate.** Measured:
2.15 ns per event marginal, so a thousand modulated params cost 0.33% of a block. Bindings evaluate
on ARRIVAL, so the eval rate is the source's rate and the audio thread only reads the latest value —
and for a cross-engine source, arrival is the pre-tick boundary drain, so the eval rate is capped
at the block rate.

**Audio-rate modulation is a CABLE, not a param.** Anything that must glide at audio rate takes a CV
input port and a slew node in front of it. This is the modular convention and it is why the engine
needs no host-imposed parameter ramp.

**An in-order signal→audio crossing is a BRIDGE node this engine owns.** A signal array that must
land as ordered samples — sonification, sample playback — enters through an audio-engine node with
a signal-dtype input slot, whose cross-engine edge derives a deeper subscriber buffer as part of
its service config (multi-engine-graph.md locks the convention). Modulation is not this: it
crosses as `nd()` at control rate, above.

**No host-side ramp in the first version.** CLAP has no per-frame slope; smoothing is the node's own.
`globals.default_param_fade` is therefore the value a node template reads, not a ramp the host
applies. Dense host ramping stays affordable if it is ever wanted — 64 events per param per block is
1.0% of a block for a hundred simultaneously gliding params.

**A node code swap crossfades; a block boundary is not enough.** 64 frames is 1.333 ms and there are
750 boundaries a second, so a step at one is still a step. ~20 ms, both instances running, and only
that node. Skip it when the node is silent, bypassed, newly added or being removed. Reinstantiate
per node, never per graph.

**`SlotType::Audio`.** Link legality (`out.kind != inp.kind`) then enforces plane homogeneity by
construction. NO new `Value` variant and NO new GOOF dtype tag — audio buffers never cross iceoryx2
or the wire, so the codec and its golden are untouched. The cost is `SlotType::name`/`from_name`,
`BOUNDARY_TYPES` 6 → 8 rows, the generated TS union, and one arm in `dtypeColor` plus a
`--dtype-audio` token.

**No plugin GUIs, and goofi draws every param itself.** goofi is a server that prints a URL and
never opens a window (principle 5), and its UI is a browser replica. CLAP's `gui` extension hands
the host a NATIVE window handle — x11, win32 or cocoa, embedded or floating — and the spec has no
offscreen or streamable form, so a plugin editor cannot reach the browser. Parameters are drawn from
`clap_plugin_params` the way every goofi node's params already are. **The cost is named, not hidden:
a plugin whose value IS its editor — a dynamic EQ curve, a wavetable display — is degraded to a
parameter list.** Deferred option, not v1: a companion editor process that opens a native window on
the machine the audio runs on, which is coherent because goofi is single-user and local by design.
This also skips `gui`, `timer` and `posix-fd`, which is where every reported hosting difficulty
lives.

**Viewing an audio slot widens `SlotFeed` to two arms** — an iceoryx2 subscriber, or an in-process
ring the audio runtime fills. Everything downstream is already payload-free, and the `line` viewer
already asks for envelope reduction, so the frontend cost is zero.

**Opaque per-node state lives in `workspace/.goofi/state/<uid_hex>/`.** That one choice inherits the
archive, the dirty fingerprint, the atomic load swap, and undo of a delete — `capture_subtree_restore`
already restores a node at the same uid, so the directory was never removed. Bytes never cross a
channel and a large sampler never enters the document. `restart_node` is the destroyer, so the flush
goes inside it, before `spawn_host`, covering both call sites. Plugins serialize their param values
INTO their `clap.state` blob, so a load has two authorities for one value; the rule is: blob first,
then goofi's param record flushed on top as param-value events. The `.gfi` record is authoritative
for params, the blob for everything else.

**The `.gfi` records nothing per node about the engine** — it is a property of the type, and a copy
in the archive is a mirror. The format change is: delete `pillar_default`, add `goofi: "<version>"`,
stay at version 1. Read `goofi:` before the version gate so a refusal can name the writer.

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

An agent or a user writes a Rust CLAP plugin, goofi compiles it to a `cdylib` and loads it while
audio runs. See `node-sources.md` for how it is discovered; the loading rules are here because they
are audio's.

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
- **`catch_unwind` inside the plugin.** CLAP's `process` is `extern "C"`, and since Rust 1.81.0 an
  escaping panic is a guaranteed abort. Cost when nothing panics: +0.17%, inside noise. A panicking
  block costs 4.4 µs, or 34 µs with `RUST_BACKTRACE=1`. Policy: catch once, zero that node's output,
  drop it from the plan, republish, and surface `NodeFault::Process`. Never retry in place — a node
  that panics panics 750 times a second.
- **A watchdog, not a per-node budget.** Stamp `Instant::now()` at callback entry; each node skips if
  the deadline is already gone; N consecutive skips disables it. `Instant::now()` is a vDSO call at
  ~20 ns, so instrumenting fifty nodes costs 0.19% of a core. The "too expensive to measure" belief
  is folklore.

Measured edit-to-audible, wall power: **~126 ms** for a small node (125 ms of it `cargo build`) and
**~241 ms** for a `fundsp` reverb, then 20 ms of crossfade. Everything goofi itself does is under a
millisecond and the audio thread takes the new plan inside one block.

## Measured, so it is not re-argued

- CLAP's C ABI over a native trait call: +8.3 ns at 0 events, +17.9 at 4, ~2.15 ns marginal per event
  — 0.13% of a block for a hundred nodes. The ABI cost is not an argument in either direction.
- Wasm versus native: parity for recursive scalar DSP (1.04×), 2.09× for a full 1024-point spectral
  node, 3.22× for a bare FFT — of which ~85% is the structural 128-bit SIMD ceiling. In budget terms
  a spectral node is 0.28% native and 0.60% all-wasm.
- Per-edge buffer copies never become measurable: suppressing them never made a block faster at any
  chain length. The cost of separate linear memories is page-walk pressure, not copying.
- `PyExprEvaluator::eval`: 509 ns for one scalar variable (≈245 bindings per core at 8 kHz), ~4.6 µs
  fixed overhead for any array variable regardless of size, 77 µs for a 1 MiB frame. Measured with
  a one-off harness that was not kept; the numbers stand as recorded.

## Reviewed, so it is not re-argued (2026-08-25)

Two independent reviews before commitment: an adversarial pass against the code, and an ecosystem
survey from primary sources. Both landed on keep.

- **CLAP is alive and deliberately boring.** 1.2.10 tagged 2026-07-13, eight additive 1.2.x
  releases across 2024–2026, no 2.0 branch; stewarded by the same u-he/Bitwig/Surge people since
  2022, and the whole free-audio org (wrapper, validator, helpers) pushed within the last month.
- **Adoption is real and plateaued below the top tier.** Bitwig, Reaper, FL Studio and Studio One
  host it; Live, Logic, Cubase, Pro Tools, Ardour, LMMS and Renoise do not. FabFilter's whole
  catalog, u-he and the indie world ship CLAP builds; Arturia, NI, iZotope and Xfer do not. goofi
  hosts plugins rather than selling one, so the hosts that matter are goofi itself and the export
  path — and `clap-wrapper` is active and production-used.
- **Internal-devices-as-plugins has decades of precedent.** REAPER's stock FX are VSTs, Ardour's
  internal processors are LV2s in its own tree, and Six Sines / Shortcircuit XT are clap-first
  products whose VST3/AU builds are clap-wrapper around their own CLAP. No project was found that
  adopted the pattern and reversed it.
- **`clack-host` is the only high-level Rust CLAP host, and the two surveys disagreed about it.**
  One recommended it, pinned to the v0.2 git rev; the other recommended writing the layer over
  `clap-sys` instead. The evidence for writing it: every host that shipped in CLAP's first eighteen
  months wrote its own layer, none has replaced it, and CLAP's own author said in 2023 that "the
  host'll be harder to generalize/glue than the plugin". Qtractor went from standing start to
  shipped host in about a month, one developer, raw C ABI; ossia took 8 days. `clack-host` is by its
  own README a thin safe wrapper over `clap-sys`, not a framework — so the real choice is a thin
  wrapper with a small user base against the same `clap-sys` underneath it.
  **Decision: adopt `clack-host`, behind goofi's own `Clap | Vst3 | Intrinsic` dispatch.** It
  encodes the main-thread/audio-thread split in the type system, which is principle 3 applied to the
  bug class that is hardest to find in a host, and it carries Miri and clap-validator in CI. The
  dispatch trait is what makes this reversible: clack sits behind one arm and touches neither the
  plan compiler, the Kahn sort nor the param bridge.
- **The exit path is written down, and it is not "fork clack".** It is: re-implement over
  `clap-sys`, reading `clap-validator`'s `src/plugin/` — an MIT, spec-authoritative host under the
  free-audio org itself — and `clap-wrapper`'s standalone `clap_proxy.cpp`. Estimated two to three
  weeks headless, from three independent shipped precedents. Binding staleness has an answer too:
  published `clap-sys` 0.5.0 tracks CLAP 1.2.2, and `Quant1um/clap-sys` already tracks 1.2.10.
  Recorded because clack is absent from the official CLAP README, which names only `nice-plug` and
  `clap-sys` — no institutional standing, whatever its technical merit.
- **The hard part of CLAP hosting is the part goofi does not do.** Every public difficulty report is
  GUI, timers and platform event loops; none is the audio path. REAPER's four-year tail is 61
  changelog lines of plugin-specific compatibility, which no framework absorbs. Skipping the GUI is
  what makes both the build and the exit path cheap.
- **The hybrid (goofi-native ABI + plugin adapters) was taken seriously and rejected.** The internal
  trait exists in BOTH designs — VST3 and the intrinsic I/O nodes guarantee it — so the native ABI
  is the same trait with a fourth arm, one only goofi would version, document and teach its agents,
  while forfeiting fork-is-a-copy and export. All-CLAP is strictly one fewer ABI, and the
  adapter-inside cost is the measured 2.15 ns/event noise.

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
  The interaction of the plugin host, RT priority and the control thread's graph lock is unmeasured,
  and the graph lock is the most plausible cause of a real xrun — a knob drag is a burst of
  `node param edit` RPCs, each taking it.
- **Windows latency.** cpal's WASAPI backend is shared-mode only and its own source says the callback
  period is always `GetDevicePeriod()` whatever is requested, so `BufferSize::Fixed` is a lie there
  and the floor is ~10 ms. `IAudioClient3` reaches 2.66 ms and cpal does not use it. ASIO needs the
  Steinberg SDK, which went GPLv3-or-proprietary in 2025, so it is not shippable in one binary.
- **macOS signing**, which costs nothing today and arrives with the first notarized release. Apple's
  documented answer for a plugin host is `com.apple.security.cs.disable-library-validation`; Ardour,
  Surge, VCV Rack, ossia score, Pure Data and BespokeSynth all ship it. A locally compiled node is
  ad-hoc signed by the linker and carries no quarantine attribute, so it loads under that entitlement.
- **MIDI** has not been designed. Its natural shape is `midir` on the control half feeding a ring the
  kernel reads as timestamped events, which is CLAP's event model already.
- **Whether goofi should run as a plugin inside a DAW.** Deliberately not recorded as an item: the
  cross-engine modulation that motivates this engine needs the signal plane, which a stripped plugin
  build would not have, so it may be a different product. One constraint is kept — the audio crate
  depends on `goofi-core`, `goofi-node` and `goofi-transport` and nothing above them (none carries
  iceoryx2 threads or tokio into the DSP path), so it can be driven by an external block callback,
  which is also what a test needs.
