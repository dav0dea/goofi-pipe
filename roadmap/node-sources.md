# Node sources: on-the-fly Rust, and one place nodes come from

Two halves of one problem. A user can write a Python node and see it in the palette; a Rust node
needs a rebuild of goofi. And the directories nodes come from grew one at a time, so there is no
central node source directory to point either at.

## The state today

- **Rust nodes are compiled in**, registered through `inventory`. There is no directory — a Rust
  node is a file in `goofi-nodes` and a `cargo build`.
- **The shipped Python tree is `nodes/`, resolved RELATIVE to the process's working directory.** A
  binary started anywhere else finds no shipped Python node and says nothing about it.
- `--extra-nodes DIR` adds directories to the scan, a later one winning a shared type name.
- **A patch's own `workspace/nodes/` is scanned LAST**, so it shadows every shipped tree. This part
  is right, and it is what makes a `.gfi` carry its own nodes.

**Decided 2026-09-02, with the audio engine: a node source ROOT holds one folder per engine, named
`nodes_<engine id>`.** The shipped `nodes/` becomes `nodes_signal/` beside a new `nodes_audio/`; a
bundle and a patch's `workspace/` hold the same pair; the rescan hands each engine its own folder
under each root. The folder decides the engine, never the file extension. No alias for `nodes/`.

## The restructure

One node source directory, resolved absolutely, that both languages and all three origins agree on:
shipped, installed, and this patch's own. The precedence order above is already correct — what is
missing is a single place that states it, and a shipped tree that does not depend on where the
binary was started.

This wants the config folder (see `config-folder.md`), which is where an installed node package
would land.

## The audio plane answers this differently, and that is decided

The audio engine (see `audio-engine.md`) locks its own answer: **every audio node implements one
goofi trait. A shipped node links in statically; an authored node is the same trait built against
goofi's SDK crate, compiled to a `cdylib`, loaded with `libloading`, and reloaded while the audio
thread runs.** Audio I/O is the engine's own, and a VST3 plugin is an adapter behind the same
trait. A toolchain is needed to AUTHOR a node and never to run one; the shipped nodes are built
with goofi.

**Why the two planes diverge, stated once.** The objection below — a `cdylib` means a versioned ABI,
and a mismatch is a crash rather than an error — is answered on the audio plane by who builds the
node: goofi does, against its own SDK at its own version, and the SDK stamps that version into a
symbol the loader checks before anything else, so a mismatch is a refusal with a message. And the
risk the boundary carries is bounded by what a DSP kernel IS: it is handed a buffer and returns a
buffer, so `#![forbid(unsafe_code)]` plus a dependency allowlist is a real envelope. A signal node is not that shape — it legitimately opens sockets and serial ports, starts
background threads and blocks for a long time — so a process boundary earns its keep there and does
not in a 64-frame block.

The measurements that settled it, on the working prototype: 126 ms from a saved edit to audible new
code, and a profile-matched incremental rebuild of 0.16–0.18 s against 0.35–0.42 s for the same node
built as plain Rust. Build time decides nothing at that scale.

The loading rules are in `audio-engine.md` and are not repeated here.

## On-the-fly Rust nodes, on the SIGNAL plane

**Decided 2026-09-02: a dynamic library, through the audio plane's pipeline, not a process.** The
owner's direction is that the built-in signal Rust nodes eventually move into `nodes_signal/` and
become authorable and dynamically loadable on the fly, exactly as audio nodes are. The audio
build pipeline takes an SDK path and a source file and knows nothing of audio, so the signal
plane's half is a signal SDK crate — the `Node` trait behind a `#[repr(C)]` vtable with the same
version symbol and the same `describe()` — and an adoption of that pipeline, not a design. The
process candidate below is recorded as what it was and is no longer pursued.

To be investigated, in this order:

1. **Whether the existing node contract carries a Rust node unchanged.** If it does, this item is
   mostly a compile pipeline and not an engine change at all.
2. **The compile pipeline.** Where the crate is generated, what it depends on, where the artifact is
   cached, and what invalidates the cache. **The audio plane will have built one**, so this is
   adoption before it is design.
3. **`cdylib` + `libloading` as the alternative**, measured against the process for latency. It only
   wins if per-frame process overhead turns out to matter, and the shared-memory transport is
   already what carries frames — so it probably does not.

## Constraints this has to answer

- **A Rust toolchain is not a goofi dependency.** Setup is `cargo run -p goofi-init` and `cargo run`,
  and a user who installs a binary has neither. A node that needs `cargo` must degrade to
  "unavailable, and here is why" on a machine without it — the same way a Python node with a missing
  import already does. On the audio plane this is narrower: only AUTHORING is absent, and every
  shipped and vendored node still loads.
- **Compile latency is seconds to minutes**, where a Python node is instant. The node lifecycle
  already models "not ready yet", so the stage machinery exists; the UX of a node that is compiling
  does not.
- **The version lives in one place.** A compiled node artifact is pinned to the goofi version that
  built it, and a stale artifact must be detected rather than loaded.
- Whether an on-the-fly Rust node can be EDITED in the app the way a Python node can, or whether it
  is a build product a user brings. **The audio plane says edited**, and makes it first-class.

## Open questions

- ANSWERED for audio, and the signal plane inherits it: a `.gfi` carries SOURCE — one `.rs` per
  node in the workspace — and the built artifact is a machine-local cache under `.goofi/build/`,
  keyed by goofi version and source hash.
- What a Rust node buys that a Python node does not, stated in measurements rather than instinct.
  The four shipped Rust nodes are a history, not a rule (see `builtin-nodes.md`).
