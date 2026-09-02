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

**A process is the leading candidate, not a dynamic library.** Runtime-linking a Rust crate means a
stable ABI across a `cdylib` boundary — versioned, and a mismatch is a crash rather than an error.
A process has neither problem, and the seam already exists: the subprocess Python tier marshals a
node over a process boundary today, under the same contract the in-process tier uses. A
compiled-out-of-tree Rust node is that contract with a different implementation behind it.

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

- Does a patch's `.gfi` carry Rust node SOURCE, a built artifact, or neither? Source is portable and
  slow; an artifact is fast and machine-specific. Today a `.gfi` carries source, because Python has
  no other form. The audio plane needs the same answer and does not yet have it.
- What a Rust node buys that a Python node does not, stated in measurements rather than instinct.
  The four shipped Rust nodes are a history, not a rule (see `builtin-nodes.md`).
