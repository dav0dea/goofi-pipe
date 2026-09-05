# Built-in node organization

The node library was reset to a **tabula rasa** so that growing it is a design exercise rather than
an act of archaeology. Growing it is the next major project, and it is to be **co-designed with the
user**, not chosen unilaterally.

The node library program's step 1 — the seam: `engine:Name` type ids, tags, pulse params, page
order, multi-slot senders — landed 2026-09-05. The library itself is rewritten in the program's
later steps; `library.md` holds the set.

## What ships today

Fourteen nodes, and each one was added to prove a seam rather than to fill a category:

- **Signal, Rust** — `Oscillator`, `Buffer`, `Filter`, `Psd`: `.rs` source in `nodes_signal/`, built
  at goofi's build time and loaded dynamically since 2026-09-02 (`node-sources.md`).
- **Audio, Rust** — `Osc`, `Gain`, `Env`, `Svf`, `Slew`, `Feedback`, `SignalIn`: the same one-file
  shape in `nodes_audio/`, against the audio SDK (`audio-engine.md`). They are a VOCABULARY rather
  than a category sweep — a source, a level, a shape, a filter, a rate limit, a cycle and the door
  from the signal plane — and the §13 set that would grow them is still the user's to choose.
- **Audio, native** — `AudioOut`, `AudioIn`, `MidiIn`: the engine's own rather than files, because
  a device and a port are the engine's to own.
- **Python** — none in the shipped tree (`nodes_signal/`). `LempelZiv`, `PermutationEntropy`, `SpectralEntropy` and
  `DetrendedFluctuation` were the four, all complexity measures over `[C, T]` that existed
  because the subprocess tier had to be proved against real packages that hold the GIL; they now
  live in `node-bundles/complexity/` (see `library.md`).

`Filter` is the first evidence that the rule below pays: it is ONE node with a `mode` param of
four options, where the old implementation had a node for each. The audio `Svf` is the second, and
it took the name before two engines could offer one; `audio:Filter` is legal now.

## The rule the reset exists to protect

Every node the old Python implementation had was there because someone once needed it, and the
result was hundreds of overlapping single-purpose nodes. The replacement is an **orthogonal**
library: a small set of nodes that compose, rather than a large set that each do one job.

Before adding a node, the question is not "would this be useful?" but "what does the library
already compose to, and is this genuinely outside that span?"

## What the library needs, by area

- **Sinks.** Nothing currently leaves the patch. Recording to disk, streaming out, and a plain
  "write this to a file" are all missing. This is the largest hole.
- **Real biosignal inputs.** LSL, OSC, serial, and whatever devices the user actually runs. These
  are the canonical shape of a node with a background receiver thread started in `setup()` — the
  subprocess tier exists for exactly this.
- **Array maths.** The general-purpose middle of any patch: reshape, slice, reduce, arithmetic.
  Where the temptation to add fifty nodes is strongest and most wrong.
- **Recording and playback.** Both directions of the same idea, and the thing that makes a patch
  reproducible.
- **More spectral work**, now that `Psd` and `Filter` stand: envelopes, coherence, time-frequency.
  Each is a test of whether the span already covers it.

## The shipped nodes, made more expressive

Each node that ships proves a seam, and its params are the minimum that proved it. Two are already
known to be short, and each is a param design rather than a new node:

- **`Buffer` sizes in samples only.** A window a user thinks of as "two seconds" or "the last
  thirty updates" is a size to recompute by hand at every rate change. The size wants a unit —
  samples, seconds against the frame's `sfreq`, or updates against the node's `ufreq` — with the
  count derived, so a rate change moves the window and not the patch.
- **`Psd` exposes a window taper and nothing else.** Averaging over sub-windows (Welch), the
  resolution, a frequency range to keep, and a log or linear scale are the choices every reading
  of a spectrum makes, and each one is made downstream today or not at all.

## Authoring constraints that shape the library

- A node's tier is NOT selectable — one probe per file routes it, and the routing is by whether its
  imports keep the free-threaded GIL disabled.
- A Python node declares itself in class attributes (`INPUTS`, `OUTPUTS`, `PARAMS`), at parity with
  the Rust manifest. There are no `config_*` functions.
- A node that cannot load must explain itself in the palette rather than vanish.
- Params persist one-to-one with no migration, ever, so a param name is a permanent decision.
- The universal `common` group is injected at one site, LAST; a node that declares a `common` param
  has said what it means by it and nothing overwrites that.
- Param pages show in the manifest's declared order, `common` last: the record's insertion order
  is the wire's, and the client sorts nothing (2026-09-05).
- A node declares `tags` from goofi's closed vocabulary — `goofi_node::Tag` in Rust, `TAGS = [...]`
  in Python — and `category` is gone; the engine and the bundle are facets the palette derives,
  never tags (2026-09-05).
- A `multi` input slot hands the node every frame with the `node.slot` that sent it, in wire
  order, and a rename of a sender reaches every consumer (2026-09-05).

## Open questions

- Where the line falls between a built-in node and a bundle node (see `library.md`).
- ANSWERED 2026-09-02: a shipped node is SOURCE in either language, in `nodes_signal/`, and a
  user's edit is a copy of that one file into the patch's own folder, which shadows it. "Rust-first
  or Python-first" is then a per-node choice with no structural cost either way.
- What decides that a node is written in Rust. Today it is "the four that had to be fast", which is
  a history rather than a rule. A user can now write one as easily as a Python node, and it pays
  one codec copy per run at the boundary (`node-sources.md`), so the answer is a measurement:
  what a Rust node buys per frame, through that boundary, against the same node in Python.
