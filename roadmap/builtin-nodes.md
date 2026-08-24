# Built-in node organization

The node library was reset to a **tabula rasa** so that growing it is a design exercise rather than
an act of archaeology. Growing it is the next major project, and it is to be **co-designed with the
user**, not chosen unilaterally.

## What ships today

Eight nodes, and each one was added to prove a seam rather than to fill a category:

- **Rust** — `Oscillator`, `Buffer`, `Filter`, `Psd`.
- **Python** — `LempelZiv`, `PermutationEntropy`, `SpectralEntropy`, `DetrendedFluctuation`, in
  `nodes/`. All four are complexity measures over `[C, T]`, and all four exist because the
  subprocess tier had to be proved against real packages that hold the GIL.

`Filter` is the first evidence that the rule below pays: it is ONE node with a `mode` param of
four options, where the old implementation had a node for each.

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

## Authoring constraints that shape the library

- A node's tier is NOT selectable — one probe per file routes it, and the routing is by whether its
  imports keep the free-threaded GIL disabled.
- A Python node declares itself in class attributes (`INPUTS`, `OUTPUTS`, `PARAMS`), at parity with
  the Rust manifest. There are no `config_*` functions.
- A node that cannot load must explain itself in the palette rather than vanish.
- Params persist one-to-one with no migration, ever, so a param name is a permanent decision.
- The universal `common` group is injected at one site; a node that declares a `common` param has
  said what it means by it and nothing overwrites that.

## Open questions

- Where the line falls between a built-in node and a marketplace node (see `node-marketplace.md`).
- Whether the shipped library is Rust-first (fast, compiled in) or Python-first (editable, the
  thing a user can fork) — and how a user's edit to a shipped node is meant to work.
- What decides that a node is written in Rust. Today it is "the four that had to be fast", which is
  a history rather than a rule.
