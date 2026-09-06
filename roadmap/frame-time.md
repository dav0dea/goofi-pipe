# Frame time: one clock, and a tick on every frame

Two nodes' frames cannot be aligned today. `Meta` carries `sfreq`, `ufreq`, `index`, `channels` and
`reduced`; nothing says WHEN a frame was produced, and `goofi-transport` carries no time at all.
`recording.md` is what needs it.

## Decisions

**One clock, and it already exists.** `Graph::patch_start()` is handed to every engine through
`Engine::reset_clock`, and `NodeCtx::now` is computed from it. No second clock is minted anywhere.

**A frame carries its node's PROCESS TICK, not a lineage.** The patch time at which this node ran.
Nothing is inherited from an input, so a node with four triggering inputs is not a special case,
and there is no join rule, no span and no per-frame provenance to keep in step.

**Sample times are DERIVED, never stored.** `sfreq` and the shape are already in `Meta`, so sample 0
of a 256-sample window at 256 Hz emitted at t sits at t − 1. A `Buffer` is correct for free. Where a
derived stream's provenance matters, the answer is to record its source too and measure the offset
from the two recordings.

**The stamp is the RUNTIME's, never the node's.** `stamp_meta` already runs after `process()`, reads
the triggering inputs, and is documented as the one stamping site. A node author writes nothing and
cannot break alignment — which is also what keeps the subprocess tier correct, since its `Instant`
has a different origin and it must never mint one.

**A rate-locked stream derives its timeline from the SAMPLE COUNT.** For audio the count IS the
clock and it is exact; a clock read per block adds scheduling jitter to a timeline that had none.
Anchor the counter to patch time once at stream start and derive every later block. A self-paced
signal node has no anchor, so its tick read is its timeline. A recording records WHICH of the two a
stream used — a derived timeline and a measured one are not the same evidence.

**Wall time is recorded ONCE**, as the UTC time of `patch_start()`, in the recording's manifest.
Never per frame: an NTP step is then a header question rather than a per-frame lie.

## Open

- Whether the tick is a typed field on `Data` beside `index`, or a `Meta` key.
- What a node that KNOWS its device acquisition time declares — LSL carries one. It is the only
  case where a node body would touch time, and it wants a typed field rather than a free key.
- `stamp_meta` inherits `index` today by matching frame counts, and falls back to a fresh counter
  when two triggering inputs are the same length. The tick makes that ambiguity visible; whether
  `index` keeps the heuristic is a separate question.
