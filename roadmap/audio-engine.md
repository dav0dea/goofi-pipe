# Audio engine

The third engine — a peer of signal and graphics inside one graph. Designed only in outline; the
graphics engine (engine #2) is what makes its seams real, and this one must plug into them
without a retrofit.

## What makes it different from the other two

An audio engine's clock is a **sample clock**, not a frame rate and not a self-paced node loop.
Its payload is a sample block, its transport wants a lock-free ring, and its deadline is hard in
a way neither of the others is: a late buffer is an audible artefact rather than a dropped frame.

## Locked decisions

- **Audio synthesis is eradicated from the signal graph.** It does not belong in a plane whose
  semantics are latest-wins with no queue — a synth needs every sample, in order. The signal
  engine stays latest-wins; audio gets its own runtime.
- **Audio and visual synthesis become separate cross-referenceable PANELS**, not a second node
  editor bolted onto the first. One graph, several editors, each showing its own engine's nodes.
- **The same platform machinery underneath.** One uid space, one `.gfi`, one command vocabulary
  with exact inverses, one CRDT document, one undo history, one workspace. Whatever the graphics
  engine needed generic, audio inherits.
- **Clock crossings are explicit nodes**, as they are for graphics — a bridge, never an implicit
  conversion, because that is where the cost and the semantics both live.

## Needs, in rough order

- A `SampleClock` runtime alongside the signal plane's self-scheduling threads.
- A sample-block payload and its wire format — the GOOF frame is a signal shape, and audio should
  get its own encoder behind the same seam rather than being squeezed into that one.
- An `rtrb`-style lock-free ring for the transport, and a real-time-safe allocation discipline on
  the audio thread.
- Device I/O: input capture and output, with the platform's own latency reporting surfaced.
- Bridge nodes both directions — signal↔audio, and eventually graphics↔audio.
- A waveform-summary viewer on the shared `/data` reduction plane, which was designed generic
  precisely so an audio payload could reduce through it.

## Open questions

- Whether an audio node's `process` runs on the audio thread at all, or whether the audio thread
  only ever drains a ring the node fills. The second is far easier to keep real-time-safe.
- How a patch expresses "this subgraph runs at the sample clock" without a second graph — the
  engine tag on a node type answers it, but the UX of a mixed patch is not designed.
- What a `.gfi` records about device selection, which is machine-specific by nature.
