# A subprocess Python node's `stop()`

`goofi.Node.stop()` releases what `setup()` acquired — a receiver thread, a socket, a MIDI port.
It is called on the in-process tier, from `PyNode`'s `Drop`, on the node's own thread.

On the subprocess tier it is NOT called. The parent's `shutdown` closes the liveness pipe and
kills the child, and `serve.rs` loops until the process dies, so nothing runs the method.

## What is already decided

- The OS is what makes this tolerable rather than broken: a killed process releases its sockets,
  its ports and its threads. What is lost is the work a `stop()` does that the OS cannot — the
  note-offs `signal:MidiOut` owes a synthesizer for every note it left held.
- The hook stays ONE method with one meaning. Whatever carries it to the child must not become a
  second teardown vocabulary that the in-process tier does not share.
- Teardown's owner stays `Drop`, on both tiers. A `stop` on the `Node` trait would be a second
  owner beside the `Drop` a Rust node already has.

## What is open

- How the child is told. A `Request::Stop` is a wire-format change, and the codec golden pins that
  format. The liveness pipe the child already holds is the other candidate: the child would watch
  it and leave its loop when it closes, which needs no new request and no golden change.
- What bounds the wait. `shutdown` cannot block on a child that will not go, so the kill stays as
  the ceiling behind whatever graceful path is added.
