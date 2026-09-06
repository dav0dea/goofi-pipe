# Recording: every engine's frames, on one timeline

Any node's output, captured losslessly, across every engine, onto one clock. It reads its timeline
from `frame-time.md` and it is the first tenant of the panel add-on door in `library.md`.

## Decisions

**Never the `/data` plane.** The reducer serves at `MAX_VIEWER_FPS`, latest-wins, and REDUCES the
frame; `drain_inputs` keeps only a wire's newest. The viewer plane is lossy by design at every hop
and no recorder may touch it.

**A dedicated iceoryx2 service per armed slot.** Every data service is built with
`subscriber_max_buffer_size(1)` and safe overflow, so a subscriber late by one inter-frame gap
loses a frame silently — a journal commit against a 250 Hz node costs five. Depth is a
service-level property and raising the shared one MULTIPLIES: 256 subscribers × 64 KB a chunk makes
a depth of 32 half a gigabyte per slot. So an armed slot opens `goofi_<base>_rec_<slot>` with
`max_subscribers(1)`, `max_publishers(1)` and a deep buffer — 16 MB, and only while armed. The
producer encodes ONCE and publishes the same bytes to both.

**The plan owns the recorder's bell.** `wire_out` replaces an output's target set WHOLE, so a bell
the recorder opened for itself is disarmed by the next re-wire: a cable three nodes away stops the
recording, with no error anywhere. The armed set is settled state, delivered through `settle`, and
the plan computes the recording bell beside the consumer bells.

**One door, and the event id is ignored.** The recorder holds a single event service; any producer
rings it and the recorder sweeps EVERY armed buffer on wake. That spends none of the `EventId`
budget — 0 control, 1..=64 input slots, 65..=128 `nd()` channels, 255 the ceiling — and a burst of
rings coalesces into one sweep.

**Recording is a door on `Engine`.** Each engine transfers its own frames losslessly; one central
file manager owns the buffers, the writers and the folder. The three do not share a problem:

- **Signal** already encodes once in `publish`, so an armed slot publishes those bytes twice. A
  64-channel 1 kHz stream is 256 KB/s.
- **Audio** has no iceoryx2 on its path and a block is the arena's own memory. The door is a
  PRE-ALLOCATED lock-free SPSC ring written on the callback thread — no allocation, no lock, no
  syscall — drained by a non-RT thread. 48 kHz stereo f32 is 384 KB/s, so a few hundred
  milliseconds of ring makes a drop practically impossible while keeping the callback honest.
- **Graphics** is not like the other two. `Rgba16Float` at 1920×1080 is 16.6 MB a frame and 60 fps
  is 995 MB/s; the uint8 hop halves it. There is no buffered writer at that rate, so the graphics
  door ENCODES in the loop, and a hardware encoder is a dependency with a platform matrix of its
  own. Arming also registers demand, since a stage with no reader does not render.

**A real-time node is never stalled.** Every engine drops and counts. A drop is a standing error on
the node and a record in the manifest with its tick time. The requirement is "never SILENTLY", not
"never": a disk that cannot keep up loses data in any design, and the recording says where.

**A recording is a DIRECTORY, not an archive.** Unlike a `.gfi` it is written incrementally and can
be enormous, so packing is a later action rather than the format. `manifest.json` beside one file
per stream, in a folder the file manager makes.

**A rebirth is a gap, and it is recorded.** `service_base` carries `gen`, bumped on EVERY birth, so
a param change that restarts a node mints a new service name. The recorder re-resolves and writes
the discontinuity; a recording that smooths over a real gap is worse than one that stops.

**One lossless container per stream is always written; WAV, MP4 and CSV are EXPORTS.** Each is lossy
against a `Data` frame — WAV loses anything not audio-shaped, CSV loses f32 below nine digits, MP4
by construction — and a format choice must not be able to undo the buffered path. Graphics is the
stated exception: at 1 GB/s there is no lossless option, and the manifest says so.

**The panel arms by drag, and the drag raises an op.** `record arm <node>/<slot>` like every other
intent, so the CLI, an agent and a test reach the same door. A drag is a gesture and needs its own
touch door.

## Order of work

1. `frame-time.md`'s tick, which every stream's timeline reads.
2. The `Engine` door, the armed set through `settle`, and the signal half.
3. The central file manager: buffers, writers, the folder, the manifest.
4. The audio ring.
5. The panel, through `library.md`'s add-on door.
6. Graphics, with its encoder — last, and separately. Signal and audio together are 640 KB/s;
   graphics alone is two thousand times that, and making one feature of them makes the easy part
   wait on the hard one.

## Open

- Which lossless container. A raw frame stream beside a JSON sidecar is the least that works.
- Whether an export is written in parallel or produced from the lossless file afterwards.
- The hardware encoder per platform, and what a machine with none does.
- Whether a recording survives a `session load`, or a load ends it.
