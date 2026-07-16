//! goofi-audio — pure real-time audio cores (no transport/hardware deps).
//!
//! Consumed by the (future) SampleClock-paced Oscillator generator and the
//! AudioOut sink. Ported piece-by-piece from the Python `goofi/audio/` package.
//! Still to come: the AudioRing SPSC jitter buffer (its lock-free split needs a
//! careful atomics design) and the cpal/transport that composes these.

mod clock;
mod continuity;
mod drift;

pub use clock::SampleClock;
pub use continuity::{crossfade, is_discontinuous, INDEX_META_KEY};
pub use drift::DriftCorrector;
