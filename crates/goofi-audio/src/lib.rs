//! goofi-audio — pure real-time audio cores (no transport/hardware deps).
//!
//! Consumed by the (future) SampleClock-paced Oscillator generator and the
//! AudioOut sink. Ported piece-by-piece from the Python `goofi/audio/` package.

mod clock;

pub use clock::SampleClock;
