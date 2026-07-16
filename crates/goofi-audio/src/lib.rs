//! goofi-audio — pure real-time audio cores (no transport/hardware deps).
//!
//! Consumed by the (future) SampleClock-paced Oscillator generator and the
//! AudioOut sink. Ported piece-by-piece from the Python `goofi/audio/` package.
//! Still to compose on top: the AudioOut non-blocking callback + the cpal
//! transport that drives it.

mod clock;
mod continuity;
mod drift;
mod ring;

pub use clock::SampleClock;
pub use continuity::{crossfade, is_discontinuous, INDEX_META_KEY};
pub use drift::DriftCorrector;
pub use ring::AudioRing;

mod sink;
pub use sink::{drain_into, AudioSink};

mod wavegen;
pub use wavegen::{WaveGen, Waveform};
