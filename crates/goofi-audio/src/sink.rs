//! `AudioSink` — the hardware-independent core of the AudioOut node, composing
//! the four pure cores. Ported from Python `nodes/outputs/audioout.py`.
//!
//! The tick thread pushes generated audio via [`AudioSink::accept`]; the device
//! callback drains the shared ring via [`drain_into`] on its own clock. The
//! stream is PRIMED — not started until the ring reaches `target_fill` — and
//! outputs zeros on an empty ring, so playback never blocks the graph and the
//! producer never blocks. On a `meta["index"]` jump (an upstream drop/reset) the
//! seam is equal-power crossfaded so the phase discontinuity is inaudible; in
//! steady state a single-frame stuff/drop tracks DAC-vs-producer clock drift. The
//! actual `cpal`/device stream composes on top of this (it's the only untestable,
//! hardware-bound piece).

use std::sync::Arc;

use crate::continuity::{crossfade, is_discontinuous};
use crate::drift::DriftCorrector;
use crate::ring::AudioRing;

/// Discontinuity-seam crossfade length in frames — short + fixed, REPLACES the
/// first frames of the post-jump block (never prepends) so it can't inflate the
/// timebase.
const CROSSFADE_FRAMES: usize = 64;

pub struct AudioSink {
    ring: Arc<AudioRing>,
    channels: usize,
    target_fill: usize,
    started: bool,
    drift: DriftCorrector,
    prev_index: Option<u64>,
    prev_tail: Option<Vec<f32>>, // held last frame, for the next crossfade
    discontinuities: u64,
}

impl AudioSink {
    /// Build a sink for `sr` Hz, `channels`-wide audio, a `buffer_ms` jitter
    /// target, and a device `blocksize` (frames per callback). The ring is floored
    /// so it can always hold one callback block plus the jitter target.
    pub fn new(sr: usize, channels: usize, buffer_ms: usize, blocksize: usize) -> AudioSink {
        let target_fill = sr * buffer_ms / 1000;
        let cap = (target_fill * 4).max(blocksize + target_fill).max(1);
        AudioSink {
            ring: Arc::new(AudioRing::new(cap, channels)),
            channels,
            target_fill,
            started: false,
            drift: DriftCorrector::new(target_fill as i64),
            prev_index: None,
            prev_tail: None,
            discontinuities: 0,
        }
    }

    /// A handle to the shared ring for the device callback (drain with
    /// [`drain_into`] on the audio thread).
    pub fn ring(&self) -> Arc<AudioRing> {
        self.ring.clone()
    }

    /// Producer side (tick thread). Crossfade the seam on a `meta["index"]` jump,
    /// apply steady-state drift correction (only once primed — priming keeps the
    /// ring deliberately below target, so there is no drift to fix yet), push to
    /// the ring, and return `true` exactly on the tick the ring first reaches
    /// `target_fill` (the caller starts the device stream then).
    pub fn accept(&mut self, block: &[f32], index: Option<u64>) -> bool {
        let mut b = block.to_vec();

        if is_discontinuous(self.prev_index, index) {
            if let Some(tail) = &self.prev_tail {
                b = crossfade(tail, &b, self.channels, CROSSFADE_FRAMES);
            }
            self.discontinuities += 1;
        }
        self.prev_index = index;

        if self.started {
            b = self.drift.correct(&b, self.channels, self.ring.fill() as i64);
        }
        self.ring.write(&b);

        if b.len() >= self.channels {
            self.prev_tail = Some(b[b.len() - self.channels..].to_vec());
        }

        let just_primed = !self.started && self.ring.fill() >= self.target_fill;
        if just_primed {
            self.started = true;
        }
        just_primed
    }

    pub fn started(&self) -> bool {
        self.started
    }
    pub fn target_fill(&self) -> usize {
        self.target_fill
    }
    pub fn discontinuities(&self) -> u64 {
        self.discontinuities
    }
    pub fn underruns(&self) -> u64 {
        self.ring.underruns()
    }
    pub fn overruns(&self) -> u64 {
        self.ring.overruns()
    }
}

/// Device-callback drain: zero `out`, then copy whatever the ring holds (a short
/// ring leaves the zeroed tail = silence, never a block or an underflow error).
pub fn drain_into(ring: &AudioRing, out: &mut [f32]) {
    out.iter_mut().for_each(|x| *x = 0.0);
    ring.read_into(out);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stereo(frames: usize, v: f32) -> Vec<f32> {
        vec![v; frames * 2]
    }

    #[test]
    fn primes_before_starting_then_signals_once() {
        // sr 48k, 30 ms -> target_fill 1440 frames; feed 500-frame blocks.
        let mut sink = AudioSink::new(48_000, 2, 30, 256);
        assert_eq!(sink.target_fill(), 1440);
        assert!(!sink.accept(&stereo(500, 0.1), Some(0)), "500 < 1440: still priming");
        assert!(!sink.accept(&stereo(500, 0.1), Some(1)), "1000 < 1440: still priming");
        assert!(sink.accept(&stereo(500, 0.1), Some(2)), "1500 >= 1440: primed now");
        assert!(sink.started());
        assert!(!sink.accept(&stereo(500, 0.1), Some(3)), "already started: no re-signal");
    }

    #[test]
    fn empty_ring_callback_outputs_silence() {
        let sink = AudioSink::new(48_000, 2, 30, 256);
        let ring = sink.ring();
        let mut out = [7.0f32; 8];
        drain_into(&ring, &mut out);
        assert_eq!(out, [0.0; 8], "an empty ring yields silence, not stale data");
    }

    #[test]
    fn contiguous_indices_never_crossfade() {
        let mut sink = AudioSink::new(48_000, 2, 30, 256);
        sink.accept(&stereo(100, 1.0), Some(5));
        sink.accept(&stereo(100, 0.0), Some(6)); // contiguous
        assert_eq!(sink.discontinuities(), 0);
    }

    #[test]
    fn index_jump_crossfades_the_seam() {
        let mut sink = AudioSink::new(48_000, 2, 30, 256);
        sink.accept(&stereo(100, 1.0), Some(1)); // block A = all 1.0
        sink.accept(&stereo(100, 0.0), Some(3)); // gap (2 skipped) -> crossfade
        assert_eq!(sink.discontinuities(), 1);

        // Drain both blocks and inspect the seam. A (frames 0..100) is 1.0; B's
        // first frames ramp DOWN from ~1.0 (held tail) toward 0.0 over 64 frames,
        // then B is untouched (0.0).
        let ring = sink.ring();
        let mut out = vec![0.0f32; 200 * 2];
        drain_into(&ring, &mut out);
        assert_eq!(out[0], 1.0, "block A passes through");
        let seam = out[100 * 2]; // first sample of block B (channel 0)
        assert!(seam > 0.5 && seam < 1.0, "seam blended down from held 1.0: {seam}");
        assert_eq!(out[170 * 2], 0.0, "beyond the 64-frame crossfade B is untouched");
    }
}
