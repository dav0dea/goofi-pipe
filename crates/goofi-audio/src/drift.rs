//! Threshold-triggered single-frame stuff/drop drift correction over interleaved
//! `[frames, channels]` float32 blocks. Ported from Python `audio/drift.py`.
//!
//! When the consumer ring's `fill` drifts outside `target_fill ± deadband`, one
//! frame is dropped (overfull) or duplicated (underfull) at the block's frame
//! closest to zero energy, so the ~20 µs edit is inaudible. Inside the deadband
//! the block passes through unchanged.

pub struct DriftCorrector {
    target_fill: i64,
    deadband: i64,
}

impl DriftCorrector {
    pub fn new(target_fill: i64) -> DriftCorrector {
        DriftCorrector::with_deadband(target_fill, 64)
    }

    pub fn with_deadband(target_fill: i64, deadband: i64) -> DriftCorrector {
        DriftCorrector {
            target_fill,
            deadband,
        }
    }

    /// Return a possibly length-adjusted copy of `block` (interleaved, `channels`
    /// wide): `frames-1` when overfull, `frames+1` when underfull, else the block
    /// unchanged (cloned).
    pub fn correct(&self, block: &[f32], channels: usize, fill: i64) -> Vec<f32> {
        if channels == 0 || block.len() < channels {
            return block.to_vec(); // no whole frame; stuff/drop is undefined
        }
        if fill > self.target_fill + self.deadband {
            let idx = zero_crossing(block, channels);
            drop_frame(block, channels, idx)
        } else if fill < self.target_fill - self.deadband {
            let idx = zero_crossing(block, channels);
            insert_frame(block, channels, idx)
        } else {
            block.to_vec()
        }
    }
}

/// Index of the frame closest to zero energy (least audible place to edit).
fn zero_crossing(block: &[f32], channels: usize) -> usize {
    let frames = block.len() / channels;
    let mut best = 0usize;
    let mut best_mag = f32::INFINITY;
    for f in 0..frames {
        // Mono energy = |mean across channels| for this frame.
        let mut sum = 0.0f32;
        for c in 0..channels {
            sum += block[f * channels + c];
        }
        let mag = (sum / channels as f32).abs();
        if mag < best_mag {
            best_mag = mag;
            best = f;
        }
    }
    best
}

fn drop_frame(block: &[f32], channels: usize, frame: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(block.len() - channels);
    out.extend_from_slice(&block[..frame * channels]);
    out.extend_from_slice(&block[(frame + 1) * channels..]);
    out
}

fn insert_frame(block: &[f32], channels: usize, frame: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(block.len() + channels);
    out.extend_from_slice(&block[..(frame + 1) * channels]);
    out.extend_from_slice(&block[frame * channels..(frame + 1) * channels]); // duplicate
    out.extend_from_slice(&block[(frame + 1) * channels..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inside_deadband_passes_through() {
        let d = DriftCorrector::new(100);
        let block = vec![0.5f32, 0.1, 0.9];
        assert_eq!(d.correct(&block, 1, 100), block);
        assert_eq!(d.correct(&block, 1, 140), block); // within +/- 64
    }

    #[test]
    fn overfull_drops_the_quietest_frame() {
        let d = DriftCorrector::with_deadband(100, 10);
        let block = vec![0.5f32, 0.1, 0.9]; // frame 1 (0.1) is nearest zero
        let out = d.correct(&block, 1, 200); // fill 200 > 110 -> drop
        assert_eq!(out, vec![0.5, 0.9]);
    }

    #[test]
    fn underfull_duplicates_the_quietest_frame() {
        let d = DriftCorrector::with_deadband(100, 10);
        let block = vec![0.5f32, 0.1, 0.9];
        let out = d.correct(&block, 1, 50); // fill 50 < 90 -> stuff
        assert_eq!(out, vec![0.5, 0.1, 0.1, 0.9]);
    }

    #[test]
    fn multichannel_drop_removes_a_whole_frame() {
        let d = DriftCorrector::with_deadband(100, 10);
        // 2 channels, 3 frames; frame 1 = (0.0, 0.0) is quietest.
        let block = vec![1.0f32, 1.0, 0.0, 0.0, 2.0, 2.0];
        let out = d.correct(&block, 2, 300);
        assert_eq!(out, vec![1.0, 1.0, 2.0, 2.0]);
    }
}
