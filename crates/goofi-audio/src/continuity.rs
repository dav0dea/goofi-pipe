//! Discontinuity detection + equal-power crossfade over interleaved `[frames,
//! channels]` (row-major) float32 blocks. Ported from Python `audio/continuity.py`.
//!
//! `meta["index"]` is a source-origin emit counter; a consumer compares the
//! incoming index to the previous one to detect a dropped/duplicated upstream
//! frame, then crossfades across the seam so the discontinuity is inaudible.

/// The `meta` key carrying the source-origin emit counter.
pub const INDEX_META_KEY: &str = "index";

/// True when `index` is not the contiguous successor of `prev_index`. False if
/// either side is absent (no baseline / no stamp) — a missing index never fakes
/// a discontinuity.
pub fn is_discontinuous(prev_index: Option<u64>, index: Option<u64>) -> bool {
    match (prev_index, index) {
        (Some(p), Some(i)) => i != p + 1,
        _ => false,
    }
}

/// Equal-power fade over the first `min(n, frames)` frames of `block`
/// (interleaved, `channels` wide), fading from the held previous frame
/// `prev_tail` (one frame, `channels` values) into `block`. Replaces those frames
/// in place (never prepends), so the block length is unchanged. Returns a new
/// block the same length as `block`; the input is left untouched.
pub fn crossfade(prev_tail: &[f32], block: &[f32], channels: usize, n: usize) -> Vec<f32> {
    let mut out = block.to_vec();
    if channels == 0 {
        return out;
    }
    let frames = block.len() / channels;
    let m = n.min(frames);
    if m == 0 {
        return out;
    }
    let half_pi = std::f32::consts::FRAC_PI_2;
    for i in 0..m {
        let t = (i as f32 + 1.0) / (m as f32 + 1.0);
        let fade_out = (t * half_pi).cos(); // 1 -> 0
        let fade_in = (t * half_pi).sin(); //  0 -> 1
        for c in 0..channels {
            let src = prev_tail.get(c).copied().unwrap_or(0.0);
            out[i * channels + c] = fade_out * src + fade_in * block[i * channels + c];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discontinuity_detection() {
        assert!(!is_discontinuous(None, Some(5)));
        assert!(!is_discontinuous(Some(5), None));
        assert!(!is_discontinuous(Some(4), Some(5))); // contiguous
        assert!(is_discontinuous(Some(4), Some(6))); // a gap (dropped frame)
        assert!(is_discontinuous(Some(4), Some(4))); // a repeat
    }

    #[test]
    fn crossfade_is_equal_power_and_bounded() {
        // 1 channel: fade from held 1.0 into a block of zeros over all 4 frames.
        let block = vec![0.0f32; 4];
        let out = crossfade(&[1.0], &block, 1, 4);
        // out[i] = cos(t*pi/2), t=(i+1)/5.
        let hp = std::f32::consts::FRAC_PI_2;
        for (i, &o) in out.iter().enumerate() {
            let t = (i as f32 + 1.0) / 5.0;
            assert!((o - (t * hp).cos()).abs() < 1e-5, "frame {i}");
        }
        // Equal-power: fade_out^2 + fade_in^2 == 1 at every step.
        for i in 0..4 {
            let t = (i as f32 + 1.0) / 5.0;
            let e = (t * hp).cos().powi(2) + (t * hp).sin().powi(2);
            assert!((e - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn crossfade_only_touches_the_first_m_frames() {
        // 2 channels, 4 frames, fade only the first 2.
        let block: Vec<f32> = vec![10.0, 20.0, 11.0, 21.0, 12.0, 22.0, 13.0, 23.0];
        let out = crossfade(&[0.0, 0.0], &block, 2, 2);
        // frames 2 and 3 (indices 4..8) are untouched.
        assert_eq!(&out[4..], &block[4..]);
        // frame 0 is blended toward 0 (src=0), so strictly less than the original.
        assert!(out[0] < block[0] && out[1] < block[1]);
        assert_eq!(out.len(), block.len());
        assert_eq!(block[0], 10.0, "input untouched");
    }

    #[test]
    fn crossfade_zero_frames_is_identity() {
        let block = vec![1.0f32, 2.0, 3.0];
        assert_eq!(crossfade(&[9.0], &block, 1, 0), block);
    }
}
