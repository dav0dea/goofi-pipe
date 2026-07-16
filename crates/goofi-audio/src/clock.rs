//! `SampleClock` — a monotonic sample-count pacer for a rate-`sr` generator.
//!
//! `advance(now)` returns how many frames to emit so the cumulative emitted count
//! tracks `floor((now - ref) * sr)` — WITHOUT the `round(dt * sr)` quantization
//! drift of naive wall-clock chunk sizing. A single oversized gap (a long stall)
//! clamps to `max_block` and realigns the reference to `now`, dropping the backlog
//! instead of spending many ticks catching up. The realign carries the reference
//! as an exact `(time, emitted)` pair — no lossy `now - emitted/sr` roundtrip — so
//! paced ticks stay sample-exact. Ported from the Python `audio/clock.py` core.

/// Exact, drift-free per-tick sample counter.
pub struct SampleClock {
    sr: f64,
    max_block: u64,
    resync_after: f64,
    ref_time: f64,
    ref_emitted: u64,
    emitted: u64,
}

impl SampleClock {
    /// A clock at `sr` Hz with the default backlog policy (`max_block` 8192,
    /// `resync_after` 0.25 s).
    pub fn new(sr: f64) -> SampleClock {
        SampleClock::with_params(sr, 8192, 0.25)
    }

    pub fn with_params(sr: f64, max_block: u64, resync_after: f64) -> SampleClock {
        SampleClock {
            sr,
            max_block,
            resync_after,
            ref_time: 0.0,
            ref_emitted: 0,
            emitted: 0,
        }
    }

    /// Anchor the clock's reference at `now` and reset the emitted count.
    pub fn start(&mut self, now: f64) {
        self.ref_time = now;
        self.ref_emitted = 0;
        self.emitted = 0;
    }

    /// Frames to emit at wall-clock `now`. Never negative; clamped + realigned on
    /// an oversized backlog.
    pub fn advance(&mut self, now: f64) -> u64 {
        // Snap the sample product to its integer within a sub-sample epsilon:
        // float wall-clock inputs carry ~1e-12-sample representation noise that
        // would otherwise floor an exact boundary one sample short.
        let elapsed = (now - self.ref_time) * self.sr;
        let target = self.ref_emitted as i64 + (elapsed + 1e-6).floor() as i64;
        let raw = target - self.emitted as i64;
        let behind =
            (now - self.ref_time) - (self.emitted as i64 - self.ref_emitted as i64) as f64 / self.sr;

        if raw > self.max_block as i64 || behind > self.resync_after {
            // Long stall / oversized backlog: emit at most one block and realign
            // the reference to `now` so the clock resumes paced next tick.
            let n = raw.clamp(0, self.max_block as i64) as u64;
            self.emitted += n;
            self.ref_time = now;
            self.ref_emitted = self.emitted;
            n
        } else {
            let n = raw.max(0) as u64;
            self.emitted += n;
            n
        }
    }

    /// Total frames emitted since `start`.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn floor_samples(t: f64, sr: f64) -> u64 {
        (t * sr + 1e-6).floor() as u64
    }

    #[test]
    fn paced_ticks_stay_sample_exact_with_no_drift() {
        // Within the resync window (gaps < 0.25 s) the emitted count must equal
        // floor(now * sr) at EVERY tick — no accumulation of rounding error.
        let sr = 48_000.0;
        let mut c = SampleClock::new(sr);
        c.start(0.0);
        // Irregular but small gaps, over a long horizon (~2 s, 200+ ticks).
        let mut t = 0.0;
        let steps = [0.007, 0.011, 0.003, 0.013, 0.009];
        for i in 0..240 {
            t += steps[i % steps.len()];
            c.advance(t);
            assert_eq!(
                c.emitted(),
                floor_samples(t, sr),
                "drifted at t={t}: emitted {} != floor {}",
                c.emitted(),
                floor_samples(t, sr)
            );
        }
        // Sanity: ~2.06 s of 48 kHz audio ≈ 99k samples, tracking the ideal.
        assert!(c.emitted() > 90_000, "emitted {} samples", c.emitted());
    }

    #[test]
    fn exact_block_boundary_is_not_short_by_one() {
        // 480 samples = exactly 10 ms at 48 kHz; the epsilon must land it on 480,
        // not 479 (the drift bug this clock exists to avoid).
        let sr = 48_000.0;
        let mut c = SampleClock::new(sr);
        c.start(0.0);
        assert_eq!(c.advance(0.010), 480);
        assert_eq!(c.advance(0.020), 480);
        assert_eq!(c.emitted(), 960);
    }

    #[test]
    fn never_emits_negative_or_moves_backward() {
        let sr = 44_100.0;
        let mut c = SampleClock::new(sr);
        c.start(1.0);
        assert_eq!(c.advance(1.0), 0); // no time elapsed
        c.advance(1.05);
        let e = c.emitted();
        assert_eq!(c.advance(1.05), 0, "same instant emits nothing");
        assert_eq!(c.emitted(), e, "emitted count must not move backward");
    }

    #[test]
    fn long_stall_clamps_to_max_block_and_realigns() {
        let sr = 48_000.0;
        let mut c = SampleClock::with_params(sr, 8192, 0.25);
        c.start(0.0);
        c.advance(0.001); // ~48 samples
        // A 10 s stall would nominally owe ~480k samples; must clamp to one block.
        let n = c.advance(10.0);
        assert_eq!(n, 8192, "an oversized backlog clamps to max_block");
        // ...and the next paced tick resumes exact 10 ms blocks, not catch-up.
        assert_eq!(c.advance(10.010), 480, "clock realigned and resumed paced");
    }
}
