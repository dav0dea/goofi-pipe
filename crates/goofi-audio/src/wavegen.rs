//! `WaveGen` — a phase-continuous oscillator waveform generator. Ported from the
//! generation body of Python `nodes/inputs/oscillator.py`.
//!
//! The [`SampleClock`](crate::SampleClock) owns pacing (exactly how many samples
//! elapsed this tick); `WaveGen` turns that count into `n` samples of the selected
//! waveform, carrying an accumulated phase forward `mod 2π` so successive blocks
//! join seamlessly (no click at the block boundary). Frequency may change per call
//! (real-time control) without breaking phase continuity.

use std::f64::consts::{PI, TAU};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Waveform {
    Sine,
    Square,
    Sawtooth,
    Pulse,
}

impl Waveform {
    pub fn parse(s: &str) -> Waveform {
        match s {
            "square" => Waveform::Square,
            "sawtooth" => Waveform::Sawtooth,
            "pulse" => Waveform::Pulse,
            _ => Waveform::Sine,
        }
    }
}

pub struct WaveGen {
    phase: f64, // accumulated, kept in [0, 2π)
    waveform: Waveform,
    duty: f64, // square only
}

impl WaveGen {
    pub fn new(waveform: Waveform) -> WaveGen {
        WaveGen {
            phase: 0.0,
            waveform,
            duty: 0.5,
        }
    }

    pub fn with_duty(mut self, duty: f64) -> WaveGen {
        self.duty = duty;
        self
    }

    /// Generate `n` phase-continuous samples at `freq` Hz for sample rate `sfreq`,
    /// advancing the phase so the next call continues seamlessly.
    pub fn generate(&mut self, freq: f64, sfreq: f64, n: usize) -> Vec<f32> {
        if n == 0 || sfreq <= 0.0 {
            return Vec::new();
        }
        let inc = TAU * freq / sfreq;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let raw = self.phase + inc * i as f64;
            let osc_phase = raw.rem_euclid(TAU);
            let sample = match self.waveform {
                Waveform::Sine => osc_phase.sin(),
                Waveform::Square => {
                    if osc_phase < TAU * self.duty {
                        1.0
                    } else {
                        -1.0
                    }
                }
                Waveform::Sawtooth => osc_phase / PI - 1.0,
                Waveform::Pulse => {
                    // Fire on the sample whose step crosses a 2π boundary (wrap).
                    let next_raw = raw + inc;
                    if (next_raw / TAU).floor() > (raw / TAU).floor() {
                        1.0
                    } else {
                        0.0
                    }
                }
            };
            out.push(sample as f32);
        }
        self.phase = (self.phase + inc * n as f64).rem_euclid(TAU);
        out
    }

    pub fn phase(&self) -> f64 {
        self.phase
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sine_quarter_period_values() {
        // freq = sfreq/4 -> inc = π/2 -> phases 0, π/2, π, 3π/2 -> sin 0, 1, 0, -1.
        let mut g = WaveGen::new(Waveform::Sine);
        let s = g.generate(12_000.0, 48_000.0, 4);
        let expect = [0.0, 1.0, 0.0, -1.0];
        for (a, e) in s.iter().zip(expect) {
            assert!((a - e).abs() < 1e-6, "got {a}, want {e}");
        }
    }

    #[test]
    fn phase_is_continuous_across_calls() {
        // Splitting a run into two calls must be bit-for-bit the same waveform as
        // one call — i.e. no phase reset / click at the block boundary.
        let (freq, sfreq) = (440.0, 48_000.0);
        let mut whole = WaveGen::new(Waveform::Sine);
        let one = whole.generate(freq, sfreq, 512);

        let mut split = WaveGen::new(Waveform::Sine);
        let mut two = split.generate(freq, sfreq, 200);
        two.extend(split.generate(freq, sfreq, 312));

        assert_eq!(one.len(), two.len());
        for (a, b) in one.iter().zip(&two) {
            assert!((a - b).abs() < 1e-6, "discontinuity at the seam: {a} vs {b}");
        }
    }

    #[test]
    fn square_respects_duty_cycle() {
        // inc = π/2: phases 0, π/2, π, 3π/2. Duty 0.5 -> threshold π -> [1,1,-1,-1].
        let mut g = WaveGen::new(Waveform::Square).with_duty(0.5);
        assert_eq!(g.generate(12_000.0, 48_000.0, 4), vec![1.0, 1.0, -1.0, -1.0]);
    }

    #[test]
    fn sawtooth_ramps_from_minus_one() {
        // osc_phase/π - 1: phases 0, π/2, π, 3π/2 -> -1, -0.5, 0, 0.5.
        let mut g = WaveGen::new(Waveform::Sawtooth);
        let s = g.generate(12_000.0, 48_000.0, 4);
        let expect = [-1.0, -0.5, 0.0, 0.5];
        for (a, e) in s.iter().zip(expect) {
            assert!((a - e).abs() < 1e-6, "got {a}, want {e}");
        }
    }

    #[test]
    fn pulse_fires_once_per_period() {
        // freq = sfreq/8 -> period 8 samples. Over 16 samples exactly 2 wraps fire.
        let mut g = WaveGen::new(Waveform::Pulse);
        let s = g.generate(6_000.0, 48_000.0, 16);
        let fired: f32 = s.iter().sum();
        assert_eq!(fired, 2.0, "one pulse per 2π wrap: {s:?}");
    }

    #[test]
    fn zero_samples_and_zero_rate_are_empty() {
        let mut g = WaveGen::new(Waveform::Sine);
        assert!(g.generate(440.0, 48_000.0, 0).is_empty());
        assert!(g.generate(440.0, 0.0, 16).is_empty());
    }
}
