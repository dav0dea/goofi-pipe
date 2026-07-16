//! goofi-dsp — pure, dependency-light DSP kernels (no goofi types), the unit-test
//! surface for the native signal nodes. Grows with biquad filters, Hilbert,
//! resampling, etc. as those nodes are ported.

use std::f32::consts::PI;

use rustfft::num_complex::Complex;
pub use rustfft::FftPlanner;

/// Half-sample-symmetric ('reflect' / scipy default) index into a length-`n`
/// signal, so an out-of-range index mirrors about the edge pixel (…c b a | a b c…).
fn reflect(i: i64, n: i64) -> usize {
    let p = 2 * n;
    let mut i = ((i % p) + p) % p; // into [0, 2n)
    if i >= n {
        i = p - 1 - i;
    }
    i as usize
}

/// 1-D Gaussian smoothing (scipy `gaussian_filter1d`, `truncate=4.0`, `mode='reflect'`).
/// Length-preserving; `sigma <= 0` (or a sub-1 radius) is the identity.
pub fn gaussian_smooth1d(signal: &[f32], sigma: f32) -> Vec<f32> {
    let n = signal.len();
    if n == 0 {
        return Vec::new();
    }
    let radius = (4.0 * sigma + 0.5) as i64; // scipy: int(truncate * sd + 0.5)
    if sigma <= 0.0 || radius < 1 {
        return signal.to_vec();
    }
    let mut kernel: Vec<f32> = (-radius..=radius)
        .map(|i| (-0.5 * (i as f32 / sigma).powi(2)).exp())
        .collect();
    let sum: f32 = kernel.iter().sum();
    for w in &mut kernel {
        *w /= sum;
    }
    let (r, nn) = (radius, n as i64);
    (0..n)
        .map(|j| {
            let j = j as i64;
            kernel
                .iter()
                .enumerate()
                .map(|(k, &w)| w * signal[reflect(j + k as i64 - r, nn)])
                .sum()
        })
        .collect()
}

/// A Hann window of length `n` (n >= 1; n == 1 yields `[1.0]`).
pub fn hann(n: usize) -> Vec<f32> {
    if n <= 1 {
        return vec![1.0; n];
    }
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / (n as f32 - 1.0)).cos())
        .collect()
}

/// Magnitude spectrum (|FFT|) of a real signal, one-sided (length `n/2 + 1`).
/// Takes a caller-owned [`FftPlanner`] so a node can keep it alive across ticks —
/// rustfft caches the plan per length, so a repeated `n` never re-plans.
pub fn magnitude_spectrum(planner: &mut FftPlanner<f32>, signal: &[f32]) -> Vec<f32> {
    let n = signal.len();
    if n == 0 {
        return Vec::new();
    }
    let mut buf: Vec<Complex<f32>> = signal.iter().map(|&x| Complex::new(x, 0.0)).collect();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut buf);
    buf.iter().take(n / 2 + 1).map(|c| c.norm()).collect()
}

/// One-sided power spectral density via a Hann-windowed periodogram.
/// Returns `(freqs_hz, power)`, each of length `n/2 + 1`. Interior bins are
/// doubled (one-sided), DC and Nyquist are not. Scaled by `1/(fs * sum(w^2))`.
/// Takes a caller-owned [`FftPlanner`] (see [`magnitude_spectrum`]) to cache the plan.
pub fn psd_periodogram(planner: &mut FftPlanner<f32>, signal: &[f32], fs: f32) -> (Vec<f32>, Vec<f32>) {
    let n = signal.len();
    if n == 0 || fs <= 0.0 {
        return (Vec::new(), Vec::new());
    }
    let w = hann(n);
    let mut buf: Vec<Complex<f32>> = signal
        .iter()
        .zip(&w)
        .map(|(&x, &wi)| Complex::new(x * wi, 0.0))
        .collect();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut buf);

    let winpow: f32 = w.iter().map(|wi| wi * wi).sum();
    let scale = 1.0 / (fs * winpow.max(f32::MIN_POSITIVE));
    let half = n / 2 + 1;
    let nyquist_bin = if n % 2 == 0 { Some(half - 1) } else { None };

    let power: Vec<f32> = buf
        .iter()
        .take(half)
        .enumerate()
        .map(|(k, c)| {
            let mut p = c.norm_sqr() * scale;
            if k != 0 && Some(k) != nyquist_bin {
                p *= 2.0; // one-sided: fold negative frequencies onto the positive
            }
            p
        })
        .collect();
    let freqs: Vec<f32> = (0..half).map(|k| k as f32 * fs / n as f32).collect();
    (freqs, power)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(f: f32, fs: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI * f * i as f32 / fs).sin())
            .collect()
    }

    fn argmax(v: &[f32]) -> usize {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    }

    #[test]
    fn psd_peak_at_sine_frequency() {
        let fs = 256.0;
        let n = 256; // 1 Hz bin resolution
        let f0 = 20.0;
        let (freqs, power) = psd_periodogram(&mut FftPlanner::new(), &sine(f0, fs, n), fs);
        assert_eq!(freqs.len(), n / 2 + 1);
        let peak = argmax(&power);
        assert!(
            (freqs[peak] - f0).abs() <= 1.0,
            "PSD peak at {} Hz, expected ~{f0} Hz",
            freqs[peak]
        );
    }

    #[test]
    fn magnitude_spectrum_length_and_dc() {
        let sig = vec![1.0f32; 8]; // constant -> all energy at DC
        let mag = magnitude_spectrum(&mut FftPlanner::new(), &sig);
        assert_eq!(mag.len(), 8 / 2 + 1);
        assert_eq!(argmax(&mag), 0, "constant signal peaks at DC");
    }

    #[test]
    fn gaussian_smooth_preserves_length_and_constants() {
        // A constant signal is unchanged (kernel sums to 1); length preserved.
        let c = vec![3.0f32; 16];
        let out = gaussian_smooth1d(&c, 2.0);
        assert_eq!(out.len(), 16);
        for v in &out {
            assert!((v - 3.0).abs() < 1e-4, "constant stays constant, got {v}");
        }
        // sigma <= 0 is the identity.
        assert_eq!(gaussian_smooth1d(&[1.0, 5.0, 2.0], 0.0), vec![1.0, 5.0, 2.0]);
    }

    #[test]
    fn gaussian_smooth_reduces_variance_and_conserves_sum() {
        // A noisy alternating signal: smoothing shrinks its peak-to-peak, and the
        // reflect boundary conserves the total sum (kernel is normalized).
        let sig: Vec<f32> = (0..32).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let out = gaussian_smooth1d(&sig, 2.0);
        let ptp = |v: &[f32]| v.iter().cloned().fold(f32::MIN, f32::max) - v.iter().cloned().fold(f32::MAX, f32::min);
        assert!(ptp(&out) < ptp(&sig), "smoothing must reduce peak-to-peak");
        assert!((out.iter().sum::<f32>() - sig.iter().sum::<f32>()).abs() < 1e-3, "reflect conserves the sum");
    }

    #[test]
    fn hann_endpoints_are_zero() {
        let w = hann(16);
        assert!(w[0].abs() < 1e-6 && w[15].abs() < 1e-6);
    }
}
