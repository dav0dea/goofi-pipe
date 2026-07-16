//! goofi-dsp — pure, dependency-light DSP kernels (no goofi types), the unit-test
//! surface for the native signal nodes. Grows with biquad filters, Hilbert,
//! resampling, etc. as those nodes are ported.

use std::f32::consts::PI;

use rustfft::{num_complex::Complex, FftPlanner};

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
pub fn magnitude_spectrum(signal: &[f32]) -> Vec<f32> {
    let n = signal.len();
    if n == 0 {
        return Vec::new();
    }
    let mut buf: Vec<Complex<f32>> = signal.iter().map(|&x| Complex::new(x, 0.0)).collect();
    let fft = FftPlanner::new().plan_fft_forward(n);
    fft.process(&mut buf);
    buf.iter().take(n / 2 + 1).map(|c| c.norm()).collect()
}

/// One-sided power spectral density via a Hann-windowed periodogram.
/// Returns `(freqs_hz, power)`, each of length `n/2 + 1`. Interior bins are
/// doubled (one-sided), DC and Nyquist are not. Scaled by `1/(fs * sum(w^2))`.
pub fn psd_periodogram(signal: &[f32], fs: f32) -> (Vec<f32>, Vec<f32>) {
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
    let fft = FftPlanner::new().plan_fft_forward(n);
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
        let (freqs, power) = psd_periodogram(&sine(f0, fs, n), fs);
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
        let mag = magnitude_spectrum(&sig);
        assert_eq!(mag.len(), 8 / 2 + 1);
        assert_eq!(argmax(&mag), 0, "constant signal peaks at DC");
    }

    #[test]
    fn hann_endpoints_are_zero() {
        let w = hann(16);
        assert!(w[0].abs() < 1e-6 && w[15].abs() < 1e-6);
    }
}
