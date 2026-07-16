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

/// Frequency-shift a real signal by translating its full (two-sided) FFT
/// spectrum `delta_bins` bins and inverting. The vacated bins are **zero-filled,
/// not wrapped** — this mirrors numpy's slice-assignment (`Y[d:] = X[:-d]`), so
/// content shifted past an edge is dropped rather than rolled around. Because a
/// one-sided shift breaks the spectrum's conjugate symmetry, the inverse is
/// complex; we return its real part (length `n`). A positive `delta_bins` moves
/// content toward higher bins. `delta_bins == 0` is the identity up to FFT
/// round-trip error; `|delta_bins| >= n` zeros the whole spectrum (all-zero out).
/// Takes a caller-owned [`FftPlanner`] (see [`magnitude_spectrum`]) so the
/// forward+inverse plans are cached across ticks.
pub fn frequency_shift(planner: &mut FftPlanner<f32>, signal: &[f32], delta_bins: i64) -> Vec<f32> {
    let n = signal.len();
    if n == 0 {
        return Vec::new();
    }
    let mut buf: Vec<Complex<f32>> = signal.iter().map(|&x| Complex::new(x, 0.0)).collect();
    planner.plan_fft_forward(n).process(&mut buf);

    // Shift with zero-fill. `unsigned_abs` avoids the i64::MIN negation overflow;
    // clamping to n makes an over-large shift a clean all-zero result (no panic).
    let mag = delta_bins.unsigned_abs().min(n as u64) as usize;
    let mut shifted = vec![Complex::new(0.0, 0.0); n];
    if delta_bins > 0 {
        shifted[mag..].copy_from_slice(&buf[..n - mag]);
    } else if delta_bins < 0 {
        shifted[..n - mag].copy_from_slice(&buf[mag..]);
    } else {
        shifted.copy_from_slice(&buf);
    }

    planner.plan_fft_inverse(n).process(&mut shifted);
    // rustfft's inverse is unnormalized -> divide by n; keep only the real part.
    let scale = 1.0 / n as f32;
    shifted.iter().map(|c| c.re * scale).collect()
}

/// Reconstruct a real time-domain signal from a magnitude spectrum and its phase
/// via the full inverse DFT — the kernel of goofi's Python IFFT node. Builds the
/// complex spectrum `mag * exp(i*phase)`, runs an unnormalized inverse FFT, and
/// returns its real part divided by `n` (matching numpy's normalized `ifft`).
///
/// `mag` and `phase` may differ in length; the shorter is treated as zero-padded
/// to the longer (numpy `np.concatenate` with zeros in the node). Returns empty
/// only when both inputs are empty. Takes a caller-owned [`FftPlanner`] (see
/// [`magnitude_spectrum`]) so a node keeps the plan cached across ticks.
pub fn inverse_fft_real(planner: &mut FftPlanner<f32>, mag: &[f32], phase: &[f32]) -> Vec<f32> {
    let n = mag.len().max(phase.len());
    if n == 0 {
        return Vec::new();
    }
    let mut buf: Vec<Complex<f32>> = (0..n)
        .map(|k| {
            // Out-of-range on the shorter input reads as a zero pad.
            let m = mag.get(k).copied().unwrap_or(0.0);
            let p = phase.get(k).copied().unwrap_or(0.0);
            // mag * exp(i*phase) = mag*(cos p + i sin p).
            Complex::new(m * p.cos(), m * p.sin())
        })
        .collect();
    let ifft = planner.plan_fft_inverse(n);
    ifft.process(&mut buf);
    // rustfft's inverse is unnormalized; numpy's ifft divides by n.
    let inv_n = 1.0 / n as f32;
    buf.iter().map(|c| c.re * inv_n).collect()
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
    fn inverse_fft_real_round_trips_a_forward_fft() {
        // Take the FULL forward FFT of a real signal to get (magnitude, phase),
        // then reconstruct: ifft(mag * e^{i*phase}).re must recover the samples.
        let n = 32usize;
        let x: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / n as f32;
                (2.0 * PI * 3.0 * t).sin() + 0.5 * (2.0 * PI * 7.0 * t).cos()
            })
            .collect();
        let mut planner = FftPlanner::new();
        let mut spec: Vec<Complex<f32>> = x.iter().map(|&v| Complex::new(v, 0.0)).collect();
        planner.plan_fft_forward(n).process(&mut spec);
        let mag: Vec<f32> = spec.iter().map(|c| c.norm()).collect();
        let phase: Vec<f32> = spec.iter().map(|c| c.arg()).collect();

        let recon = inverse_fft_real(&mut planner, &mag, &phase);
        assert_eq!(recon.len(), n);
        for (a, b) in x.iter().zip(&recon) {
            assert!((a - b).abs() < 1e-4, "reconstructed {b} != original {a}");
        }
    }

    #[test]
    fn inverse_fft_real_zero_pads_the_shorter_input() {
        // Only a DC magnitude given; phase is longer, so mag is zero-padded to its
        // length. A pure-DC spectrum reconstructs to a constant = dc / n.
        let recon = inverse_fft_real(&mut FftPlanner::new(), &[4.0], &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(recon.len(), 4, "output length follows the longer input");
        for v in &recon {
            assert!((v - 1.0).abs() < 1e-6, "DC-only spectrum -> constant, got {v}");
        }
        // Both empty -> empty (the node treats this as no-emit).
        assert!(inverse_fft_real(&mut FftPlanner::new(), &[], &[]).is_empty());
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

    #[test]
    fn frequency_shift_zero_is_identity() {
        // A zero-bin shift is just an FFT round-trip; the real part must recover
        // the input (within f32 FFT round-trip error).
        let sig: Vec<f32> = sine(16.0, 128.0, 128)
            .iter()
            .zip(sine(5.0, 128.0, 128))
            .map(|(a, b)| a + 0.4 * b)
            .collect();
        let out = frequency_shift(&mut FftPlanner::new(), &sig, 0);
        assert_eq!(out.len(), sig.len());
        for (o, s) in out.iter().zip(&sig) {
            assert!((o - s).abs() < 1e-3, "round-trip drifted: {o} vs {s}");
        }
    }

    #[test]
    fn frequency_shift_splits_sine_into_two_tones() {
        // A zero-filled two-sided shift of a real sine at bin k by d bins moves the
        // +freq image to k+d and the -freq image (at n-k) to n-(k-d); after taking
        // the real part that folds back to bin |k-d|. So the output is the sum of
        // two equal tones at bins {k+d, |k-d|} — the exact analytic signature of
        // this shift method. Here k=16, d=8 -> peaks at bins 24 and 8, nothing else.
        let (fs, n, k, d) = (128.0f32, 128usize, 16usize, 8i64);
        let out = frequency_shift(&mut FftPlanner::new(), &sine(k as f32, fs, n), d);
        let mag = magnitude_spectrum(&mut FftPlanner::new(), &out);
        let (lo, hi) = (k - d as usize, k + d as usize); // 8 and 24
        assert!(mag[lo] > 10.0 && mag[hi] > 10.0, "expected tones at {lo} and {hi}");
        assert!((mag[lo] - mag[hi]).abs() < 1.0, "the two tones must be equal-amplitude");
        for (b, &m) in mag.iter().enumerate() {
            if b != lo && b != hi {
                assert!(m < 1.0, "bin {b} should be silent, got {m}");
            }
        }
    }
}
