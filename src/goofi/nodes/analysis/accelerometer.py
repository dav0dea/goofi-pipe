import numpy as np
import pandas as pd
from scipy import signal
from numpy.linalg import eigh

from goofi.data import Data, DataType, to_data
from goofi.node import Node
from goofi.params import FloatParam


# ---------- Node: single-window, multi-device ----------
class Accelerometer(Node):
    """
    Single-chunk (no internal window/hop) accelerometer features for 1..N devices.
    Inputs x,y,z,tot can be 1D or 2D. If 2D, the larger dim is treated as time.
    Outputs a TABLE where each column is a vector (n_devices,).
    """
    @staticmethod
    def config_input_slots():
        return {"x": DataType.ARRAY, "y": DataType.ARRAY, "z": DataType.ARRAY, "tot": DataType.ARRAY}

    @staticmethod
    def config_output_slots():
        return {"features": DataType.TABLE}

    @staticmethod
    def config_params():
        return {
            "processing": {
                "hp_hz": FloatParam(0.25, 0.0, 5.0, doc="High-pass cutoff (Hz)"),
                "lp_hz": FloatParam(15.0, 5.0, 50.0, doc="Low-pass cutoff (Hz)"),
            }
        }

    def process(self, x: Data, y: Data, z: Data, tot: Data):
        if x is None and y is None and z is None and tot is None:
            return None

        hp_hz = self.params.processing.hp_hz.value
        lp_hz = self.params.processing.lp_hz.value

        # choose fs from first available stream
        fs = 100.0
        for d in (x, y, z, tot):
            if d is not None:
                fs = d.meta.get("sfreq", fs); break

        # unwrap arrays (they can be 1D/2D)
        x_arr = x.data if x is not None else None
        y_arr = y.data if y is not None else None
        z_arr = z.data if z is not None else None
        t_arr = tot.data if tot is not None else None

        feats = accel_features_block(
            x=x_arr, y=y_arr, z=z_arr, tot=t_arr,
            fs=fs, hp_hz=hp_hz, lp_hz=lp_hz
        )

        # pack to goofi TABLE: each key -> vector (n_devices,)
        table_output = {}
        for k, v in feats.items():
            table_output[k] = to_data(np.asarray(v, dtype=np.float64), meta={})

        return {"features": (table_output, {})}


# ---------- helpers ----------
def _to_devices_by_samples(arr):
    """Return (n_dev, n_samp) view. Accepts 1D or 2D; assumes larger dim is time."""
    a = np.asarray(arr)
    if a.ndim == 1:
        return a[None, :]
    if a.shape[1] >= a.shape[0]:
        return a  # (n_dev, n_samp) if second dim looks like time? keep as-is
    else:
        return a.T  # otherwise transpose

def _butter_zero_phase(sig, fs, hp, lp):
    sos_list = []
    if hp is not None and hp > 0:
        sos_list.append(signal.butter(2, hp/(fs/2), btype="highpass", output="sos"))
    if lp is not None and lp > 0:
        sos_list.append(signal.butter(4, lp/(fs/2), btype="lowpass", output="sos"))
    out = sig
    for sos in sos_list:
        out = signal.sosfiltfilt(sos, out)
    return out

def _spectral_stats(sig, fs, band_a=(0.5, 2.0), band_b=(2.0, 5.0)):
    s = sig - np.mean(sig)
    if np.allclose(s, 0):
        return dict(dom_freq_hz=np.nan, dom_power_frac=np.nan,
                    spec_entropy=np.nan, spec_centroid_hz=np.nan,
                    spec_spread_hz=np.nan, band_a_frac=np.nan, band_b_frac=np.nan)
    win = np.hanning(len(s))
    fft = np.fft.rfft(s * win)
    freqs = np.fft.rfftfreq(len(s), d=1/fs)
    power = (fft.real**2 + fft.imag**2)

    p_no0 = power.copy()
    if len(p_no0) > 0:
        p_no0[0] = 0.0
    total_pow = power.sum() + 1e-12
    k_dom = int(np.argmax(p_no0))
    dom_freq = float(freqs[k_dom])
    dom_frac = float(power[k_dom] / total_pow)

    p = power / total_pow
    spec_ent = -np.sum(p * np.log(p + 1e-12)) / np.log(len(p) + 1e-12)
    centroid = float(np.sum(freqs * power) / total_pow)
    spread = float(np.sqrt(np.sum(((freqs - centroid)**2) * power) / total_pow))

    def band_frac(band):
        lo, hi = band
        idx = np.where((freqs >= lo) & (freqs <= hi))[0]
        return float(power[idx].sum() / total_pow) if idx.size else np.nan

    return dict(dom_freq_hz=dom_freq,
                dom_power_frac=dom_frac,
                spec_entropy=float(spec_ent),
                spec_centroid_hz=centroid,
                spec_spread_hz=spread,
                band_a_frac=band_frac(band_a),
                band_b_frac=band_frac(band_b))

def _autocorr_first_peak(sig, fs, min_period=0.2, max_period=3.0):
    s = sig - np.mean(sig)
    if np.allclose(s, 0):
        return np.nan, np.nan
    acf_full = signal.correlate(s, s, mode="full")
    acf = acf_full[acf_full.size // 2:]
    acf /= (acf[0] + 1e-12)
    lags = np.arange(len(acf)) / fs
    lo = int(np.ceil(min_period * fs))
    hi = min(int(np.floor(max_period * fs)), len(acf) - 1)
    if hi <= lo:
        return np.nan, np.nan
    k = lo + int(np.argmax(acf[lo:hi+1]))
    return float(lags[k]), float(acf[k])

def _safe_corr(a, b):
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])

# ---------- core: single-chunk features per device ----------
def accel_features_block(
    x=None, y=None, z=None, tot=None,
    fs=100.0,
    hp_hz=0.25, lp_hz=15.0,
    band_a=(0.5, 2.0), band_b=(2.0, 5.0),
):
    """
    Compute features over ONE chunk per device (no internal window/hop).
    Inputs can be 1D or 2D; if 2D, we assume the larger dimension is time.
    If x,y,z provided, tot is optional; if tot missing, magnitude is computed.
    Returns a dict[str, np.ndarray] with shape (n_devices,) per feature.
    """
    have_xyz = (x is not None) and (y is not None) and (z is not None)
    have_tot = (tot is not None)

    if not have_xyz and not have_tot:
        raise ValueError("Provide either (x,y,z) or tot.")

    if have_xyz:
        X = _to_devices_by_samples(x)
        Y = _to_devices_by_samples(y)
        Z = _to_devices_by_samples(z)
        # align shapes
        n_dev = min(X.shape[0], Y.shape[0], Z.shape[0])
        n_s   = min(X.shape[1], Y.shape[1], Z.shape[1])
        X, Y, Z = X[:n_dev, :n_s], Y[:n_dev, :n_s], Z[:n_dev, :n_s]
        if have_tot:
            T = _to_devices_by_samples(tot)[:n_dev, :n_s]
        else:
            T = np.sqrt(X*X + Y*Y + Z*Z)
        # filter
        Xf = np.vstack([_butter_zero_phase(X[i], fs, hp_hz, lp_hz) for i in range(n_dev)])
        Yf = np.vstack([_butter_zero_phase(Y[i], fs, hp_hz, lp_hz) for i in range(n_dev)])
        Zf = np.vstack([_butter_zero_phase(Z[i], fs, hp_hz, lp_hz) for i in range(n_dev)])
        Mf = np.vstack([_butter_zero_phase(T[i], fs, hp_hz, lp_hz) for i in range(n_dev)])
        VM = np.sqrt(Xf*Xf + Yf*Yf + Zf*Zf)
    else:
        T = _to_devices_by_samples(tot)
        n_dev, n_s = T.shape
        Mf = np.vstack([_butter_zero_phase(T[i], fs, hp_hz, lp_hz) for i in range(n_dev)])
        VM = Mf  # only magnitude available
        Xf = Yf = Zf = None

    # allocate outputs
    out = {}

    # intensity & distribution on magnitude
    mean_mag = np.mean(Mf, axis=1)
    std_mag  = np.std(Mf, axis=1)
    rms_mag  = np.sqrt(np.mean(Mf**2, axis=1))
    p2p_mag  = (np.max(Mf, axis=1) - np.min(Mf, axis=1))
    q25      = np.percentile(Mf, 25, axis=1)
    q75      = np.percentile(Mf, 75, axis=1)
    iqr_mag  = q75 - q25
    med      = np.median(Mf, axis=1)
    mad_mag  = 1.4826 * np.median(np.abs(Mf - med[:, None]), axis=1)
    cv_mag   = std_mag / (np.abs(mean_mag) + 1e-12)

    out.update(dict(
        mean_mag=mean_mag, std_mag=std_mag, rms_mag=rms_mag,
        p2p_mag=p2p_mag, iqr_mag=iqr_mag, mad_mag=mad_mag, cv_mag=cv_mag
    ))

    # jerk (m/s^3) on magnitude
    jerk_rms = np.full(n_dev, np.nan)
    jerk_mean_abs = np.full(n_dev, np.nan)
    if n_s > 1:
        d = np.diff(Mf, axis=1) * fs
        jerk_rms = np.sqrt(np.mean(d**2, axis=1))
        jerk_mean_abs = np.mean(np.abs(d), axis=1)
    out.update(dict(jerk_rms=jerk_rms, jerk_mean_abs=jerk_mean_abs))

    # spectral & periodicity on magnitude
    dom_freq_hz = np.empty(n_dev); dom_power_frac = np.empty(n_dev)
    spec_entropy = np.empty(n_dev); spec_centroid_hz = np.empty(n_dev)
    spec_spread_hz = np.empty(n_dev); band_a_frac = np.empty(n_dev); band_b_frac = np.empty(n_dev)
    ac_peak_lag_s = np.empty(n_dev); ac_peak_val = np.empty(n_dev)
    zero_cross_rate = np.empty(n_dev); robust_outlier_rate = np.empty(n_dev)

    for i in range(n_dev):
        s = Mf[i]
        spec = _spectral_stats(s, fs, band_a, band_b)
        dom_freq_hz[i]     = spec["dom_freq_hz"]
        dom_power_frac[i]  = spec["dom_power_frac"]
        spec_entropy[i]    = spec["spec_entropy"]
        spec_centroid_hz[i]= spec["spec_centroid_hz"]
        spec_spread_hz[i]  = spec["spec_spread_hz"]
        band_a_frac[i]     = spec["band_a_frac"]
        band_b_frac[i]     = spec["band_b_frac"]

        lag, val = _autocorr_first_peak(s, fs)
        ac_peak_lag_s[i], ac_peak_val[i] = lag, val

        s0 = s - np.mean(s)
        zero_cross_rate[i] = np.mean(np.abs(np.diff(np.signbit(s0))).astype(float))
        if mad_mag[i] > 1e-12:
            zrob = (s - np.median(s)) / mad_mag[i]
            robust_outlier_rate[i] = np.mean(np.abs(zrob) > 3.0)
        else:
            robust_outlier_rate[i] = 0.0

    out.update(dict(
        dom_freq_hz=dom_freq_hz, dom_power_frac=dom_power_frac,
        spec_entropy=spec_entropy, spec_centroid_hz=spec_centroid_hz,
        spec_spread_hz=spec_spread_hz, band_a_frac=band_a_frac, band_b_frac=band_b_frac,
        ac_peak_lag_s=ac_peak_lag_s, ac_peak_val=ac_peak_val,
        zero_cross_rate=zero_cross_rate, robust_outlier_rate=robust_outlier_rate
    ))

    # multi-axis extras (only if x,y,z available)
    if have_xyz:
        mean_x = np.mean(Xf, axis=1); mean_y = np.mean(Yf, axis=1); mean_z = np.mean(Zf, axis=1)
        std_x  = np.std(Xf, axis=1);  std_y  = np.std(Yf, axis=1);  std_z  = np.std(Zf, axis=1)
        sma    = np.mean(np.abs(Xf) + np.abs(Yf) + np.abs(Zf), axis=1)
        rms_vm = np.sqrt(np.mean(VM**2, axis=1))
        p2p_vm = (np.max(VM, axis=1) - np.min(VM, axis=1))
        out.update(dict(
            mean_x=mean_x, mean_y=mean_y, mean_z=mean_z,
            std_x=std_x, std_y=std_y, std_z=std_z,
            mean_vm=np.mean(VM, axis=1), std_vm=np.std(VM, axis=1),
            sma=sma, rms_vm=rms_vm, p2p_vm=p2p_vm
        ))

        # correlations and geometry per device
        corr_xy = np.full(n_dev, np.nan); corr_yz = np.full(n_dev, np.nan); corr_zx = np.full(n_dev, np.nan)
        pca_var_pc1 = np.full(n_dev, np.nan); pca_planarity = np.full(n_dev, np.nan); pca_sphericity = np.full(n_dev, np.nan)
        fa = np.full(n_dev, np.nan); shape_L = np.full(n_dev, np.nan); shape_P = np.full(n_dev, np.nan)
        shape_S = np.full(n_dev, np.nan); asphericity_kappa2 = np.full(n_dev, np.nan)
        iso_power = np.full(n_dev, np.nan); dir_resultant_R = np.full(n_dev, np.nan); spherical_variance = np.full(n_dev, np.nan)

        for i in range(n_dev):
            sx, sy, sz = Xf[i], Yf[i], Zf[i]
            corr_xy[i] = _safe_corr(sx, sy)
            corr_yz[i] = _safe_corr(sy, sz)
            corr_zx[i] = _safe_corr(sz, sx)

            M = np.vstack([sx - np.mean(sx), sy - np.mean(sy), sz - np.mean(sz)])
            if M.shape[1] > 1:
                C = (M @ M.T) / (M.shape[1] - 1)
            else:
                C = np.eye(3)
            evals, _ = eigh(C)
            lam = np.sort(evals)[::-1].astype(float)  # e1>=e2>=e3
            e1, e2, e3 = lam
            total_var = e1 + e2 + e3 + 1e-12
            pca_var_pc1[i] = e1 / total_var
            pca_planarity[i] = e2 / (e1 + 1e-12)
            pca_sphericity[i] = e3 / (e1 + 1e-12)

            lbar = lam.mean()
            fa[i] = np.sqrt(1.5) * np.linalg.norm(lam - lbar) / (np.linalg.norm(lam) + 1e-12)
            shape_L[i] = (lam[0] - lam[1]) / (lam[0] + 1e-12)
            shape_P[i] = (lam[1] - lam[2]) / (lam[0] + 1e-12)
            shape_S[i] = lam[2] / (lam[0] + 1e-12)
            asphericity_kappa2[i] = 1.0 - (3.0 * (lam[0]*lam[1] + lam[1]*lam[2] + lam[2]*lam[0])) / ((lam.sum())**2 + 1e-12)

            var_axes = np.array([np.var(sx), np.var(sy), np.var(sz)]) + 1e-12
            iso_power[i] = 3.0 * var_axes.min() / var_axes.sum()

            A = np.vstack([sx, sy, sz]).T
            A0 = A - A.mean(axis=0, keepdims=True)
            norms = np.linalg.norm(A0, axis=1)
            mask = norms > 1e-6
            if np.any(mask):
                U = A0[mask] / norms[mask, None]
                w = norms[mask] + 1e-12
                U_wm = (U * w[:, None]).sum(axis=0) / w.sum()
                dir_resultant_R[i] = np.linalg.norm(U_wm)
                spherical_variance[i] = 2.0 * (1.0 - dir_resultant_R[i])
            else:
                dir_resultant_R[i] = np.nan
                spherical_variance[i] = np.nan

        out.update(dict(
            corr_xy=corr_xy, corr_yz=corr_yz, corr_zx=corr_zx,
            pca_var_pc1=pca_var_pc1, pca_planarity=pca_planarity, pca_sphericity=pca_sphericity,
            fa=fa, shape_L=shape_L, shape_P=shape_P, shape_S=shape_S, asphericity_kappa2=asphericity_kappa2,
            iso_power=iso_power, dir_resultant_R=dir_resultant_R, spherical_variance=spherical_variance
        ))

    return out
