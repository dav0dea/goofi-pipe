import numpy as np  
import pandas as pd  
from scipy import signal  
from numpy.linalg import eigh  
  
from goofi.data import Data, DataType, to_data  
from goofi.node import Node  
from goofi.params import FloatParam  
  
class Accelerometer(Node):  
    """  
    Computes windowed accelerometer features from x, y, z acceleration streams.  
    Extracts temporal, spectral, and geometric features for motion analysis.  
      
    Inputs:  
    - x: X-axis acceleration array (m/s^2)  
    - y: Y-axis acceleration array (m/s^2)  
    - z: Z-axis acceleration array (m/s^2)  
    - tot: Optional total magnitude array (if None, computed from x,y,z)  
      
    Outputs:  
    - features: TABLE containing all computed features as Data objects  
    """  
      
    @staticmethod  
    def config_input_slots():  
        return {  
            "x": DataType.ARRAY,  
            "y": DataType.ARRAY,  
            "z": DataType.ARRAY,  
            "tot": DataType.ARRAY  
        }  
      
    @staticmethod  
    def config_output_slots():  
        return {"features": DataType.TABLE}  
      
    @staticmethod  
    def config_params():  
        return {  
            "processing": {  
                "window_sec": FloatParam(5.0, 0.5, 30.0, doc="Window length in seconds"),  
                "hop_sec": FloatParam(2.5, 0.1, 15.0, doc="Hop length in seconds"),  
                "hp_hz": FloatParam(0.25, 0.0, 5.0, doc="High-pass cutoff (Hz)"),  
                "lp_hz": FloatParam(15.0, 5.0, 50.0, doc="Low-pass cutoff (Hz)"),  
            }  
        }  
      
    def process(self, x: Data, y: Data, z: Data, tot: Data):  
        if x is None and y is None and z is None and tot is None:  
            return None  
        
          
        
        # Get parameters  
        window_sec = self.params.processing.window_sec.value  
        hop_sec = self.params.processing.hop_sec.value  
        hp_hz = self.params.processing.hp_hz.value  
        lp_hz = self.params.processing.lp_hz.value  
        
        if x is not None and y is not None and z is not None:
            # Extract arrays  
            x_arr = x.data.flatten()  
            y_arr = y.data.flatten()  
            z_arr = z.data.flatten()  
            tot_arr = tot.data.flatten() if tot is not None else None  
            
            # Extract sampling frequency from metadata  
            fs = x.meta.get("sfreq", 100.0)
            # Call your feature extraction function  
            features_df = accel_features_multi_axis(  
                x_arr, y_arr, z_arr,   
                total=tot_arr,  
                fs=fs,  
                window_sec=window_sec,  
                hop_sec=hop_sec,  
                hp_hz=hp_hz,  
                lp_hz=lp_hz  
            )  
        elif tot is not None:
            # Extract array  
            tot_arr = tot.data.flatten()  
            
            # Extract sampling frequency from metadata  
            fs = tot.meta.get("sfreq", 100.0)
            
            # Call your feature extraction function  
            features_df = accel_features_total(  
                tot_arr,  
                fs=fs,  
                window_sec=window_sec,  
                hop_sec=hop_sec,  
                hp_hz=hp_hz,  
                lp_hz=lp_hz  
            )
        else:
            return None
          
        # Convert DataFrame to goofi TABLE format  
        # Each column becomes a key with a Data object value  
        table_output = {}  
        for col in features_df.columns:  
            values = features_df[col].values  
            # Convert to numpy array and wrap in Data  
            table_output[col] = to_data(np.array(values, dtype=np.float64), meta={})  
          
        return {"features": (table_output, {})} 
    
import numpy as np
import pandas as pd
from scipy import signal
from numpy.linalg import eigh

def accel_features_multi_axis(
    x, y, z, total=None,
    fs=100.0,
    window_sec=5.0,
    hop_sec=2.5,
    hp_hz=0.25,         # high-pass to remove drift
    lp_hz=15.0,         # low-pass to focus on human motion
    band_a=(0.5, 2.0),  # slow wave / sway band
    band_b=(2.0, 5.0),  # faster gestures/steps band
):
    # ---- prep arrays ----
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    n = min(len(x), len(y), len(z))
    x, y, z = x[:n], y[:n], z[:n]

    vm = np.sqrt(x*x + y*y + z*z)  # vector magnitude
    mag = np.asarray(total, dtype=float)[:n] if total is not None else vm

    # ---- filtering (zero-phase) ----
    def butter_filter(sig, fs, hp, lp):
        sos_list = []
        if hp is not None and hp > 0:
            sos_list.append(signal.butter(2, hp/(fs/2), btype='highpass', output='sos'))
        if lp is not None and lp > 0:
            sos_list.append(signal.butter(4, lp/(fs/2), btype='lowpass', output='sos'))
        out = sig.copy()
        for sos in sos_list:
            out = signal.sosfiltfilt(sos, out)
        return out

    x_f = butter_filter(x, fs, hp_hz, lp_hz)
    y_f = butter_filter(y, fs, hp_hz, lp_hz)
    z_f = butter_filter(z, fs, hp_hz, lp_hz)
    mag_f = butter_filter(mag, fs, hp_hz, lp_hz)

    # ---- windowing ----
    win = int(round(window_sec * fs))
    hop = int(round(hop_sec * fs))
    if win <= 1 or hop <= 0 or win > n:
        raise ValueError("Check fs/window_sec/hop_sec — invalid window/hop sizes.")
    starts = np.arange(0, n - win + 1, hop)

    # ---- helpers ----
    def safe_corr(a, b):
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])

    def spectral_stats(sig, fs):
        sig_d = sig - np.mean(sig)
        if np.allclose(sig_d, 0):
            return dict(dom_freq_hz=np.nan, dom_power_frac=np.nan,
                        spec_entropy=np.nan, spec_centroid_hz=np.nan,
                        spec_spread_hz=np.nan, band_a_frac=np.nan, band_b_frac=np.nan)
        fft = np.fft.rfft(sig_d * np.hanning(len(sig_d)))
        freqs = np.fft.rfftfreq(len(sig_d), d=1/fs)
        power = (fft.real**2 + fft.imag**2)
        p_no0 = power.copy()
        if len(p_no0) > 0: p_no0[0] = 0.0
        total_pow = power.sum() + 1e-12
        k_dom = int(np.argmax(p_no0))
        dom_freq = freqs[k_dom]
        dom_frac = float(power[k_dom] / total_pow)
        p = power / total_pow
        spec_ent = -np.sum(p * np.log(p + 1e-12)) / np.log(len(p) + 1e-12)
        centroid = float(np.sum(freqs * power) / total_pow)
        spread = float(np.sqrt(np.sum(((freqs - centroid)**2) * power) / total_pow))
        def band_frac(band):
            lo, hi = band
            idx = np.where((freqs >= lo) & (freqs <= hi))[0]
            return float(power[idx].sum() / total_pow) if idx.size else np.nan
        return dict(dom_freq_hz=float(dom_freq),
                    dom_power_frac=float(dom_frac),
                    spec_entropy=float(spec_ent),
                    spec_centroid_hz=float(centroid),
                    spec_spread_hz=float(spread),
                    band_a_frac=band_frac(band_a),
                    band_b_frac=band_frac(band_b))

    def autocorr_first_peak(sig, fs, min_period=0.2, max_period=3.0):
        s = sig - np.mean(sig)
        if np.allclose(s, 0):
            return np.nan, np.nan
        acf_full = signal.correlate(s, s, mode='full')
        acf = acf_full[acf_full.size // 2:]
        acf /= (acf[0] + 1e-12)
        lags = np.arange(len(acf)) / fs
        lo = int(np.ceil(min_period * fs))
        hi = min(int(np.floor(max_period * fs)), len(acf) - 1)
        if hi <= lo:
            return np.nan, np.nan
        k = lo + int(np.argmax(acf[lo:hi+1]))
        return float(lags[k]), float(acf[k])

    rows = []
    for s in starts:
        e = s + win
        sx, sy, sz, sm = x_f[s:e], y_f[s:e], z_f[s:e], mag_f[s:e]
        vm_w = np.sqrt(sx*sx + sy*sy + sz*sz)

        # basic stats
        mean_x, mean_y, mean_z = float(np.mean(sx)), float(np.mean(sy)), float(np.mean(sz))
        std_x, std_y, std_z   = float(np.std(sx)),  float(np.std(sy)),  float(np.std(sz))
        mean_vm, std_vm       = float(np.mean(vm_w)), float(np.std(vm_w))
        sma = float(np.mean(np.abs(sx) + np.abs(sy) + np.abs(sz)))
        rms_vm = float(np.sqrt(np.mean(vm_w**2)))
        p2p_vm = float(np.max(vm_w) - np.min(vm_w))

        # jerk
        if len(vm_w) > 1:
            jerk = np.diff(vm_w) * fs
            jerk_rms = float(np.sqrt(np.mean(jerk**2)))
            jerk_mean_abs = float(np.mean(np.abs(jerk)))
        else:
            jerk_rms = np.nan
            jerk_mean_abs = np.nan

        # spectral on magnitude
        spec = spectral_stats(vm_w, fs)

        # periodicity
        ac_lag_s, ac_val = autocorr_first_peak(vm_w, fs)

        # inter-axis correlation
        corr_xy = safe_corr(sx, sy)
        corr_yz = safe_corr(sy, sz)
        corr_zx = safe_corr(sz, sx)

        # --- geometry: covariance & eigenvalues ---
        M = np.vstack([sx - np.mean(sx), sy - np.mean(sy), sz - np.mean(sz)])
        C = (M @ M.T) / (M.shape[1] - 1) if M.shape[1] > 1 else np.eye(3)
        evals, _ = eigh(C)                # ascending
        evals = np.sort(evals)[::-1]      # e1 >= e2 >= e3
        e1, e2, e3 = evals
        total_var = float(e1 + e2 + e3 + 1e-12)

        # legacy ratios
        var_pc1   = float(e1 / total_var)
        planarity = float(e2 / (e1 + 1e-12))
        sphericity_ratio = float(e3 / (e1 + 1e-12))

        # ===== Added cheap & reliable sphericity/isotropy features =====
        lam = evals.astype(float)
        lbar = lam.mean()
        fa = float(np.sqrt(1.5) * np.linalg.norm(lam - lbar) / (np.linalg.norm(lam) + 1e-12))
        L = float((lam[0] - lam[1]) / (lam[0] + 1e-12))
        P = float((lam[1] - lam[2]) / (lam[0] + 1e-12))
        S = float(lam[2] / (lam[0] + 1e-12))
        kappa2 = float(1.0 - (3.0 * (lam[0]*lam[1] + lam[1]*lam[2] + lam[2]*lam[0])) /
                       ((lam.sum())**2 + 1e-12))

        # Axis power balance (variance similarity across axes)
        var_axes = np.array([np.var(sx), np.var(sy), np.var(sz)]) + 1e-12
        iso_power = float(3.0 * var_axes.min() / var_axes.sum())

        # Directional isotropy (unit vectors), mean-centered
        A = np.vstack([sx, sy, sz]).T
        A0 = A - A.mean(axis=0, keepdims=True)            # <-- key change
        norms = np.linalg.norm(A0, axis=1)
        mask = norms > 1e-6
        if np.any(mask):
            U = A0[mask] / norms[mask, None]
            # Optional weighting by magnitude before unit normalization
            w = norms[mask] + 1e-12
            U_weighted_mean = (U * w[:, None]).sum(axis=0) / w.sum()
            R_dir = float(np.linalg.norm(U_weighted_mean))
            sph_var = float(2.0 * (1.0 - R_dir))
        else:
            R_dir = np.nan
            sph_var = np.nan
        


        # ===============================================================

        rows.append({
            "t_start_s": s / fs,
            "t_center_s": (s + win/2) / fs,
            "t_end_s": e / fs,

            # intensity
            "mean_x": mean_x, "mean_y": mean_y, "mean_z": mean_z,
            "std_x": std_x,   "std_y": std_y,   "std_z": std_z,
            "mean_vm": mean_vm, "std_vm": std_vm,
            "sma": sma, "rms_vm": rms_vm, "p2p_vm": p2p_vm,
            "jerk_rms": jerk_rms, "jerk_mean_abs": jerk_mean_abs,

            # spectral & rhythm
            **spec,
            "ac_peak_lag_s": ac_lag_s, "ac_peak_val": ac_val,

            # inter-axis coupling (within-person)
            "corr_xy": corr_xy, "corr_yz": corr_yz, "corr_zx": corr_zx,

            # geometry (legacy)
            "pca_var_pc1": var_pc1,
            "pca_planarity": planarity,
            "pca_sphericity": sphericity_ratio,

            # new sphericity/isotropy (cheap & robust)
            "fa": fa,               # fractional anisotropy (↓ spherical)
            "shape_L": L,           # linearity
            "shape_P": P,           # planarity
            "shape_S": S,           # sphericity (~1/3 for perfect sphere)
            "asphericity_kappa2": kappa2,   # ↓ spherical
            "iso_power": iso_power,         # ↑ spherical (≈1 if equal power)
            "dir_resultant_R": R_dir,       # ↓ spherical
            "spherical_variance": sph_var,  # ↑ spherical
        })

    return pd.DataFrame(rows)

def accel_features_total(
    total,
    fs=100.0,
    window_sec=5.0,
    hop_sec=2.5,
    hp_hz=0.25,         # high-pass (set None/0 to disable)
    lp_hz=15.0,         # low-pass
    band_a=(0.5, 2.0),  # slow sway/steps (adapt to your task)
    band_b=(2.0, 5.0),  # faster gestures/steps
):
    """
    Windowed features from a single total-acceleration (magnitude) stream.

    Returns a pandas DataFrame with:
      - intensity: mean_mag, std_mag, rms_mag, p2p_mag, iqr_mag, mad_mag, cv_mag
      - jerk: jerk_rms, jerk_mean_abs
      - spectrum: dom_freq_hz, dom_power_frac, spec_entropy, spec_centroid_hz,
                  spec_spread_hz, band_a_frac, band_b_frac
      - periodicity: ac_peak_lag_s, ac_peak_val
      - zero_cross_rate
      - robust_outlier_rate (|zscore|>3, robust using MAD)
    """
    m = np.asarray(total, dtype=float).copy()
    n = len(m)
    if n < 3:
        raise ValueError("total must have length >= 3")

    # ---- filtering (zero-phase) ----
    def butter_filter(sig, fs, hp, lp):
        sos_list = []
        if hp is not None and hp > 0:
            sos_list.append(signal.butter(2, hp/(fs/2), btype='highpass', output='sos'))
        if lp is not None and lp > 0:
            sos_list.append(signal.butter(4, lp/(fs/2), btype='lowpass', output='sos'))
        out = sig
        for sos in sos_list:
            out = signal.sosfiltfilt(sos, out)
        return out

    m_f = butter_filter(m, fs, hp_hz, lp_hz)

    # ---- windowing ----
    win = int(round(window_sec * fs))
    hop = int(round(hop_sec * fs))
    if win <= 1 or hop <= 0 or win > n:
        raise ValueError("Check fs/window_sec/hop_sec — invalid window/hop sizes.")
    starts = np.arange(0, n - win + 1, hop)

    # ---- helpers ----
    def spectral_stats(sig, fs):
        sig_d = sig - np.mean(sig)
        if np.allclose(sig_d, 0):
            return dict(dom_freq_hz=np.nan, dom_power_frac=np.nan,
                        spec_entropy=np.nan, spec_centroid_hz=np.nan,
                        spec_spread_hz=np.nan, band_a_frac=np.nan, band_b_frac=np.nan)
        win = np.hanning(len(sig_d))
        fft = np.fft.rfft(sig_d * win)
        freqs = np.fft.rfftfreq(len(sig_d), d=1/fs)
        power = (fft.real**2 + fft.imag**2)

        # remove DC from dominance calc
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

    def autocorr_first_peak(sig, fs, min_period=0.2, max_period=3.0):
        s = sig - np.mean(sig)
        if np.allclose(s, 0):
            return np.nan, np.nan
        acf_full = signal.correlate(s, s, mode='full')
        acf = acf_full[acf_full.size // 2:]
        acf /= (acf[0] + 1e-12)
        lags = np.arange(len(acf)) / fs
        lo = int(np.ceil(min_period * fs))
        hi = min(int(np.floor(max_period * fs)), len(acf) - 1)
        if hi <= lo:
            return np.nan, np.nan
        k = lo + int(np.argmax(acf[lo:hi+1]))
        return float(lags[k]), float(acf[k])

    def zero_cross_rate(sig):
        s = sig - np.mean(sig)
        return float(np.mean(np.abs(np.diff(np.signbit(s))).astype(float)))

    rows = []
    for s in starts:
        e = s + win
        w = m_f[s:e]

        # ----- intensity & distribution (cheap, robust)
        mean_mag = float(np.mean(w))
        std_mag  = float(np.std(w))
        rms_mag  = float(np.sqrt(np.mean(w**2)))
        p2p_mag  = float(np.max(w) - np.min(w))
        q25, q75 = np.percentile(w, [25, 75])
        iqr_mag  = float(q75 - q25)
        mad_mag  = float(np.median(np.abs(w - np.median(w))) * 1.4826)  # robust σ
        cv_mag   = float(std_mag / (np.abs(mean_mag) + 1e-12))          # coefficient of variation

        # ----- jerk (from magnitude)
        if len(w) > 1:
            jerk = np.diff(w) * fs
            jerk_rms = float(np.sqrt(np.mean(jerk**2)))
            jerk_mean_abs = float(np.mean(np.abs(jerk)))
        else:
            jerk_rms = np.nan
            jerk_mean_abs = np.nan

        # ----- spectrum
        spec = spectral_stats(w, fs)

        # ----- periodicity & rate
        ac_lag_s, ac_val = autocorr_first_peak(w, fs)
        zcr = zero_cross_rate(w)

        # ----- robust outlier rate (quick motion spikes)
        # z-score with robust sigma
        if mad_mag > 1e-12:
            zrob = (w - np.median(w)) / mad_mag
            out_rate = float(np.mean(np.abs(zrob) > 3.0))
        else:
            out_rate = 0.0

        rows.append(dict(
            t_start_s = s / fs,
            t_center_s = (s + win/2) / fs,
            t_end_s = e / fs,

            mean_mag = mean_mag,
            std_mag = std_mag,
            rms_mag = rms_mag,
            p2p_mag = p2p_mag,
            iqr_mag = iqr_mag,
            mad_mag = mad_mag,
            cv_mag = cv_mag,

            jerk_rms = jerk_rms,
            jerk_mean_abs = jerk_mean_abs,

            **spec,

            ac_peak_lag_s = ac_lag_s,
            ac_peak_val = ac_val,
            zero_cross_rate = zcr,
            robust_outlier_rate = out_rate,
        ))

    return pd.DataFrame(rows)
