import time
from collections import deque

import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, FloatParam, IntParam, StringParam


class DreamInceptor(Node):
    """
    This node monitors an incoming EEG signal and detects specific brain activity patterns suitable for targeted dream intervention (dream inception). Depending on configuration, it operates in two modes: (1) theta/alpha z-score detection, using baseline statistics to monitor significant changes in spectral ratios or signal complexity; or (2) hypnodensity-based detection, leveraging a deep learning model to compute state probabilities and entropy for distinguishing sleep stages. When detection criteria are met, the node outputs a trigger signal indicating the optimal moment for dream incubation, along with relevant feature data for tracking and analysis.

    Inputs:
    - data: One-dimensional EEG time series data used for ongoing analysis and detection.
    - start: Optional control input to initiate the dream inception detection process.
    - reset: Optional control input to abort and reset the detection process and baseline.

    Outputs:
    - trigger: Array output (single value) set to 1 when detection criteria are met (indicating to trigger dream incubation), or 0 when baseline collection is complete, otherwise None.
    - z_theta_alpha: Array output with the current z-scored theta/alpha ratio for the EEG segment (only in theta_alpha mode).
    - z_lempel_ziv: Array output with the current z-scored Lempel-Ziv complexity for the EEG segment (only in theta_alpha mode).
    - baseline_stats: Table of mean or quantile baseline statistics for theta/alpha and Lempel-Ziv values (provided during and after baseline collection; only available in theta_alpha mode).
    - hypnodensities: Array output with 6 values per window: 5-element hypnodensity state probabilities (from a neural network classifier) plus normalized entropy, updated each window (only in hypnodensity mode).
    """

    def config_input_slots():
        return {"data": DataType.ARRAY, "start": DataType.ARRAY, "reset": DataType.ARRAY}

    def config_output_slots():
        return {
            "trigger": DataType.ARRAY,
            "z_theta_alpha": DataType.ARRAY,
            "z_lempel_ziv": DataType.ARRAY,
            "baseline_stats": DataType.TABLE,
            "hypnodensities": DataType.ARRAY,  # <-- New output: 6 floats per window
        }

    def config_params():
        return {
            "control": {
                "start": BoolParam(False, trigger=True, doc="Start the dream inception process"),
                "reset": BoolParam(False, trigger=True, doc="Reset and stop the process"),
                "wait_time": IntParam(100, 15, 300, doc="Waiting time after trigger in seconds"),
                "detection_method": StringParam(
                    "theta_alpha",
                    options=["theta_alpha", "hypnodensity"],
                    doc="Select the detection method: theta/alpha z-score or hypnodensity entropy",
                ),
            },
            "baseline": {
                "n_seconds": IntParam(30, 15, 300, doc="Baseline duration in seconds"),
                "method": StringParam("mean", options=["mean", "quantile"], doc="Baseline computation method"),
            },
            "features": {
                "n_features": IntParam(100, 5, 500, doc="Number of feature values to accumulate"),
                "lz_binarization": StringParam("mean", options=["mean", "median"], doc="LZ binarization method"),
            },
            "feature_detection": {  # <-- Renamed from 'detection'
                "threshold": FloatParam(2.0, 0.5, 5.0, doc="Theta/alpha z-score threshold"),
                "n_windows": IntParam(20, 5, 100, doc="Number of successive windows required"),
            },
            "hypnodensity_detection": {  # <-- New panel for hypnodensity detection
                "entropy_threshold": FloatParam(0.2, 0.01, 1.0, doc="Entropy threshold for hypnodensity-based detection (0–1)"),
                "n_windows": IntParam(20, 5, 100, doc="Number of successive windows required for hypnodensity detection"),
                "fmin": FloatParam(1.0, 0.1, 10.0, doc="Low frequency cutoff for bandpass filter"),
                "fmax": FloatParam(30.0, 10.0, 50.0, doc="High frequency cutoff for bandpass filter"),
            },
        }

    def setup(self):
        # Import required libraries
        from antropy import lziv_complexity
        from scipy import signal
        from gssc.infer import ArrayInfer  # <-- Add this import

        self.compute_lzc = lziv_complexity
        self.signal = signal
        self.last_trigger_time = None

        # Add GSSC ArrayInfer instance (set use_cuda=True if you want GPU)
        self.array_infer = ArrayInfer(use_cuda=False)

        # Initialize state variables
        self.reset_state()

        # read the start value once to avoid starting before the button is pressed
        self.params.control.start.value

        self.gssc_hidden = None
        self.hidden_iteration = 0

    def reset_state(self):
        """Reset all internal state variables"""
        self.is_running = False
        self.last_trigger_time = None
        self.baseline_data = []
        self.baseline_computed = False
        self.baseline_stats = {}
        self.feature_buffer = deque(maxlen=self.params.features.n_features.value if hasattr(self, "params") else 10)
        self.successive_count = 0
        self.time_origin = None
        self.hypnodensity_buffer = deque(
            maxlen=self.params.hypnodensity_detection.n_windows.value if hasattr(self, "params") else 10
        )
        self.gssc_hidden = None  # Reset hidden state
        self.hidden_iteration = 0

    def process(self, data: Data, start: Data = None, reset: Data = None):
        if start is not None:
            self.params.control.start.value = True
            self.input_slots["start"].clear()
        if reset is not None:
            self.params.control.reset.value = True
            self.input_slots["reset"].clear()

        if data is None or data.data is None:
            print("No data received")
            return None

        # Handle control parameters
        if self.params.control.reset.value:
            print("Resetting DreamInceptor state")
            self.reset_state()
            return {
                "trigger": None,
                "z_theta_alpha": None,
                "z_lempel_ziv": None,
                "baseline_stats": None,
                "hypnodensities": None,
            }

        if self.params.control.start.value and not self.is_running:
            self.is_running = True
            self.time_origin = time.time()
            self.baseline_data = []
            self.baseline_computed = False

        if not self.is_running:
            print("DreamInceptor is not running. Waiting for start signal.")
            return {
                "trigger": None,
                "z_theta_alpha": None,
                "z_lempel_ziv": None,
                "baseline_stats": None,
                "hypnodensities": None,
            }

        eeg_signal = np.asarray(data.data)
        assert eeg_signal.ndim == 1, "Expected 1d time series"

        send_trigger = None
        detection_method = self.params.control.detection_method.value
        if detection_method == "theta_alpha":
            # Phase 1: Baseline computation (first minute)
            if not self.baseline_computed:
                elapsed_time = time.time() - self.time_origin

                if elapsed_time < self.params.baseline.n_seconds.value:
                    # Still collecting baseline data
                    self.baseline_data.extend(eeg_signal)
                    return {
                        "trigger": None,
                        "z_theta_alpha": None,
                        "z_lempel_ziv": None,
                        "baseline_stats": None,
                        "hypnodensities": None,
                    }
                else:
                    # Compute baseline statistics
                    self._compute_baseline_stats()
                    self.baseline_computed = True
                    send_trigger = np.array(0), data.meta  # 0 means baseline finished

            # Phase 2: Feature extraction and detection
            if self.baseline_computed and len(self.baseline_data) > 0:
                # Extract features from current window
                lz_complexity = self._compute_lempel_ziv(eeg_signal)
                theta_alpha_ratio = self._compute_theta_alpha_ratio(eeg_signal, data.meta)

                # Compute z-scores using baseline
                lz_zscore = self._compute_zscore(lz_complexity, "lz")
                ta_zscore = self._compute_zscore(theta_alpha_ratio, "theta_alpha")

                # Add features to buffer
                self.feature_buffer.append({"lz_zscore": lz_zscore, "ta_zscore": ta_zscore})

                # Check detection criteria
                detected = self._check_detection_criteria()

                # --- Trigger cooldown logic ---
                wait_time = self.params.control.wait_time.value
                now = time.time()
                if detected:
                    if (self.last_trigger_time is None) or ((now - self.last_trigger_time) >= wait_time):
                        send_trigger = np.array(1), data.meta  # 1 means incubation triggered
                        self.last_trigger_time = now  # reset cooldown
                    else:
                        send_trigger = None  # within cooldown window

                baseline_stats_table = {
                    "lz_mean": Data(DataType.ARRAY, np.array([self.baseline_stats["lz"]["mean"]]), {}),
                    "lz_std": Data(DataType.ARRAY, np.array([self.baseline_stats["lz"]["std"]]), {}),
                    "ta_mean": Data(DataType.ARRAY, np.array([self.baseline_stats["theta_alpha"]["mean"]]), {}),
                    "ta_std": Data(DataType.ARRAY, np.array([self.baseline_stats["theta_alpha"]["std"]]), {}),
                }

                return {
                    "trigger": send_trigger,
                    "z_theta_alpha": (np.array([ta_zscore]), data.meta),
                    "z_lempel_ziv": (np.array([lz_zscore]), data.meta),
                    "baseline_stats": (baseline_stats_table, data.meta),
                    "hypnodensities": None,  # No hypnodensity output for theta/alpha method
                }
        elif detection_method == "hypnodensity":

            current_time = time.time()
            if not hasattr(self, "last_hidden_reset"):
                self.last_hidden_reset = current_time

            if current_time - self.last_hidden_reset > 10:  # Reset every 5 minutes
                self.gssc_hidden = None
                self.last_hidden_reset = current_time

            fmin = self.params.hypnodensity_detection.fmin.value
            fmax = self.params.hypnodensity_detection.fmax.value
            global_mean = np.nanmean(eeg_signal)
            global_std = np.nanstd(eeg_signal)

            # 2. Compute hypnodensity and entropy with context & accumulated normalization
            # probs, entropy, self.gssc_hidden = compute_hypnodensity_entropy_single_with_context_accum(
            #     eeg_signal,
            #     original_sampling_rate=data.meta.get("sfreq", 256) if hasattr(data, "meta") and data.meta else 256,
            #     array_infer=self.array_infer,
            #     hidden=self.gssc_hidden,
            #     global_mean=global_mean,
            #     global_std=global_std,
            # )
            probs, entropy, self.gssc_hidden = compute_hypnodensity_entropy_single(
                eeg_signal,
                original_sampling_rate=data.meta.get("sfreq", 256) if hasattr(data, "meta") and data.meta else 256,
                array_infer=self.array_infer,
                hidden=None,
                l_freq=fmin,
                h_freq=fmax,
            )

            # self.hidden_iteration += 1
            # if self.hidden_iteration == 900:
            #     self.gssc_hidden = hidden
            #     self.hidden_iteration = 0

            # print('HIDDEN SHAPE', self.gssc_hidden.shape)
            # Append to buffer for successive window logic
            self.hypnodensity_buffer.append(entropy)
            # Prepare the hypnodensity output (5 probs + entropy)
            hypnodensity_output = np.concatenate([probs, [entropy]])

            # Successive windows logic
            entropy_threshold = self.params.hypnodensity_detection.entropy_threshold.value
            n_windows = self.params.hypnodensity_detection.n_windows.value

            # Count how many of the most recent N windows are above threshold (must be contiguous from most recent)
            successive_above = 0
            for ent in reversed(self.hypnodensity_buffer):
                if ent > entropy_threshold:
                    successive_above += 1
                else:
                    break

            detected = successive_above >= n_windows

            # Cooldown logic (re-use wait_time and last_trigger_time)
            wait_time = self.params.control.wait_time.value
            now = time.time()
            send_trigger = None
            if detected:
                if (self.last_trigger_time is None) or ((now - self.last_trigger_time) >= wait_time):
                    send_trigger = np.array(1), data.meta  # 1 means incubation triggered
                    self.last_trigger_time = now
                else:
                    send_trigger = None

            return {
                "trigger": send_trigger,
                "z_theta_alpha": None,
                "z_lempel_ziv": None,
                "baseline_stats": None,
                "hypnodensities": (hypnodensity_output, data.meta),
            }

    def _compute_baseline_stats(self):
        """Compute baseline statistics for z-score normalization"""
        baseline_array = np.array(self.baseline_data)

        # Compute Lempel-Ziv complexity for baseline
        lz_values = []
        window_size = min(1000, len(baseline_array) // 10)  # Adaptive window size

        for i in range(0, len(baseline_array) - window_size, window_size // 2):
            window = baseline_array[i : i + window_size]
            lz_val = self._compute_lempel_ziv(window)
            lz_values.append(lz_val)

        # Compute theta/alpha ratios for baseline
        ta_values = []
        for i in range(0, len(baseline_array) - window_size, window_size // 2):
            window = baseline_array[i : i + window_size]
            ta_val = self._compute_theta_alpha_ratio(window, {})
            ta_values.append(ta_val)

        # Store baseline statistics
        if self.params.baseline.method.value == "mean":
            self.baseline_stats = {
                "lz": {"mean": np.nanmean(lz_values), "std": np.nanstd(lz_values)},
                "theta_alpha": {"mean": np.nanmean(ta_values), "std": np.nanstd(ta_values)},
            }
        else:  # quantile method
            self.baseline_stats = {
                "lz": {"q25": np.percentile(lz_values, 25), "q75": np.percentile(lz_values, 75)},
                "theta_alpha": {"q25": np.percentile(ta_values, 25), "q75": np.percentile(ta_values, 75)},
            }

    def _compute_lempel_ziv(self, signal_data):
        """Compute Lempel-Ziv complexity"""
        if len(signal_data) == 0:
            return 0.0

        # Binarize signal
        if self.params.features.lz_binarization.value == "mean":
            binarized = signal_data > np.nanmean(signal_data)
        else:  # median
            binarized = signal_data > np.nanmedian(signal_data)

        # Compute LZ complexity
        try:
            lzc = self.compute_lzc(binarized, normalize=True)
            return float(lzc)
        except:
            return 0.0

    def _compute_theta_alpha_ratio(self, signal_data, meta):
        """Compute theta/alpha power ratio"""
        if len(signal_data) < 100:  # Need minimum samples for FFT
            print("Data too short")
            return 0.0

        signal_data = signal_data[~np.isnan(signal_data)]
        # Get sampling frequency from metadata or use default
        fs = meta.get("sfreq", 256.0) if isinstance(meta, dict) else 256.0
        # Compute power spectral density
        freqs, psd = self.signal.welch(
            signal_data, fs=fs, nperseg=min(512, len(signal_data) // 4), noverlap=min(400, len(signal_data) // 5)
        )
        # Define frequency bands
        theta_band = (4, 8)
        alpha_band = (8, 12)

        # Extract power in each band
        theta_mask = (freqs >= theta_band[0]) & (freqs <= theta_band[1])
        alpha_mask = (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])

        theta_power = np.nansum(psd[theta_mask])
        alpha_power = np.nansum(psd[alpha_mask])
        # Compute ratio (avoid division by zero)
        if alpha_power > 1e-10:
            return theta_power / alpha_power
        else:
            return 0.0

    def _compute_zscore(self, value, feature_type):
        """Compute z-score using baseline statistics"""
        if feature_type not in self.baseline_stats:
            return 0.0

        stats = self.baseline_stats[feature_type]

        if self.params.baseline.method.value == "mean":
            mean = stats["mean"]
            std = stats["std"]
            return (value - mean) / (std + 1e-8)
        else:  # quantile method
            q25, q75 = stats["q25"], stats["q75"]
            iqr = q75 - q25
            median = (q25 + q75) / 2
            return (value - median) / (iqr + 1e-8)

    def _check_detection_criteria(self):
        """Check if detection criteria are met"""
        if len(self.feature_buffer) < self.params.feature_detection.n_windows.value:
            return 0

        # Check last n_windows for threshold crossing
        recent_features = list(self.feature_buffer)[-self.params.feature_detection.n_windows.value :]
        threshold = self.params.feature_detection.threshold.value

        # Count successive windows above threshold
        successive_above = 0
        for features in reversed(recent_features):
            if features["ta_zscore"] > threshold:
                successive_above += 1
            else:
                break

        if successive_above >= self.params.feature_detection.n_windows.value:
            return 1
        else:
            return 0


def compute_hypnodensity_entropy_single(
    signal_1d,
    original_sampling_rate,
    array_infer,
    hidden=None,
    target_sampling_rate=85.33333,
    l_freq=1,
    h_freq=30.0,
    segment_start=None,  # Start index in seconds (for multi-window, else None)
    segment_length=30,  # Length of segment in seconds
):
    """
    Processes the input EEG, returns:
      probabilities: ndarray of shape (5,)
      entropy: float (normalized)
      hidden: hidden state for model context (if needed)

    - Normalization is done using global mean/std of filtered/resampled full signal.
    - segment_start: optional, if you want a moving window (in seconds).
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scipy import signal as scipy_signal
    import mne

    # 1. Remove NaNs
    signal_1d = np.asarray(signal_1d)
    signal_1d = signal_1d[~np.isnan(signal_1d)]

    # 2. Resample
    if original_sampling_rate != target_sampling_rate:
        num_samples = int(len(signal_1d) * target_sampling_rate / original_sampling_rate)
        resampled_signal = scipy_signal.resample(signal_1d, num_samples)
    else:
        resampled_signal = signal_1d

    # 3. Bandpass filter (full signal!)
    info = mne.create_info(ch_names=["EEG"], sfreq=target_sampling_rate, ch_types=["eeg"])
    raw_mne = mne.io.RawArray(resampled_signal[np.newaxis, :], info, verbose=False)
    raw_mne.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin", verbose=False)
    filtered_signal = raw_mne.get_data()[0] * 1e6  # µV

    filtered_signal = reject_outliers(filtered_signal, threshold=5.0)

    # Optional: interpolate NaNs (for short artifacts), or just remove NaNs in normalization:
    filtered_signal = filtered_signal[~np.isnan(filtered_signal)]

    # 4. Compute global mean/std (of entire filtered signal!)
    global_mean = np.nanmean(filtered_signal)
    global_std = np.nanstd(filtered_signal)

    # 5. Select window for current segment
    target_samples = int(target_sampling_rate * segment_length)
    if segment_start is None:
        # Use the last window (for real-time/online use)
        epoch_data = filtered_signal[-target_samples:]
        if len(epoch_data) < target_samples:
            # Pad if needed
            epoch_data = np.pad(epoch_data, (target_samples - len(epoch_data), 0), mode="edge")
    else:
        # Use a specific window (for offline/batch processing)
        start_idx = int(segment_start * target_sampling_rate)
        end_idx = start_idx + target_samples
        if end_idx > len(filtered_signal):
            epoch_data = filtered_signal[start_idx:]
            # Pad to reach target_samples
            epoch_data = np.pad(epoch_data, (0, target_samples - len(epoch_data)), mode="edge")
        else:
            epoch_data = filtered_signal[start_idx:end_idx]

    # 6. Normalize using global mean/std
    if global_std > 1e-10:
        epoch_data_z = (epoch_data - global_mean) / global_std
    else:
        epoch_data_z = epoch_data * 0.0

    # 7. Inference
    eeg_tensor = torch.tensor(epoch_data_z[np.newaxis, np.newaxis, :], dtype=torch.float32)
    sigs = {"eeg": eeg_tensor}
    logits, nocontext_logits, hidden = array_infer.infer(sigs, hidden=None)
    probabilities = F.softmax(nocontext_logits, dim=1)[0].detach().cpu().numpy()

    # 8. Entropy
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
    entropy_normalized = entropy / np.log(5)

    return probabilities, entropy_normalized, hidden


# def compute_hypnodensity_entropy_single_with_context(
#     signal_1d,
#     original_sampling_rate,
#     array_infer,
#     hidden=None,
#     target_sampling_rate=85.33333,
#     l_freq=1,  # Changed to match GSSC training
#     h_freq=30.0
# ):
#     """
#     Takes a single segment, maintains hidden state for temporal context
#     Returns: probabilities, entropy, updated_hidden
#     """
#     import numpy as np
#     import torch
#     import torch.nn.functional as F
#     from scipy import signal as scipy_signal
#     import mne
#     from gssc.utils import epo_arr_zscore

#     # Remove NaNs
#     signal_1d = np.asarray(signal_1d)
#     signal_1d = signal_1d[~np.isnan(signal_1d)]

#     # For real-time processing, you might want to pad/truncate to expected length
#     target_samples = int(target_sampling_rate * 30)  # 30 seconds worth

#     # Step 1: Resample to target sampling rate if needed
#     if original_sampling_rate != target_sampling_rate:
#         num_samples = int(len(signal_1d) * target_sampling_rate / original_sampling_rate)
#         resampled_data = scipy_signal.resample(signal_1d, num_samples)
#     else:
#         resampled_data = signal_1d

#     # Ensure we have exactly the right number of samples
#     if len(resampled_data) > target_samples:
#         resampled_data = resampled_data[:target_samples]
#     elif len(resampled_data) < target_samples:
#         # Pad with zeros or repeat last value
#         pad_length = target_samples - len(resampled_data)
#         resampled_data = np.pad(resampled_data, (0, pad_length), mode='edge')

#     # Step 2: Bandpass filter (0.3-30 Hz to match GSSC training)
#     info = mne.create_info(ch_names=['EEG'], sfreq=target_sampling_rate, ch_types=['eeg'])
#     raw_mne = mne.io.RawArray(resampled_data[np.newaxis, :], info)
#     raw_mne.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin')
#     filtered_data = raw_mne.get_data()[0]

#     # Step 3: Microvolts and z-score normalization
#     epoch_data = filtered_data * 1e6  # to microvolts
#     epoch_data_z = epo_arr_zscore(epoch_data[np.newaxis, np.newaxis, :])[0, 0]

#     # Step 4: Inference with maintained hidden state
#     eeg_tensor = torch.tensor(epoch_data_z[np.newaxis, np.newaxis, :], dtype=torch.float32)
#     sigs = {"eeg": eeg_tensor}

#     # Pass the hidden state to maintain temporal context
#     logits, nocontext_logits, updated_hidden = array_infer.infer(sigs, hidden=hidden)
#     probabilities = F.softmax(logits, dim=1)[0].detach().cpu().numpy()

#     # Step 5: Entropy (normalized)
#     entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
#     entropy_normalized = entropy / np.log(5)

#     return probabilities, entropy_normalized, updated_hidden


def compute_hypnodensity_entropy_single_with_context_accum(
    signal_1d,
    original_sampling_rate,
    array_infer,
    hidden=None,
    global_mean=None,
    global_std=None,
    target_sampling_rate=85.33333,
    l_freq=2,
    h_freq=30.0,
):
    """
    Takes a single segment, uses accumulated normalization if provided.
    Returns: probabilities, entropy, updated_hidden
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scipy import signal as scipy_signal
    import mne

    # Remove NaNs
    signal_1d = np.asarray(signal_1d)
    signal_1d = signal_1d[~np.isnan(signal_1d)]

    target_samples = int(target_sampling_rate * 30)  # 30 seconds

    # Step 1: Resample to target sampling rate if needed
    if original_sampling_rate != target_sampling_rate:
        num_samples = int(len(signal_1d) * target_sampling_rate / original_sampling_rate)
        resampled_data = scipy_signal.resample(signal_1d, num_samples)
    else:
        resampled_data = signal_1d

    # Ensure we have exactly the right number of samples
    if len(resampled_data) > target_samples:
        resampled_data = resampled_data[:target_samples]
    elif len(resampled_data) < target_samples:
        pad_length = target_samples - len(resampled_data)
        resampled_data = np.pad(resampled_data, (0, pad_length), mode="edge")

    # Step 2: Bandpass filter
    info = mne.create_info(ch_names=["EEG"], sfreq=target_sampling_rate, ch_types=["eeg"])
    raw_mne = mne.io.RawArray(resampled_data[np.newaxis, :], info)
    raw_mne.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin")
    filtered_data = raw_mne.get_data()[0]

    # Step 3: Microvolts conversion
    epoch_data = filtered_data * 1e6  # to microvolts

    # Step 4: Accumulated/global normalization if provided
    if (global_mean is not None) and (global_std is not None) and (global_std > 1e-10):
        epoch_data_z = (epoch_data - global_mean) / global_std
    else:
        # fallback to window normalization if not enough data
        epoch_data_z = (epoch_data - np.mean(epoch_data)) / (np.std(epoch_data) + 1e-10)

    # Step 5: Inference with context
    eeg_tensor = torch.tensor(epoch_data_z[np.newaxis, np.newaxis, :], dtype=torch.float32)
    sigs = {"eeg": eeg_tensor}
    logits, nocontext_logits, updated_hidden = array_infer.infer(sigs, hidden=None)
    probabilities = F.softmax(logits, dim=1)[0].detach().cpu().numpy()

    # Step 6: Entropy (normalized)
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
    entropy_normalized = entropy / np.log(5)

    return probabilities, entropy_normalized, updated_hidden


def reject_outliers(signal, threshold=5.0):
    """Returns a signal where samples >threshold std from mean are replaced with NaN."""
    mean = np.nanmean(signal)
    std = np.nanstd(signal)
    outliers = np.abs(signal - mean) > (threshold * std)
    signal_clean = signal.copy()
    signal_clean[outliers] = np.nan
    return signal_clean
