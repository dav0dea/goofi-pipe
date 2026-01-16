import re

import numpy as np
from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, FloatParam, StringParam


class Hologram(Node):
    """
    Convert channel-wise PSD (channels x freqs) into parametric hologram data
    for volumetric ray marching (TouchDesigner / RayTK).

    Input:
      - psd: ARRAY, shape (N_channels, N_freqs)

    Outputs:
      - pos:  (N, 3) electrode positions (normalized)
      - freq: (F,)   frequencies
      - amp:  (N, F) amplitudes
    """

    # ---------------- Goofi plumbing ----------------

    def config_input_slots():
        return {"psd": DataType.ARRAY}

    def config_output_slots():
        return {
            "pos": DataType.ARRAY,
            "freq": DataType.ARRAY,
            "amp": DataType.ARRAY,
        }

    def config_params():
        return {
            "hologram": {
                "montage": StringParam(
                    "standard_1005",
                    options=["standard_1005", "standard_1020", "standard_alphabetic"],
                ),
                "ch_regex_pattern": StringParam(
                    "", doc="Regex pattern to match in channel names (e.g., 'eeg-' or '^EEG_'). Leave empty to disable."
                ),
                "ch_regex_replace": StringParam(
                    "", doc="Replacement string for matched pattern (e.g., '' to remove). Used only if pattern is set."
                ),
                "pos_space": StringParam("centered", options=["centered", "unit"]),
                "pos_scale": FloatParam(0.9, 0.0, 5.0),
                "log_amp": BoolParam(False),
                "amp_norm": StringParam("l2", options=["none", "l2", "max"]),
                "amp_eps": FloatParam(1e-8, 1e-12, 1e-2),
            }
        }

    # ---------------- helpers ----------------

    @staticmethod
    def _normalize_positions(pos, space, scale):
        valid = np.isfinite(pos).all(axis=1)
        if not np.any(valid):
            return np.zeros_like(pos, dtype=np.float32)

        p = pos[valid]
        vmin, vmax = p.min(axis=0), p.max(axis=0)
        denom = np.where(vmax - vmin > 0, vmax - vmin, 1.0)

        u = (pos - vmin) / denom
        u = (u - 0.5) * scale + 0.5

        if space == "centered":
            return (u - 0.5).astype(np.float32)

        return u.astype(np.float32)

    @staticmethod
    def _condition_amp(a, log_amp, norm, eps):
        a = a.astype(np.float32, copy=False)

        if log_amp:
            a = np.log1p(np.maximum(a, 0.0))

        if norm == "none":
            return a

        if norm == "l2":
            return a / (np.linalg.norm(a) + eps)

        if norm == "max":
            return a / (np.max(np.abs(a)) + eps)

        raise ValueError(f"Unknown amp_norm: {norm}")

    # ---------------- lifecycle ----------------

    def setup(self):
        import mne  # ensure available

    def process(self, psd: Data):
        if psd is None or psd.data is None:
            return None

        amp = np.asarray(psd.data, dtype=np.float32)  # (N, F)
        meta = psd.meta

        # ---- metadata ----
        ch_names = meta["channels"]["dim0"]
        freqs = np.asarray(meta["channels"]["dim1"], dtype=np.float32)

        # ---- apply regex transformation to channel names ----
        pattern = self.params.hologram.ch_regex_pattern.value
        replacement = self.params.hologram.ch_regex_replace.value

        if pattern:
            ch_names_montage = [re.sub(pattern, replacement, ch) for ch in ch_names]
        else:
            ch_names_montage = ch_names

        # ---- electrode positions via montage ----
        import mne

        montage_name = self.params.hologram.montage.value
        montage = mne.channels.make_standard_montage(montage_name)
        pos_dict = montage.get_positions()["ch_pos"]

        pos = np.full((len(ch_names), 3), np.nan, dtype=np.float32)
        for i, ch_montage in enumerate(ch_names_montage):
            if ch_montage in pos_dict:
                pos[i] = pos_dict[ch_montage][[1, 0, 2]]

        # fill missing channels with mean position
        valid = np.isfinite(pos).all(axis=1)
        if np.any(valid) and not np.all(valid):
            pos[~valid] = pos[valid].mean(axis=0)

        pos = self._normalize_positions(
            pos,
            space=self.params.hologram.pos_space.value,
            scale=self.params.hologram.pos_scale.value,
        )

        # ---- amplitude conditioning ----
        amp = self._condition_amp(
            amp,
            log_amp=self.params.hologram.log_amp.value,
            norm=self.params.hologram.amp_norm.value,
            eps=self.params.hologram.amp_eps.value,
        )

        # ---- output metas ----
        pos_meta = {
            **meta,
            "channels": {"dim0": ch_names, "dim1": ["x", "y", "z"]},
        }

        freq_meta = {
            **meta,
            "channels": {"dim0": freqs.tolist()},
        }

        amp_meta = {
            **meta,
            "channels": {"dim0": ch_names, "dim1": freqs.tolist()},
        }

        return {
            "pos": (pos, pos_meta),
            "freq": (freqs, freq_meta),
            "amp": (amp, amp_meta),
        }
