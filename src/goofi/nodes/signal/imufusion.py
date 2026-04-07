import math
from copy import deepcopy
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, StringParam

_DT_MIN = 1e-4
_DT_MAX = 0.1


def _euler_deg_to_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Same convention as Musefusioncube: R = Rz @ Ry @ Rx (roll, pitch, yaw in degrees)."""
    r, p, y = (math.radians(x) for x in (roll_deg, pitch_deg, yaw_deg))
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def _as_channel_time3(a: np.ndarray) -> np.ndarray:
    """Return shape (3, N): rows are X, Y, Z; columns are time samples."""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        if a.size != 3:
            raise ValueError(f"Expected 3 elements for a 1D array, got shape {a.shape}.")
        return a.reshape(3, 1)
    if a.ndim == 2:
        if a.shape[0] == 3:
            return a
        # Common alternate layout: (N, 3) time-major — same as GUI (n_samples, n_channels) style.
        if a.shape[1] == 3:
            return a.T
        raise ValueError(
            f"Expected shape (3, N) or (N, 3) with 3 spatial channels, got {a.shape}. "
            "Rows/columns must include exactly 3 X,Y,Z samples."
        )
    raise ValueError(f"Expected 1D or 2D array, got ndim={a.ndim} shape {a.shape}.")


def _sample_rate_hz(meta: dict) -> Optional[float]:
    """Hz from metadata; supports ``sfreq`` or common typo ``sfreg``."""
    for key in ("sfreq", "sfreg"):
        if key in meta and meta[key] is not None:
            return float(meta[key])
    return None


def _fix_output_meta_channels(meta: dict, out: np.ndarray, fmt: str) -> None:
    """
    Input meta may describe (3,) or (3, N) arrays; output is 1D (3,) or (4,). Align ``channels`` so
    ``Data`` validation does not fail (length per dim must match ``out.shape``).
    """
    if "channels" not in meta or not isinstance(meta["channels"], dict):
        meta["channels"] = {}
    ch = meta["channels"]
    nd = out.ndim
    for key in list(ch.keys()):
        if key.startswith("dim"):
            d = int(key[3:])
            if d >= nd:
                del ch[key]
    if nd < 1:
        return
    need = int(out.shape[0])
    if f"dim0" not in ch or len(ch["dim0"]) != need:
        if fmt == "euler_deg":
            ch["dim0"] = ["roll", "pitch", "yaw"]
        else:
            ch["dim0"] = ["x", "y", "z", "w"]


class ImuFusion(Node):
    """
    Complementary fusion of 3-axis accelerometer and gyroscope samples into orientation.

    Inputs are `acc` and `gyro` arrays: shape `(3,)` for a single sample, or `(3, N)` (or `(N, 3)`)
    with rows X, Y, Z and `N` time samples. If the two inputs have different `N` (e.g. different buffer sizes),
    the node uses the last `min(N_acc, N_gyro)` samples from each so fusion keeps running without errors.

    If metadata includes `sfreq` (or `sfreg`), per-step `dt` is `1/sfreq` for each sample in the chunk;
    otherwise `dt = 1/UPDATE_HZ`. Accelerometer corrects roll/pitch; yaw follows `gz` only (drifts without a magnetometer).

    Gyroscope components are assumed to be **degrees per second**.

    Outputs either Euler angles `[roll, pitch, yaw]` in degrees or a quaternion `[x, y, z, w]` (scalar-last)
    matching the rotation `Rz @ Ry @ Rx` used in the Musefusioncube example.
    """

    def config_input_slots():
        return {"acc": DataType.ARRAY, "gyro": DataType.ARRAY}

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def config_params():
        return {
            "fusion": {
                "ALPHA": FloatParam(
                    0.97,
                    0.0,
                    1.0,
                    doc="Blend factor: higher trusts integrated gyro more for roll/pitch.",
                ),
                "UPDATE_HZ": FloatParam(
                    30.0,
                    0.1,
                    2000.0,
                    doc="Fallback rate if input metadata has no sfreq: dt = 1/UPDATE_HZ per sample.",
                ),
                "output_format": StringParam(
                    "euler_deg",
                    options=["euler_deg", "quaternion_xyzw"],
                    doc="euler_deg: shape (3,) roll,pitch,yaw in degrees; quaternion_xyzw: shape (4,) x,y,z,w.",
                ),
            }
        }

    def setup(self):
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

    def _fusion_step(self, ax: float, ay: float, az: float, gx: float, gy: float, gz: float, dt: float) -> None:
        alpha = float(self.params.fusion.ALPHA.value)
        self.roll += gx * dt
        self.pitch += gy * dt
        self.yaw += gz * dt
        a_roll = math.degrees(math.atan2(ay, az))
        a_pitch = math.degrees(math.atan2(-ax, math.sqrt(ay * ay + az * az)))
        self.roll = alpha * self.roll + (1.0 - alpha) * a_roll
        self.pitch = alpha * self.pitch + (1.0 - alpha) * a_pitch

    def process(self, acc: Data, gyro: Data):
        if acc is None or gyro is None:
            return None

        # Standalone `process()` calls do not run `setup()` (unlike `__call__` / graph execution).
        if not hasattr(self, "roll"):
            self.setup()

        acc_a = _as_channel_time3(acc.data)
        gyro_a = _as_channel_time3(gyro.data)
        n_acc, n_gyro = acc_a.shape[1], gyro_a.shape[1]
        if n_acc != n_gyro:
            # Pipelines often deliver different buffer lengths per input; fuse the overlapping tail so
            # both sensors stay time-aligned (same end index).
            n = min(n_acc, n_gyro)
            acc_a = acc_a[:, -n:]
            gyro_a = gyro_a[:, -n:]
        _, n = acc_a.shape
        if n == 0:
            return None
        sf = _sample_rate_hz(acc.meta) or _sample_rate_hz(gyro.meta)
        if sf is not None and sf > 0:
            dt = 1.0 / sf
        else:
            hz = float(self.params.fusion.UPDATE_HZ.value)
            if hz <= 0:
                raise ValueError("UPDATE_HZ must be positive when sfreq is absent from metadata.")
            dt = 1.0 / hz
        dt = max(_DT_MIN, min(_DT_MAX, dt))

        for t in range(n):
            ax, ay, az = float(acc_a[0, t]), float(acc_a[1, t]), float(acc_a[2, t])
            gx, gy, gz = float(gyro_a[0, t]), float(gyro_a[1, t]), float(gyro_a[2, t])
            self._fusion_step(ax, ay, az, gx, gy, gz, dt)

        fmt = self.params.fusion.output_format.value
        meta = deepcopy(acc.meta)
        if fmt == "euler_deg":
            out = np.array([self.roll, self.pitch, self.yaw], dtype=float)
        elif fmt == "quaternion_xyzw":
            rmat = _euler_deg_to_matrix(self.roll, self.pitch, self.yaw)
            quat = Rotation.from_matrix(rmat).as_quat()
            out = np.asarray(quat, dtype=float)
        else:
            raise ValueError(f"Unknown output_format: {fmt!r}")

        meta["shape"] = out.shape
        _fix_output_meta_channels(meta, out, fmt)
        return {"out": (out, meta)}
