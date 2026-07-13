import numpy as np

INDEX_META_KEY = "index"


def is_discontinuous(prev_index, index) -> bool:
    """True when `index` is not the contiguous successor of `prev_index`.

    Returns False whenever either side is None (no baseline / no stamp).
    """
    return prev_index is not None and index is not None and index != prev_index + 1


def crossfade(prev_tail: np.ndarray, block: np.ndarray, n: int) -> np.ndarray:
    """Equal-power fade over the first `min(n, len(block))` frames.

    Fades from the held previous value `prev_tail[-1]` into `block`, REPLACING
    those frames in place (never prepending) so the timebase length is
    unchanged. Returns a new `(frames, channels)` array the same length as
    `block`; `block` itself is left untouched.
    """
    out = block.copy()
    m = min(int(n), block.shape[0])
    if m <= 0:
        return out
    t = (np.arange(m, dtype=np.float32) + 1.0) / (m + 1.0)
    fade_out = np.cos(t * (np.pi / 2.0))[:, None]  # 1 -> 0
    fade_in = np.sin(t * (np.pi / 2.0))[:, None]   # 0 -> 1
    src = prev_tail[-1][None, :]                    # held frame, (1, channels)
    out[:m] = fade_out * src + fade_in * block[:m]
    return out
