import numpy as np


class DriftCorrector:
    """Threshold-triggered single-frame stuff/drop drift correction.

    When the consumer's ring `fill` drifts outside `target_fill ± deadband`,
    one frame is dropped (overfull) or duplicated (underfull) at the block's
    nearest-to-zero sample so the ~20 us edit is inaudible. Inside the deadband
    the block passes through unchanged (same object). Returns `len(block) -/+ 1`
    on an edit.
    """

    def __init__(self, target_fill: int, *, deadband: int = 64) -> None:
        self.target_fill = int(target_fill)
        self.deadband = int(deadband)

    def correct(self, block: np.ndarray, fill: int) -> np.ndarray:
        if block.shape[0] == 0:
            return block  # nothing to stuff/drop; _zero_crossing is undefined on empty
        if fill > self.target_fill + self.deadband:
            idx = self._zero_crossing(block)
            return np.delete(block, idx, axis=0)
        if fill < self.target_fill - self.deadband:
            idx = self._zero_crossing(block)
            return np.insert(block, idx, block[idx], axis=0)
        return block

    @staticmethod
    def _zero_crossing(block: np.ndarray) -> int:
        # Frame closest to zero energy -> the least audible place to edit.
        mono = block.mean(axis=1) if block.ndim == 2 else block
        return int(np.argmin(np.abs(mono)))
