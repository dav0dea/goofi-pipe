"""LempelZiv — LZ76 complexity of a signal (a common EEG complexity measure).

A stopgap Python node exercising the in-process (pyo3 free-threaded) class-contract
tier. Pure numpy: binarize the input against its mean, count Lempel-Ziv 1976 production
steps, emit a length-1 f32 array.
"""

import goofi
import numpy as np


def _lz76(binary):
    # Kaspar-Schuster LZ76 complexity of a 0/1 sequence.
    n = len(binary)
    if n == 0:
        return 0
    i, k, l, c, kmax = 0, 1, 1, 1, 1
    while l + k <= n:
        if binary[i + k - 1] == binary[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > kmax:
                kmax = k
            i += 1
            if i == l:
                c += 1
                l += kmax
                i = 0
                k = 1
                kmax = 1
            else:
                k = 1
    return c


class LempelZiv(goofi.Node):
    """LZ76 complexity of a signal."""

    # `required`: process() reads `data.data` unconditionally, so a tick with an empty slot cannot
    # work. The engine refuses the tick before process() is entered.
    manifest = goofi.Manifest(
        inputs={"data": goofi.InputSlot(goofi.DataType.ARRAY, required=True)},
        outputs={"out": goofi.DataType.ARRAY},
    )

    def process(self, data):
        x = np.asarray(data.data).ravel()
        b = (x > x.mean()).astype(np.uint8)
        return {"out": np.asarray([float(_lz76(b))], dtype=np.float32)}
