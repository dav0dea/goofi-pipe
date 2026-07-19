"""LempelZiv — LZ76 complexity of a signal (a common EEG complexity measure).

A stopgap Python implementation used to exercise the in-process (pyo3 free-threaded)
node tier; the real node will eventually be native Rust. Pure numpy: binarize the input
against its mean, then count Lempel-Ziv 1976 production steps. Emits a length-1 f32 array.
"""

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


def process(x):
    x = np.asarray(x).ravel()
    b = (x > x.mean()).astype(np.uint8)
    return np.asarray([float(_lz76(b))], dtype=np.float32)
