"""Image dtype coercion helpers (A0 uint8-image convention).

Image producers emit uint8 [0,255] (4x less wire data than float32/255).
Consumers that need floats call `as_float01`; consumers that feed cv2/PIL call
`as_uint8`. Both accept either representation, so a node works regardless of
which a producer hands it.

Kept dependency-light (numpy only) so lightweight input nodes — VideoStream,
LoadFile — don't pull heavy imports just to coerce a frame.
"""
import numpy as np


def as_uint8(img: np.ndarray) -> np.ndarray:
    """Coerce an image array to uint8 [0,255].

    uint8 passes through unchanged; float (assumed [0,1]) is rounded, scaled,
    and clipped. Use before handing an image to cv2/PIL, which want uint8.
    """
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        return arr
    return np.clip(np.round(arr.astype(np.float32) * 255.0), 0, 255).astype(np.uint8)


def as_float01(img: np.ndarray) -> np.ndarray:
    """Coerce an image array to float32 [0,1].

    uint8 is scaled down by 255; float is cast through (assumed already
    normalized). Use when a node's math needs floats in [0,1].
    """
    arr = np.asarray(img)
    if np.issubdtype(arr.dtype, np.floating):
        return arr.astype(np.float32)
    return arr.astype(np.float32) / 255.0
