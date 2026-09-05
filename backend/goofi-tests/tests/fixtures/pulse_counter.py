import goofi
import numpy as np


class PulseCounter(goofi.Node):
    """Counts up, and starts over on a pulse."""

    OUTPUTS = {"out": goofi.DataType.ARRAY}
    PRODUCER = True
    PARAMS = {"count": {"reset": goofi.PulseParam(doc="Start the count over.")}}

    def setup(self):
        self.n = 0

    def process(self):
        self.n += 1
        return np.array([float(self.n)], dtype=np.float32)

    def pulse_count_reset(self):
        self.n = 0
