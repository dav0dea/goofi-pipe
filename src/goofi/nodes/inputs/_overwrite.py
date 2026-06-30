import time
from typing import Optional

from goofi.data import Data


class _OverwriteHold:
    """Shared 'hold the latest overwrite input for a timeout, then revert to the constant'
    state machine for the Constant* input nodes.

    A plain mixin, NOT a Node — and the module is underscore-prefixed so the node
    auto-discovery walk skips it (one Node subclass per discovered module). It relies on
    `self.params`/`self.input_slots` provided by the Node it is mixed into.
    """

    def setup_overwrite(self) -> None:
        self.last_overwrite_time = None
        self.last_overwrite_data = None

    def held_override(self, overwrite: Optional[Data]) -> Optional[Data]:
        """Capture a new overwrite, expire a stale one, and return the Data to emit while
        the override holds — or None to fall through to the constant. The caller chooses the
        output meta (ConstantString emits {}, ConstantArray carries the override's own meta).
        A timeout of 0 never expires the override."""
        if overwrite is not None:
            self.last_overwrite_data = overwrite
            self.last_overwrite_time = time.time()
            self.input_slots["overwrite"].clear()

        if self.last_overwrite_data is not None:
            timeout_val = self.params.constant.overwrite_timeout.value
            if timeout_val > 0 and (time.time() - self.last_overwrite_time) > timeout_val:
                self.last_overwrite_data = None
                self.last_overwrite_time = None
            else:
                return self.last_overwrite_data
        return None
