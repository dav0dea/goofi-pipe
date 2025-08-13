from goofi.node import Node
import threading, time, os, sys
import numpy as np

from goofi.data import Data, DataType
from goofi.params import BoolParam, IntParam, StringParam, FloatParam


class ShutdownGoofi(Node):
    def config_params():
        return {
            "shutdown": {
                "delay": FloatParam(0, 0, 300, doc="Shutdown delay in seconds"),
                "shutdown": BoolParam(trigger=True, doc="Shutdown this goofi-pipe instance after the specified delay"),
                "cancel": BoolParam(trigger=True, doc="Cancel a scheduled shutdown"),
            },
            "common": {"autotrigger": True},
        }

    def config_input_slots():
        return {"shutdown": DataType.ARRAY}

    # ---- runtime ----
    def setup(self):
        self._timer_thread = None
        self._cancel_event = threading.Event()
        self._lock = threading.Lock()

    def process(self, shutdown: Data):
        if shutdown is not None and np.any(shutdown.data > 0):
            self.input_slots["shutdown"].clear()
            self.shutdown_shutdown_changed(True)

    # ---- param callbacks ----
    def shutdown_shutdown_changed(self, val):
        if not val:
            return
        # capture delay AT PRESS; ignore later changes
        self._start_shutdown_timer(self.params.shutdown.delay.value)

    def shutdown_cancel_changed(self, val):
        if not val:
            return
        self._cancel_shutdown_timer()

    # ---- timer logic ----
    def _start_shutdown_timer(self, delay_s: float):
        with self._lock:
            self._cancel_shutdown_timer(inside_lock=True)
            self._cancel_event.clear()

            def _worker(captured_delay: float):
                cancelled = self._cancel_event.wait(timeout=max(0.0, captured_delay))
                if not cancelled:
                    self.request_shutdown()

            self._timer_thread = threading.Thread(target=_worker, args=(delay_s,), daemon=True)
            self._timer_thread.start()

    def _cancel_shutdown_timer(self, inside_lock: bool = False):
        ctx = self._lock if not inside_lock else None
        if ctx is not None:
            ctx.acquire()
        try:
            if self._timer_thread and self._timer_thread.is_alive():
                self._cancel_event.set()
                self._timer_thread.join(timeout=0.2)
            self._timer_thread = None
        finally:
            if ctx is not None:
                ctx.release()
