import threading
import time
import numpy as np

from oscpy.server import OSCThreadServer
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, IntParam, StringParam


class OSCIn(Node):
    def config_params():
        return {
            "osc": {
                "backend": StringParam("oscpy", options=["oscpy", "python-osc"], doc="OSC backend"),
                "address": StringParam("0.0.0.0"),
                "port": IntParam(9000, 0, 65535),
                "find_usable_port_tries": IntParam(
                    0, 0, 100, doc="If > 0, increment port until an unoccupied port is found (up to N tries)."
                ),
                "keep_messages": BoolParam(True, doc="Keep all received messages"),
                "clear": BoolParam(trigger=True, doc="Clear all stored messages"),
            },
            "common": {"autotrigger": True},
        }

    def config_input_slots():
        return {"address": DataType.STRING, "port": DataType.ARRAY}

    def config_output_slots():
        return {"message": DataType.TABLE}

    # ---------------- lifecycle ---------------- #

    def setup(self):
        self.messages: dict[str, Data] = {}
        self._srv = None
        self._srv_thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._start_backend()

    def teardown(self):
        self._stop_backend(silent=True)

    # ---------------- backend control ---------------- #

    def _start_backend(self):
        with self._lock:
            addr = self.params.osc.address.value
            base_port = self.params.osc.port.value
            tries_left = self.params.osc.find_usable_port_tries.value
            backend = self.params.osc.backend.value

            last_err = None
            for i in range(tries_left + 1):
                port = base_port + i
                try:
                    if backend == "oscpy":
                        self._start_oscpy(addr, port)
                    else:
                        self._start_pythonosc(addr, port)
                    return
                except OSError as e:
                    last_err = e
                    self._stop_backend(silent=True)
                    time.sleep(0.02)  # help Windows release the port
            if last_err:
                raise last_err

    def _stop_backend(self, silent: bool = False):
        with self._lock:
            try:
                if isinstance(self._srv, OSCThreadServer):
                    # CLI pattern shutdown: terminate then join
                    self._srv.terminate_server()
                    self._srv.join_server()
                elif isinstance(self._srv, ThreadingOSCUDPServer):
                    if self._srv_thread and self._srv_thread.is_alive():
                        self._srv.shutdown()
                        self._srv.server_close()
                        self._srv_thread.join(timeout=2.0)
            except Exception:
                if not silent:
                    raise
            finally:
                self._srv = None
                self._srv_thread = None

    # ---------------- oscpy backend (CLI default_handler pattern) ---------------- #

    def _start_oscpy(self, address: str, port: int):
        # Match oscpy CLI: use default_handler instead of per-pattern binds.
        osc = OSCThreadServer(
            encoding="utf-8",
            encoding_errors="replace",
            default_handler=self._oscpy_default_handler,
        )
        osc.listen(address=address, port=port, default=True)
        self._srv = osc

    def _oscpy_default_handler(self, address: bytes, *values):
        # address is bytes; values may be bytes / numbers, etc.
        addr_str = address.decode("utf-8", errors="ignore")
        self._handle_message(addr_str, *values)

    # ---------------- python-osc backend ---------------- #

    def _start_pythonosc(self, address: str, port: int):
        disp = Dispatcher()
        disp.set_default_handler(self._pythonosc_cb, needs_reply_address=False)

        srv = ThreadingOSCUDPServer((address, port), disp)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()

        self._srv = srv
        self._srv_thread = t

    def _pythonosc_cb(self, addr: str, *args):
        self._handle_message(addr, *args)

    # ---------------- shared handling ---------------- #

    def _handle_message(self, address: str, *args):
        # Normalize None to b"None" for consistency
        norm = [("None".encode() if a is None else a) for a in args]

        # Single string payload → STRING
        if norm and isinstance(norm[0], (bytes, str)):
            if len(norm) > 1:
                raise ValueError(
                    "OSCIn does not support multiple string args per address; "
                    f"received {[(x.decode() if isinstance(x, bytes) else x) for x in norm]}"
                )
            s = norm[0].decode() if isinstance(norm[0], bytes) else str(norm[0])
            val = Data(DataType.STRING, s, {})
        else:
            # Mixed/numeric payload → ARRAY (dtype=object keeps heterogenous types intact)
            val = Data(DataType.ARRAY, np.array(norm, dtype=object), {})

        self.messages[address] = val

    def process(self, address: Data, port: Data):
        if address is not None:
            self.input_slots["address"].clear()
            self.params.osc.address.value = address.data
            self.osc_address_changed(address.data)
        if port is not None:
            self.input_slots["port"].clear()
            self.params.osc.port.value = int(port.data)
            self.osc_port_changed(int(port.data))

        if not self.messages:
            return None
        out = self.messages
        if (not self.params.osc.keep_messages.value) or self.params.osc.clear.value:
            self.messages = {}
        return {"message": (out, {})}

    # -------- live param changes (restart backend safely) -------- #

    def osc_backend_changed(self, _):
        self._stop_backend(silent=True)
        self._start_backend()

    def osc_address_changed(self, _):
        self._stop_backend(silent=True)
        self._start_backend()

    def osc_port_changed(self, _):
        self._stop_backend(silent=True)
        self._start_backend()
