"""Bridge default bind must be loopback; non-loopback binds must warn.

Report F1: the bridge has no authentication and expression nodes execute
arbitrary Python, so a default `0.0.0.0` bind is a LAN-reachable RCE. The
default must be `127.0.0.1`; opting into a wider bind must print a warning.
"""

import inspect

from goofi.bridge.server import BridgeServer, bind_warning, start_bridge
from goofi.manager import Manager, build_arg_parser


def test_manager_default_bridge_host_is_loopback():
    assert inspect.signature(Manager.__init__).parameters["bridge_host"].default == "127.0.0.1"


def test_bridge_server_defaults_are_loopback():
    assert inspect.signature(BridgeServer.__init__).parameters["host"].default == "127.0.0.1"
    assert inspect.signature(start_bridge).parameters["host"].default == "127.0.0.1"


def test_cli_bind_default_is_loopback():
    parser = build_arg_parser()
    assert parser.parse_args([]).bind == "127.0.0.1"


def test_bind_warning_silent_for_loopback():
    assert bind_warning("127.0.0.1") is None
    assert bind_warning("localhost") is None
    assert bind_warning("::1") is None


def test_bind_warning_fires_for_nonloopback():
    msg = bind_warning("0.0.0.0")
    assert msg is not None
    assert "0.0.0.0" in msg
    # The warning must name the two reasons this is dangerous.
    assert "authentication" in msg.lower()
    assert "127.0.0.1" in msg
