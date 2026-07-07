"""AudioStream / AudioOut must survive an empty device enumeration.

On a machine with sounddevice installed but zero input/output devices (a
headless box, a container, an unplugged interface), ``list_audio_devices()``
returns [] with no exception — so the ``except`` fallback never fires and
``devices[0]`` used to raise IndexError at child bootstrap, landing the node in
a permanent error instead of booting with a ``default`` device.
"""
import pytest

from goofi.nodes.inputs.audiostream import AudioStream
from goofi.nodes.outputs.audioout import AudioOut


@pytest.mark.parametrize("cls", [AudioStream, AudioOut])
def test_empty_device_list_falls_back_to_default(cls, monkeypatch):
    monkeypatch.setattr(cls, "list_audio_devices", staticmethod(lambda: []))
    params = cls.config_params()
    device = params["audio"]["device"]
    assert device.value == "default"
    assert device.options == ["default"]
