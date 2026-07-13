import numpy as np

from .audio_utils import FakeStream, FixedClock, fake_stream_factory


def test_fake_stream_pump_invokes_callback_and_captures():
    seen = {}

    def callback(outdata, frames, time_info, status):
        seen["frames"] = frames
        outdata[:] = 0.5

    stream = fake_stream_factory(
        samplerate=48000, channels=2, blocksize=64, device=None, callback=callback
    )
    assert isinstance(stream, FakeStream)
    assert stream.started is False
    stream.start()
    assert stream.started is True

    out = stream.pump(64)
    assert seen["frames"] == 64
    assert out.shape == (64, 2)
    assert np.allclose(out, 0.5)
    assert len(stream.captured) == 1
    assert np.allclose(stream.captured[0], 0.5)


def test_fake_stream_close_marks_closed():
    stream = fake_stream_factory(
        samplerate=48000, channels=2, blocksize=64, device=None,
        callback=lambda o, f, t, s: None,
    )
    stream.close()
    assert stream.closed is True


def test_fixed_clock_emits_scripted_sizes_then_zero():
    clock = FixedClock([10, 20])
    clock.start(0.0)
    assert clock.advance(0.0) == 10
    assert clock.advance(0.0) == 20
    assert clock.advance(0.0) == 0
    assert clock.emitted == 0
