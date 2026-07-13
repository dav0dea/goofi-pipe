import numpy as np

from goofi.data import Data, DataType
from goofi.nodes.array.math import Math
from goofi.nodes.inputs.oscillator import Oscillator
from goofi.nodes.outputs.audioout import AudioOut

from .audio_utils import FixedClock, fake_stream_factory


def test_oscillator_math_audioout_chain_is_click_free():
    # The user's real scenario: an Oscillator through a Math node scaling its
    # amplitude (×0.5) into AudioOut. Math is length-preserving and passes its
    # input meta through, so the source-origin index flows unbroken and the sink
    # never sees a discontinuity — the scaled sine stays click-free.
    sr, blocksize, n_blocks = 48000, 256, 80

    osc = Oscillator.create_standalone()
    osc.setup()
    osc.clock = FixedClock([blocksize] * n_blocks)
    osc.params.oscillator.sampling_frequency.value = float(sr)
    osc.params.oscillator.type.value = "sine"

    math = Math.create_standalone()
    math.params.math.multiply.value = 0.5           # scale amplitude by half

    sink = AudioOut.create_standalone()
    sink.stream_factory = fake_stream_factory
    sink.params.audio.sampling_rate.value = str(sr)
    sink.params.audio.buffer_ms.value = 10          # target_fill = 480 frames
    sink.params.audio.blocksize.value = blocksize
    sink.setup()

    # A frequency sweep across the run. The phase-continuous SampleClock
    # oscillator must keep the waveform continuous through every change.
    sweep = np.linspace(100.0, 800.0, n_blocks)

    # In the wired system the base stamps a source-origin `index` and propagates
    # it through the length-preserving Math (node.py publish path, Phase 3). The
    # standalone harness reproduces that: a fresh counter at the generator,
    # copied through Math (which returns its input meta unchanged).
    for i in range(n_blocks):
        osc.params.oscillator.frequency.value = float(sweep[i])
        osc_val, osc_meta = osc.process(None)["out"]
        osc_meta = dict(osc_meta)
        osc_meta["index"] = i

        math_val, math_meta = math.process(
            Data(DataType.ARRAY, np.asarray(osc_val), osc_meta)
        )["out"]

        sink.process(Data(DataType.ARRAY, np.asarray(math_val), math_meta), None)

        # Drain the DAC once primed, only when a full block is ready so the ring
        # never underruns (an underrun would inject a zero-fill seam = a click).
        if sink.started and sink.ring.fill >= blocksize:
            sink.stream.pump(blocksize)

    # No index gap reached the sink, so it never crossfaded a seam.
    assert sink.discontinuities == 0

    played = np.concatenate(sink.stream.captured, axis=0)[:, 0]
    assert played.size >= blocksize

    # Click-freeness: the largest sample-to-sample step stays far below the ~O(1)
    # jump a phase discontinuity would produce (analytic step here is ~0.05).
    assert float(np.max(np.abs(np.diff(played)))) < 0.2
