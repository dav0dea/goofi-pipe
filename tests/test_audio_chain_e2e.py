import numpy as np

from goofi.data import Data, DataType
from goofi.nodes.inputs.oscillator import Oscillator
from goofi.nodes.outputs.audioout import AudioOut
from goofi.nodes.signal.gain import Gain

from .audio_utils import FixedClock, fake_stream_factory


def test_oscillator_gain_audioout_chain_is_click_free():
    sr, blocksize, n_blocks = 48000, 256, 80

    osc = Oscillator.create_standalone()
    osc.setup()
    osc.clock = FixedClock([blocksize] * n_blocks)
    osc.params.oscillator.sampling_frequency.value = float(sr)
    osc.params.oscillator.type.value = "sine"

    gain = Gain.create_standalone()
    gain.setup()
    gain.params.gain.gain.value = 0.5

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
    # it through the length-preserving Gain (node.py publish path, Phase 3). The
    # standalone harness reproduces that: a fresh counter at the generator,
    # copied through Gain.
    for i in range(n_blocks):
        osc.params.oscillator.frequency.value = float(sweep[i])
        osc_val, osc_meta = osc.process(None)["out"]
        osc_meta = dict(osc_meta)
        osc_meta["index"] = i

        gain_val, gain_meta = gain.process(
            Data(DataType.ARRAY, np.asarray(osc_val), osc_meta)
        )["out"]
        gain_meta = dict(gain_meta)
        gain_meta["index"] = osc_meta["index"]      # length-preserving: index copied through

        sink.process(Data(DataType.ARRAY, np.asarray(gain_val), gain_meta), None)

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
