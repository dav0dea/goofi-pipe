import numpy as np
import sounddevice as sd

from goofi.audio.continuity import INDEX_META_KEY, crossfade, is_discontinuous
from goofi.audio.drift import DriftCorrector
from goofi.audio.ring import AudioRing
from goofi.data import Data, DataType
from goofi.node import InputSlot, Node
from goofi.params import FloatParam, IntParam, StringParam


# Discontinuity seam crossfade length (samples): short + fixed, REPLACES the
# first frames of the post-jump block (never prepends), so it cannot inflate
# the timebase the way the old per-block transition ramp did.
CROSSFADE_SAMPLES = 64


def _sounddevice_stream_factory(*, samplerate, channels, blocksize, device, callback):
    """Default stream factory: a real non-blocking sounddevice callback stream."""
    return sd.OutputStream(
        samplerate=samplerate,
        channels=channels,
        blocksize=blocksize,
        device=device,
        callback=callback,
    )


class AudioOut(Node):
    """
    This node plays incoming audio to an output device in real-time using a
    non-blocking callback stream fed by a jitter-buffer ring. Audio blocks are
    written into the ring on the tick thread; the device callback drains the ring
    on its own clock. The stream is primed (not started until the ring reaches
    its target fill) and outputs zeros on an empty ring, so playback never blocks
    the graph and never blocks the producer.

    Inputs:
    - data: Array of audio samples to play through the output device.
    - device: Name of the audio output device to use.

    Outputs:
    - finished: Signals that a block was accepted into the playback ring.
    """

    # Overridable in tests: a callable returning a stream object exposing
    # start()/stop()/close() and driven by the injected callback. A staticmethod
    # so `self.stream_factory(...)` never binds `self`; a test sets an instance
    # attribute (a plain function, likewise unbound).
    stream_factory = staticmethod(_sounddevice_stream_factory)

    def config_input_slots():
        return {"data": InputSlot(DataType.ARRAY, queue=True), "device": DataType.STRING}

    def config_output_slots():
        return {"finished": DataType.ARRAY}

    def config_params():
        # Device enumeration is runtime state: under the registry's static
        # evaluation this falls to the fallback (the live node reports the real
        # list via its state push).
        try:
            devices = AudioOut.list_audio_devices() or ["default"]
        except Exception:
            devices = ["default"]
        return {
            "audio": {
                "sampling_rate": StringParam("48000", options=["48000", "44100", "32000", "16000"]),
                "device": StringParam(devices[0], options=devices, refresh="_refresh_audio_devices"),
                "buffer_ms": IntParam(30, 5, 200),
                "blocksize": IntParam(256, 32, 2048),
            },
            # Free-run the DAC: the callback stream, not the tick limiter, paces output.
            "common": {"max_frequency": FloatParam(0.0, 0.0, 60.0)},
        }

    def _refresh_audio_devices(self):
        """Re-enumerate live output devices for the device dropdown's ⟳ button."""
        return AudioOut.list_audio_devices() or ["default"]

    def _stats_extra(self):
        """Surface the sink's live output latency and jitter-ring under/overrun
        counts on the NODE_STATS channel so they render in the inspector."""
        ring = getattr(self, "ring", None)
        stream = getattr(self, "stream", None)
        latency = getattr(stream, "latency", 0.0) if stream is not None else 0.0
        return {
            "latency_ms": round(float(latency) * 1000.0, 2),
            "underruns": ring.underruns if ring is not None else 0,
            "overruns": ring.overruns if ring is not None else 0,
        }

    def setup(self):
        if getattr(self, "stream", None) is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass

        sr = int(self.params.audio.sampling_rate.value)
        self.target_fill = int(sr * self.params.audio.buffer_ms.value / 1000)
        self.ring = AudioRing(capacity_frames=max(self.target_fill * 4, 1), channels=2)
        self.started = False
        self.drift = DriftCorrector(self.target_fill)
        self._prev_index = None
        self._prev_tail = None
        self.discontinuities = 0

        self.stream = self.stream_factory(
            samplerate=sr,
            channels=2,
            blocksize=self.params.audio.blocksize.value,
            device=self.params.audio.device.value,
            callback=self._callback,
        )

    def _callback(self, outdata, frames, time_info, status):
        # Runs on the device thread: fill from the ring, zero any shortfall.
        outdata[:] = 0.0
        self.ring.read_into(outdata)

    def _to_stereo(self, arr):
        samples = np.asarray(arr, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.T.squeeze()
        if samples.ndim == 1:
            # Mono -> duplicate into stereo.
            samples = np.stack((samples, samples), axis=-1)
        return np.ascontiguousarray(samples, dtype=np.float32)

    def process(self, data: Data, device: Data):
        if device is not None:
            self.params.audio.device.value = device.data
            self.audio_device_changed(device.data)
            self.input_slots["device"].clear()

        if data is None:
            return

        block = self._to_stereo(data.data)

        # `index` is source-origin + propagated (base-stamped at node.py's publish
        # path, Phase 3). A jump > 1 means an upstream drop/reset: crossfade the
        # seam so the phase discontinuity is inaudible. Contiguous audio never
        # crossfades.
        index = data.meta.get(INDEX_META_KEY)
        if is_discontinuous(self._prev_index, index):
            if self._prev_tail is not None:
                block = crossfade(self._prev_tail, block, CROSSFADE_SAMPLES)
            self.discontinuities += 1
        self._prev_index = index

        # Threshold-triggered single-sample stuff/drop at a zero crossing to
        # track DAC-vs-producer clock drift (inaudible, ~one edit / 0.3 s). Only
        # in steady state: during priming the ring is deliberately below target,
        # so correcting there would stuff a duplicate frame into every pre-
        # playback write. The DAC isn't draining yet, so there is no drift to fix.
        if self.started:
            block = self.drift.correct(block, self.ring.fill)
        self.ring.write(block)
        if block.shape[0] > 0:
            self._prev_tail = block

        if not self.started and self.ring.fill >= self.target_fill:
            self.stream.start()
            self.started = True

        return {"finished": (np.array([1]), {})}

    def audio_sampling_rate_changed(self, value):
        self.setup()

    def audio_device_changed(self, value):
        self.setup()

    @staticmethod
    def list_audio_devices():
        """Returns a list of available audio output devices."""
        devices = sd.query_devices()
        device_names = []
        for device in devices:
            if device["max_output_channels"] > 0:
                device_names.append(device["name"])
        return device_names
