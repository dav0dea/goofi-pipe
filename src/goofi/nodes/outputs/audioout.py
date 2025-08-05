import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import StringParam


class AudioOut(Node):
    """
    This node plays incoming audio data to an audio output device in real-time. It receives audio arrays and routes the signal to the selected audio hardware, handling transitions between consecutive audio blocks for seamless playback. The node can also switch audio devices dynamically.

    Inputs:
    - data: Array of audio samples to play through the output device.
    - device: Name of the audio output device to use.

    Outputs:
    - finished: Signals completion of playback for the current audio data block.
    """

    def config_input_slots():
        return {"data": DataType.ARRAY, "device": DataType.STRING}

    def config_output_slots():
        return {"finished": DataType.ARRAY}

    def config_params():
        return {
            "audio": {
                "sampling_rate": StringParam("44100", options=["44100", "48000", "32000", "16000"]),
                "device": StringParam(AudioOut.list_audio_devices()[0], options=AudioOut.list_audio_devices()),
                "transition_samples": 100,
            }
        }

    def setup(self):
        import sounddevice as sd

        if hasattr(self, "stream") and self.stream:
            self.stream.stop()
            self.stream.close()

        self.stream = sd.OutputStream(
            samplerate=int(self.params.audio.sampling_rate.value),
            device=self.params.audio.device.value,
            channels=2,
        )
        self.stream.start()

        self.last_sample = None

    def process(self, data: Data, device: Data):
        if device is not None:
            self.params.audio.device.value = device.data
            self.audio_device_changed(device.data)
            self.input_slots["device"].clear()

        if data is None:
            return

        if self.stream is None:
            raise RuntimeError("Audio output stream is not available.")

        # set data type to float32
        samples = data.data.astype(np.float32).T.squeeze()
        # Handle Mono to Stereo or Stereo to Mono Conversion
        # Verify that the samples array has the correct number of dimensions
        if samples.data.ndim == 1:
            # Mono audio: duplicate the channel for stereo output
            samples = np.stack((samples.data, samples.data), axis=-1)
        else:
            # For already stereo or multi-channel data, use as is
            samples = samples.data

        if self.last_sample is None:
            self.last_sample = samples[-1]

        transition = np.linspace(self.last_sample, samples[0], num=self.params.audio.transition_samples.value)
        samples = np.concatenate((transition, samples), axis=0)

        self.last_sample = samples[-1]

        # Send the audio data to the output device after ensuring it's C-contiguous
        self.stream.write(np.ascontiguousarray(samples))

        return {"finished": (np.array([1]), {})}

    def audio_sampling_rate_changed(self, value):
        self.setup()

    @staticmethod
    def list_audio_devices():
        """Returns a list of available audio devices."""
        import sounddevice as sd

        if sd is None:
            return ["None"]

        devices = sd.query_devices()
        device_names = []
        for device in devices:
            # check if the device is an output device
            if device["max_output_channels"] > 0:
                device_names.append(device["name"])
        return device_names

    def audio_device_changed(self, value):
        self.setup()
