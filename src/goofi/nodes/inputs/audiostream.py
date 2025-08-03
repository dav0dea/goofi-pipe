import numpy as np

from goofi.data import DataType
from goofi.node import Node
from goofi.params import StringParam


class AudioStream(Node):
    """
    This node captures live audio data from an input device (such as a microphone) and outputs the audio as a NumPy array. It supports real-time operation and can optionally convert multichannel audio to mono. The output includes audio data along with its sampling frequency.

    Outputs:
    - out: A NumPy array containing the recorded audio data (mono or multi-channel, depending on configuration) and a metadata dictionary containing the sampling frequency ('sfreq').
    """

    def config_params():
        return {
            "audio": {
                "sampling_rate": StringParam("44100", options=["44100", "48000"]),
                "device": StringParam(AudioStream.list_audio_devices()[0], options=AudioStream.list_audio_devices()),
                "convert_to_mono": True,
            },
            "common": {"autotrigger": True},
        }

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def setup(self):
        import sounddevice as sd

        if hasattr(self, "stream") and self.stream:
            self.stream.stop()
            self.stream.close()

        self.buffer = None

        self.stream = sd.InputStream(
            callback=self.audio_callback,
            samplerate=int(self.params.audio.sampling_rate.value),
            device=self.params.audio.device.value,
        )
        self.stream.start()

    def audio_callback(self, indata, frames, time, status):
        """This callback receives audio data from the audio stream."""
        if self.buffer is None:
            self.buffer = np.array(indata.T)
        else:
            self.buffer = np.concatenate((self.buffer, indata.T), axis=1)

    def process(self):
        if self.stream is None:
            raise RuntimeError("Audio stream is not available.")

        if self.buffer is None:
            return None

        data = np.squeeze(np.array(self.buffer))
        self.buffer = None

        # convert to mono if required
        if self.params.audio.convert_to_mono.value and data.ndim > 1:
            data = np.mean(data, axis=0, keepdims=False)

        return {"out": (data, {"sfreq": float(self.params.audio.sampling_rate.value)})}

    def audio_sampling_frequency_changed(self, value):
        self.setup()

    def audio_device_changed(self, value):
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
            # check if the device is an input device
            if device["max_input_channels"] > 0:
                device_names.append(device["name"])
        return device_names
