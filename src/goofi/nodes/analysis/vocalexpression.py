from os import path, environ
import numpy as np
import asyncio
import base64
from io import BytesIO
from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, StringParam


class VocalExpression(Node):
    """
    This node analyzes voice audio input to detect emotional expressions using the Hume AI API. It processes a 1D audio signal and returns the predominant vocal emotions and their associated confidence scores, based on prosody and burst vocal analytics.

    Inputs:
    - data: A 1D array containing the raw audio waveform data.

    Outputs:
    - prosody_label: The name of the strongest detected emotion from prosody-based analysis.
    - burst_label: The name of the strongest detected emotion from burst-based analysis.
    - prosody_score: The confidence score of the detected prosody emotion as a 1-element array.
    - burst_score: The confidence score of the detected burst emotion as a 1-element array.
    """

    def setup(self):
        self.load_api_key()
        self.AudioSegment, self.HumeStreamClient, self.BurstConfig, self.ProsodyConfig = import_audio_libs()

    @staticmethod
    def config_input_slots():
        return {"data": DataType.ARRAY}

    @staticmethod
    def config_output_slots():
        return {
            "prosody_label": DataType.STRING,
            "burst_label": DataType.STRING,
            "prosody_score": DataType.ARRAY,
            "burst_score": DataType.ARRAY,
        }

    @staticmethod
    def config_params():
        return {
            "vocal_analysis": {
                "api_key": StringParam("hume.key", doc="Hume API key"),
                "emotion_threshold": FloatParam(0.0, 0.0, 1.0, doc="Threshold for filtering emotions"),
            }
        }

    def load_api_key(self):
        self.api_key = self.params["vocal_analysis"]["api_key"].value

        if path.exists(self.api_key):
            with open(self.api_key, "r") as f:
                self.api_key = f.read().strip()
        elif self.api_key is str and not self.api_key.startswith("hume."):
            self.api_key = environ.get("HUME_API_KEY", self.api_key)
        elif not self.api_key:
            raise ValueError("API key not found")

    async def decode_emotion_prosody(self, encoded_audio_sample):
        client = self.HumeStreamClient(self.params["vocal_analysis"]["api_key"].value)
        burst_config = self.BurstConfig()
        prosody_config = self.ProsodyConfig()

        prosody_label = ""
        prosody_score = 0.0
        burst_label = ""
        burst_score = 0.0
        emotion_threshold = self.params["vocal_analysis"]["emotion_threshold"].value

        for attempt in range(3):  # Retry up to 3 times
            try:
                async with client.connect([burst_config, prosody_config]) as socket:
                    # Reset connection stream context between samples
                    await socket.reset_stream()

                    # Send binary data
                    result = await socket.send_bytes(encoded_audio_sample)

                    if "prosody" in result:
                        if "warning" in result["prosody"]:
                            print(result["prosody"]["warning"])
                        else:
                            emotions = result["prosody"]["predictions"][0]["emotions"]
                            if emotions:
                                main_emotion = max(emotions, key=lambda e: e["score"])
                                if main_emotion["score"] >= emotion_threshold:
                                    prosody_label = main_emotion["name"]
                                    prosody_score = main_emotion["score"]

                    if "burst" in result:
                        if "warning" in result["burst"]:
                            print(result["burst"]["warning"])
                        else:
                            emotions = result["burst"]["predictions"][0]["emotions"]
                            if emotions:
                                main_emotion = max(emotions, key=lambda e: e["score"])
                                if main_emotion["score"] >= emotion_threshold:
                                    burst_label = main_emotion["name"]
                                    burst_score = main_emotion["score"]

                    return prosody_label, burst_label, prosody_score, burst_score
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(1)  # Wait before retrying
        raise Exception("Failed to connect to Hume API after multiple attempts")

    def process(self, data: Data):
        if data is None:
            return None

        audio_sample = data.data
        if audio_sample.ndim > 1:
            raise ValueError("Data must be 1D")

        # Convert numpy array to AudioSegment
        audio_segment = self.AudioSegment(
            data=audio_sample.tobytes(),
            sample_width=audio_sample.dtype.itemsize,
            frame_rate=44100,  # Assuming the sample rate is 44100 Hz
            channels=1,  # Assuming the audio is mono
        )

        # Export AudioSegment to WAV format in memory
        audio_io = BytesIO()
        audio_segment.export(audio_io, format="wav")
        audio_io.seek(0)
        encoded_audio_sample = base64.b64encode(audio_io.read()).decode("utf-8")

        prosody_label, burst_label, prosody_score, burst_score = asyncio.run(
            self.decode_emotion_prosody(encoded_audio_sample.encode("utf-8"))
        )

        return {
            "prosody_label": (prosody_label, {}),
            "burst_label": (burst_label, {}),
            "prosody_score": (np.array([prosody_score]), {}),
            "burst_score": (np.array([burst_score]), {}),
        }


def import_audio_libs():
    try:
        from pydub import AudioSegment
    except ImportError:
        raise ImportError("You need to install pydub to use the VocalExpression node: pip install pydub")
    try:
        from hume import HumeStreamClient
        from hume.models.config import BurstConfig, ProsodyConfig
    except ImportError:
        raise ImportError("You need to install hume to use the VocalExpression node: pip install hume")
    return AudioSegment, HumeStreamClient, BurstConfig, ProsodyConfig
