import io

import numpy as np
import soundfile as sf

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, StringParam


class SpeechSynthesis(Node):
    """
    This node provides speech synthesis and transcription capabilities using OpenAI's API. It can convert input text into synthesized speech audio, or transcribe input speech audio into text.

    Inputs:
    - text: The text string to be converted into speech audio.
    - voice: An array representing audio data to be transcribed into text.

    Outputs:
    - speech: An array containing the synthesized speech audio corresponding to the input text, or an empty array if text input is not provided.
    - transcript: The transcribed text from the input voice array, or an empty string if voice input is not provided.
    """

    def config_input_slots():
        return {"text": DataType.STRING, "voice": DataType.ARRAY}

    def config_output_slots():
        return {"speech": DataType.ARRAY, "transcript": DataType.STRING}

    def config_params():
        return {
            "speech_generation": {
                "openai_key": StringParam("openai.key"),
                "speed": FloatParam(1.0, 0.1, 2.0),
                "model": StringParam("tts-1", options=["tts-1", "tts-1-hd"]),
                "voice": StringParam("alloy", options=["alloy", "echo", "fable", "onyx", "nova", "shimmer"]),
            }
        }

    def setup(self):
        import librosa
        import openai

        self.librosa = librosa
        self.openai = openai

        key = self.params["speech_generation"]["openai_key"].value
        with open(key, "r") as f:
            key = f.read().strip()

        self.openai.api_key = key
        self.client = openai.OpenAI(api_key=key)

    def process(self, text: Data, voice: Data):
        speech = None
        transcript = ""

        if text is None and voice is None:
            return {"speech": (np.array([]), {}), "transcript": ("", {})}

        if text is not None:
            audio_generator = self.synthesize_speech_stream(text.data, self.params["speech_generation"]["speed"].value)
            # convert bytes to numpy array
            speech = self.bytes_to_array(b"".join([speech_bytes for speech_bytes in audio_generator]))
        elif voice is not None:
            # ensure transcript is a string
            transcript = self.transcribe_voice(voice.data) or ""

        return {"speech": (speech, {}), "transcript": (transcript, {})}

    def synthesize_speech_stream(self, text, speed):
        response = self.openai.audio.speech.create(
            model=self.params.speech_generation.model.value,
            voice=self.params.speech_generation.voice.value,
            input=text,
            speed=speed,
        )
        # yield audio chunks
        for chunk in response.iter_bytes():
            yield chunk

    def bytes_to_array(self, audio_bytes):
        with io.BytesIO(audio_bytes) as audio_file:
            with sf.SoundFile(audio_file) as sf_file:
                audio_array = np.array(sf_file.read(dtype="float32"))
                # resample if the sample rate is different from 44100 Hz
                if sf_file.samplerate != 44100:
                    audio_array = self.resample_audio(audio_array, sf_file.samplerate, 44100)
                return audio_array

    def resample_audio(self, audio, input_rate, output_rate):
        # resample audio from input_rate to output_rate
        return self.librosa.resample(audio, orig_sr=input_rate, target_sr=output_rate)

    def transcribe_voice(self, voice_buffer):
        # convert the numpy array buffer to a WAV file in memory
        with open("tmp.wav", "wb") as audio_stream:
            sf.write(audio_stream, voice_buffer, 44100, format="wav")

        with open("tmp.wav", "rb") as audio_stream:
            # send the audio file to OpenAI for transcription
            response = self.client.audio.transcriptions.create(model="whisper-1", file=audio_stream)

        return response.text
