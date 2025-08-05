import soundfile as sf

from goofi.data import Data, DataType
from goofi.node import InputSlot, Node
from goofi.params import FloatParam, IntParam, StringParam


class Audio2Txt(Node):
    """
    This node converts an audio signal into natural language text by using a large language model specialized for audio captioning or transcription. It can optionally take a prompt to guide the generation of the output text. The audio is processed and sent to the selected provider, which returns a generated text description or transcription of the audio content.

    Inputs:
    - prompt: Optional text prompt that can guide or condition the text generation.
    - audio: Array containing the audio data to be transcribed or described.

    Outputs:
    - generated_text: The text generated from the input audio, which may be a transcription or caption depending on the model and prompt.
    """

    @staticmethod
    def config_input_slots():
        return {
            "prompt": InputSlot(DataType.STRING, trigger_process=False),
            "audio": DataType.ARRAY,
        }

    @staticmethod
    def config_output_slots():
        return {"generated_text": DataType.STRING}

    @staticmethod
    def config_params():
        return {
            "audio_to_text": {
                "provider": StringParam("huggingface", options=["huggingface", "nexa"], doc="Provider for audio LMs"),
                "model": StringParam(
                    "Qwen/Qwen2-Audio-7B-Instruct",
                    doc="Huggingface model ID for audio captioning",
                ),
                "max_new_tokens": IntParam(30, 10, 1024, doc="Maximum number of tokens to generate"),
                "temperature": FloatParam(0.7, 0.1, 2.0, doc="Sampling temperature for text generation"),
            }
        }

    def setup(self):
        import librosa

        self.librosa = librosa

        provider = self.params.audio_to_text.provider.value
        if provider == "huggingface":
            self.setup_huggingface()
        elif provider == "nexa":
            self.setup_nexa()

    def process(self, prompt: Data, audio: Data):
        if audio.data is None:
            return None

        prompt = prompt.data

        if self.params.audio_to_text.provider.value == "huggingface":
            generated_text = self.generate_huggingface(prompt, audio)
        elif self.params.audio_to_text.provider.value == "nexa":
            generated_text = self.generate_nexa(prompt, audio)
        else:
            raise ValueError(f"Unsupported provider: {self.params.audio_to_text.provider.value}")

        return {"generated_text": (generated_text, {})}

    def audio_to_text_provider_changed(self, value):
        self.setup()

    def audio_to_text_model_changed(self, value):
        self.setup()

    def setup_huggingface(self):
        import torch

        try:
            from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Please install transformers to use Qwen2-Audio models via pip install transformers")

        try:
            self.processor = AutoProcessor.from_pretrained(
                self.params["audio_to_text"]["model"].value, torch_dtype=torch.float16
            )
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                self.params["audio_to_text"]["model"].value,
                device_map="cuda",
                torch_dtype=torch.float16,
            )
        except Exception as e:
            print(f"Error initializing Huggingface model: {e}")
            raise

    def generate_huggingface(self, prompt, audio):
        # Ensure the audio is sampled at the model's expected rate
        sampling_rate = self.processor.feature_extractor.sampling_rate
        audio_array = self.librosa.resample(audio.data, orig_sr=audio.meta["sfreq"], target_sr=sampling_rate)

        # Prepare the conversation input
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": "embedded"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

        inputs = self.processor(
            text=text,
            audios=[audio_array],
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate text
        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.params.audio_to_text.max_new_tokens.value,
            temperature=self.params.audio_to_text.temperature.value,
        )
        generate_ids = generate_ids[:, inputs["input_ids"].size(1) :]

        generated_text = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return generated_text

    def setup_nexa(self):
        try:
            from nexa.gguf.nexa_inference_audio_lm import NexaAudioLMInference
        except Exception as e:
            print(f"Error initializing Nexa model: {e}")
            raise

        self.model = NexaAudioLMInference(self.params.audio_to_text.model.value)

    def generate_nexa(self, prompt, audio):
        # save audio to temp file (nexa expects 16kHz audio)
        audio_file = "./temp_audio.wav"
        sf.write(audio_file, self.librosa.resample(audio.data, orig_sr=audio.meta["sfreq"], target_sr=16000), 16000)

        # TODO: this doesn't work due to current limitiation in Nexa
        self.model.params["max_new_tokens"] = self.params.audio_to_text.max_new_tokens.value
        self.model.params["temperature"] = self.params.audio_to_text.temperature.value

        # generate text
        response = self.model.inference(audio_file, prompt)

        # TODO: avoid reloading the model for each inference, the model should stay loaded in memory but currently it seems to reload every time so we need to cleanup
        self.model.cleanup()

        return response
