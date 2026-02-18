import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, StringParam


class Gemma3nAudio(Node):
    """
    Gemma3nAudio node extracts audio features using the Gemma3n conditional generation model.
    It processes input audio through the audio encoder and projects it into the language model space,
    outputting feature embeddings that can be used for downstream tasks or audio analysis.

    Inputs:
    - audio: 1D or 2D audio data array. If 2D, should be (channels, samples).

    Outputs:
    - features: Audio feature embeddings extracted from the input audio as a NumPy array.
    """

    @staticmethod
    def config_input_slots():
        return {"audio": DataType.ARRAY}

    @staticmethod
    def config_output_slots():
        return {"features": DataType.ARRAY}

    @staticmethod
    def config_params():
        return {
            "model": {
                "model_path": StringParam(
                    "google/gemma-3n-e2b-it", doc="Huggingface model ID or local path to Gemma3n model"
                ),
                "device": StringParam("cuda", options=["cuda", "cpu"], doc="Device to run the model on"),
                "dtype": StringParam("bfloat16", options=["bfloat16", "float16", "float32"], doc="Model dtype"),
            },
            "processing": {
                "target_samplerate": FloatParam(
                    16000.0, 8000.0, 48000.0, doc="Target sample rate for feature extraction"
                ),
            },
        }

    def setup(self):
        import torch

        self.torch = torch

        try:
            from transformers import Gemma3nForConditionalGeneration, Gemma3nProcessor
        except ImportError:
            try:
                from transformers import (
                    AutoModelForConditionalGeneration,
                    AutoProcessor,
                )

                # Fallback to Auto classes if Gemma3n classes not available
                print("Gemma3n classes not found, using Auto classes")
                Gemma3nForConditionalGeneration = AutoModelForConditionalGeneration
                Gemma3nProcessor = AutoProcessor
            except ImportError:
                raise ImportError(
                    "Please install transformers with Gemma3n support: " "pip install transformers>=4.50.0"
                )

        # Get model configuration
        model_path = self.params.model.model_path.value
        device = self.params.model.device.value
        dtype_str = self.params.model.dtype.value

        # Map dtype string to torch dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.dtype = dtype_map.get(dtype_str, torch.bfloat16)

        # Check if CUDA is available if cuda is selected
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = "cpu"
            if self.dtype == torch.bfloat16:
                print("bfloat16 not well supported on CPU, switching to float32")
                self.dtype = torch.float32

        self.device = device

        try:
            # Load the processor (includes feature extractor)
            self.processor = Gemma3nProcessor.from_pretrained(model_path)
            print(f"Loaded Gemma3n processor from {model_path}")

            # Load the conditional generation model
            self.model = Gemma3nForConditionalGeneration.from_pretrained(
                model_path, device_map=device, dtype=self.dtype
            ).eval()
            print(f"Loaded Gemma3n model from {model_path}")

            # Get the feature extractor's expected sample rate
            self.expected_sr = getattr(
                self.processor.feature_extractor, "sampling_rate", self.params.processing.target_samplerate.value
            )
            print(f"Feature extractor expects sample rate: {self.expected_sr} Hz")

        except Exception as e:
            print(f"Error loading Gemma3n model: {e}")
            raise

    def process(self, audio: Data):
        if audio is None or audio.data is None:
            return None

        audio_data = audio.data
        input_sr = audio.meta.get("sfreq", 44100)

        # Handle audio shape: convert to 1D if needed
        if audio_data.ndim == 2:
            # If stereo or multi-channel, take mean across channels
            if audio_data.shape[0] < audio_data.shape[1]:
                # Shape is (channels, samples)
                audio_data = np.mean(audio_data, axis=0)
            else:
                # Shape is (samples, channels)
                audio_data = np.mean(audio_data, axis=1)
        elif audio_data.ndim > 2:
            raise ValueError(f"Audio data has unsupported shape: {audio_data.shape}")

        # Resample if needed
        if input_sr != self.expected_sr:
            try:
                import librosa

                audio_data = librosa.resample(audio_data, orig_sr=input_sr, target_sr=self.expected_sr)
            except ImportError:
                print("Warning: librosa not available for resampling. Using audio as-is.")
                print(f"Expected {self.expected_sr} Hz but got {input_sr} Hz")

        # Ensure audio is 2D for the processor: (batch, samples) or (channels, samples)
        # The feature extractor expects shape (channels, samples) typically
        if audio_data.ndim == 1:
            audio_data = audio_data[np.newaxis, :]  # Add channel dimension

        try:
            # Extract features using the processor's feature extractor
            features = self.processor.feature_extractor(audio_data, sampling_rate=self.expected_sr, return_tensors="pt")

            # Move features to the correct device
            features = {k: v.to(self.device) for k, v in features.items()}

            # Use the model's get_audio_features method to extract features
            with self.torch.no_grad():
                audio_features, audio_mask = self.model.model.get_audio_features(
                    features["input_features"], ~features["input_features_mask"]
                )

            # Convert to numpy (bfloat16 not supported by numpy, so convert to float32 first)
            if audio_features.dtype == self.torch.bfloat16:
                feature_array = audio_features.cpu().to(self.torch.float32).numpy()
            else:
                feature_array = audio_features.cpu().numpy()

            # Squeeze batch dimension if present
            if feature_array.ndim > 2:
                feature_array = feature_array.squeeze(0)

            return {"features": (feature_array, {})}

        except Exception as e:
            print(f"Error extracting features: {e}")
            import traceback

            traceback.print_exc()
            return None

    def model_model_path_changed(self, value):
        """Reload processor when model path changes"""
        self.setup()

    def model_device_changed(self, value):
        """Reload processor when device changes"""
        self.setup()

    def model_dtype_changed(self, value):
        """Reload processor when dtype changes"""
        self.setup()
