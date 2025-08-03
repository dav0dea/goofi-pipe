import cv2
import numpy as np
from PIL import Image

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, StringParam


class Embedding(Node):
    """
    This node generates embeddings for text and image data using a selectable machine learning model. It takes either text, array data (such as an image), or both, and outputs their respective embeddings as arrays. Different models are used depending on the selected embedding method, supporting both text embeddings (via models like CLIP, SBERT, FastText, Word2Vec) and image embeddings (via CLIP). The node processes the input, computes fixed-size vector representations (embeddings), and outputs them for downstream use.

    Inputs:
    - text: Input string to be converted into an embedding vector or set of vectors.
    - data: Input array data, typically representing an image, to be converted into an embedding vector.

    Outputs:
    - text_embeddings: Array containing the embedding(s) generated from the input text.
    - data_embeddings: Array containing the embedding generated from the input array data.
    """

    def config_input_slots():
        return {"text": DataType.STRING, "data": DataType.ARRAY}

    def config_output_slots():
        return {
            "text_embeddings": DataType.ARRAY,
            "data_embeddings": DataType.ARRAY,
        }

    def config_params():
        return {
            "embedding": {
                "model": StringParam(
                    "openai/clip-vit-base-patch32",
                    options=[
                        "openai/clip-vit-base-patch32",
                        "openai/clip-vit-large-patch14",
                        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                        "all-MiniLM-L6-v2",
                        "word2vec",
                    ],
                    doc="Model ID or name for embedding generation",
                ),
                "split_by_comma": BoolParam(
                    False,
                    doc="Whether to split text input by comma and generate embeddings for each part separately",
                ),
                "device": StringParam("cpu", options=["cpu", "cuda"]),
            }
        }

    def setup(self):
        # Explicitly reset attributes
        self.model = None
        self.processor = None
        self.model_type = None
        self.device = None

        # Select device
        import torch

        self.torch = torch

        torch.set_grad_enabled(False)

        self.device = torch.device(self.params.embedding.device.value)
        self.model_id = self.params.embedding.model.value
        print(f"Selected model: {self.model_id}, using device: {self.device}")

        # Load CLIP models
        if "clip" in self.model_id.lower():
            self.model_type = "clip"

            print("Initializing CLIP model...")
            from transformers import CLIPModel, CLIPProcessor

            try:
                # try loading the local model (local_files_only=False interferes with goofi-pipe's multiprocessing environment)
                # related issue: https://github.com/CompVis/stable-diffusion/issues/90#issuecomment-1228726914
                self.model = CLIPModel.from_pretrained(self.model_id, local_files_only=True).to(self.device)
            except OSError:
                self.model = CLIPModel.from_pretrained(self.model_id).to(self.device)

            self.processor = CLIPProcessor.from_pretrained(self.model_id)
        # Load other models
        elif "sbert" in self.model_id.lower() or "MiniLM" in self.model_id:
            self.model_type = "sbert"

            print("Initializing SBERT model...")
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_id).to(self.device)
        elif "fasttext" in self.model_id.lower():
            self.model_type = "fasttext"

            print("Initializing FastText model...")
            import gensim.downloader as api

            self.model = api.load("fasttext-wiki-news-subwords-300")
        elif "word2vec" in self.model_id.lower():
            self.model_type = "word2vec"

            print("Initializing Word2Vec model...")
            import gensim.downloader as api

            self.model = api.load("word2vec-google-news-300")
        else:
            raise ValueError(f"Unsupported model type: {self.model_id}")
        print(f"Model {self.model_id} initialized successfully as {self.model_type} on {self.device}.")

    def process(self, text: Data, data: Data):
        # Initialize outputs and metadata
        text_embeddings = None  # Default to None
        data_embeddings = None  # Default to None
        metadata = {}

        # Compute text embeddings if text input is provided
        if text is not None:
            input_text = text.data
            metadata["text"] = text.meta

            if self.params.embedding.split_by_comma.value:
                input_texts = [t.strip() for t in input_text.split(",")]
            else:
                input_texts = [input_text]

            with self.torch.inference_mode():
                if self.model_type == "clip":  # Use CLIP for text
                    inputs_text = self.processor(text=input_texts, return_tensors="pt", padding=True).to(self.device)
                    outputs_text = self.model.get_text_features(**inputs_text)
                    text_embeddings = outputs_text.cpu().numpy()

                elif self.model_type == "sbert":  # Use SBERT for text
                    text_embeddings = self.model.encode(input_texts, convert_to_numpy=True)

                elif self.model_type == "fasttext":  # Use FastText for text
                    text_embeddings = []
                    word_vectors = {
                        word: self.model[word] for sentence in input_texts for word in sentence.split() if word in self.model
                    }
                    for sentence in input_texts:
                        words = sentence.split()
                        vectors = [word_vectors[word] for word in words if word in word_vectors]
                        if vectors:
                            sentence_embedding = np.mean(vectors, axis=0)
                            text_embeddings.append(sentence_embedding)
                    text_embeddings = np.array(text_embeddings) if text_embeddings else None

                elif self.model_type == "word2vec":  # Use Word2Vec for text
                    text_embeddings = []
                    for sentence in input_texts:
                        words = sentence.split()
                        vectors = [self.model[word] for word in words if word in self.model]
                        if vectors:
                            sentence_embedding = np.mean(vectors, axis=0)
                            text_embeddings.append(sentence_embedding)
                    text_embeddings = np.array(text_embeddings) if text_embeddings else None

                else:
                    raise ValueError(f"Model {self.model_id} does not support text embeddings.")

        # Compute image embeddings if data input is provided
        if data is not None:
            input_data = data.data
            metadata["data"] = data.meta  # Preserve metadata

            with self.torch.inference_mode():
                # Preprocess and generate image embeddings using CLIP
                if self.model_type == "clip":
                    # Check and preprocess numpy array
                    if input_data.ndim == 4 and input_data.shape[0] == 1:  # Remove batch dimension if present
                        input_data = input_data.squeeze(0)

                    # Ensure image is resized and normalized
                    if input_data.shape[:2] != (224, 224):  # Resize if not already 224x224
                        input_data = cv2.resize(input_data, (224, 224))

                    # Normalize pixel values if necessary
                    if input_data.max() > 1.0:  # Normalize to [0, 1] if data is in [0, 255]
                        input_data = np.clip(input_data / 255.0, 0.0, 1.0)

                    # Convert to PIL Image for CLIP compatibility
                    input_data_pil = Image.fromarray((input_data * 255).astype(np.uint8))

                    # Use CLIPProcessor for final preprocessing
                    inputs_data = self.processor(images=input_data_pil, return_tensors="pt", padding=True)
                    inputs_data = {k: v.to(self.device) for k, v in inputs_data.items()}

                    # Generate embeddings
                    outputs_data = self.model.get_image_features(**inputs_data)
                    data_embeddings = outputs_data.cpu().numpy()
                else:
                    raise ValueError(f"Model {self.model_id} does not support image embeddings.")

        # Return separate embeddings and a dictionary of metadata
        return {
            "text_embeddings": (text_embeddings, metadata) if text_embeddings is not None else None,
            "data_embeddings": (data_embeddings, metadata) if data_embeddings is not None else None,
        }

    def embedding_model_changed(self, _):
        """Reinitialize the stream."""
        self.setup()

    def embedding_device_changed(self, _):
        """Reinitialize the stream."""
        self.setup()
