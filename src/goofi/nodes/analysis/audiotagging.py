import pickle
from os.path import join

import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, FloatParam, IntParam


class AudioTagging(Node):
    """
    AudioTagging node performs automatic audio tagging and embedding extraction from input audio data. It uses a pre-trained audio tagging model to identify the most likely audio classes present in the input, returning their probabilities and associated feature embeddings. The output tags are filtered by confidence and optionally by category.

    Inputs:
    - audioIn: 1D audio data array sampled at 32 kHz.

    Outputs:
    - tags: List of detected audio tag names as a newline-separated string.
    - probabilities: Array of confidence values for each detected tag.
    - embedding: Feature embedding array representing the input audio.
    """

    def config_input_slots():
        return {"audioIn": DataType.ARRAY}

    def config_output_slots():
        return {"tags": DataType.STRING, "probabilities": DataType.ARRAY, "embedding": DataType.ARRAY}

    def config_params():
        return {
            "selection": {
                "threshold": FloatParam(0.5, 0.0, 1.0),
                "number_of_tags": IntParam(3, 1, 527),
                "all_categories": BoolParam(True),
                "human_sounds": BoolParam(False),
                "animal": BoolParam(False),
                "music": BoolParam(False),
                "sounds_of_things": BoolParam(False),
                "natural_sounds": BoolParam(False),
                "source_ambiguous_sounds": BoolParam(False),
                "channel_environment_and_background": BoolParam(False),
            },
        }

    def setup(self):
        from panns_inference import AudioTagging, labels

        self.at = AudioTagging(checkpoint_path=None, device="cpu")
        self.labels = labels
        audio_tags_path = join(self.assets_path, "audio_tags_structure.pkl")
        with open(audio_tags_path, "rb") as f:
            self.audio_tags = pickle.load(f)

    def process(self, audioIn: Data):
        if audioIn.data is None:
            return None

        if audioIn.meta["sfreq"] != 32000:
            raise ValueError("Sampling frequency must be 32k")

        threshold = self.params["selection"]["threshold"].value
        n_tags = self.params["selection"]["number_of_tags"].value
        audioIn.data = audioIn.data.reshape(1, -1)

        (clipwise_output, embedding) = self.at.inference(audioIn.data)

        best_labels = np.argsort(clipwise_output[0])[::-1]
        tags = np.array(self.labels)[best_labels]
        probabilities = clipwise_output[0][best_labels]
        embedding = embedding.squeeze()

        if self.params["selection"]["all_categories"].value:
            # If all_categories is set to True, include all tags
            active_categories = [tag for sublist in self.audio_tags.values() for tag in sublist]
        else:
            active_categories = []
            if self.params["selection"]["human_sounds"].value:
                active_categories.extend(self.audio_tags["Human sounds"])
            if self.params["selection"]["animal"].value:
                active_categories.extend(self.audio_tags["Animal"])
            if self.params["selection"]["music"].value:
                active_categories.extend(self.audio_tags["Music"])
            if self.params["selection"]["sounds_of_things"].value:
                active_categories.extend(self.audio_tags["Sounds of things"])
            if self.params["selection"]["natural_sounds"].value:
                active_categories.extend(self.audio_tags["Natural sounds"])
            if self.params["selection"]["source_ambiguous_sounds"].value:
                active_categories.extend(self.audio_tags["Source-ambiguous sounds"])
            if self.params["selection"]["channel_environment_and_background"].value:
                active_categories.extend(self.audio_tags["Channel, environment and background"])

        # Filter based on active categories
        indices = [i for i, x in enumerate(tags) if x in active_categories and probabilities[i] > threshold]
        formatted_tags = "\n".join(tags[i] for i in indices[: n_tags - 1])
        formatted_probs = np.array([probabilities[i] for i in indices[: n_tags - 1]])

        return {
            "tags": (formatted_tags, audioIn.meta),
            "probabilities": (formatted_probs, audioIn.meta),
            "embedding": (embedding, audioIn.meta),
        }
