from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, StringParam


class Audiocraft(Node):
    """
    This node generates audio waveforms from a given text prompt using pretrained generative models. It leverages AudioGen or MusicGen models to synthesize audio based on the input description.

    Inputs:
    - prompt: A string containing the textual description or prompt for the desired audio.

    Outputs:
    - wav: An array representing the generated waveform audio data for the provided prompt. The output also includes metadata for the audio's sampling rate.
    """

    def config_params():
        return {
            "audiocraft": {
                "model": StringParam(
                    "facebook/audiogen-medium",
                    options=[
                        "facebook/audiogen-medium",
                        "facebook/musicgen-small",
                        "facebook/musicgen-medium",
                        "facebook/musicgen-large",
                    ],
                ),
                "device": StringParam("cuda", options=["cuda", "cpu"]),
            },
            "setup": {"install": BoolParam(False, trigger=True)},
        }

    def config_input_slots():
        return {"prompt": DataType.STRING}

    def config_output_slots():
        return {"wav": DataType.ARRAY}

    def setup(self):
        try:
            from audiocraft.models import AudioGen, MusicGen
        except ImportError:
            raise ImportError(
                "Install audiocraft using the 'install' button or according to the instructions here: "
                "https://github.com/facebookresearch/audiocraft/tree/main?tab=readme-ov-file#installation"
            )

        try:
            import torch
        except ImportError:
            raise ImportError(
                "Please first install torch according to these instructions: https://pytorch.org/get-started/locally/"
            )

        torch.set_grad_enabled(False)

        self.current_model = self.params.audiocraft.model.value

        if "audiogen" in self.params.audiocraft.model.value:
            self.model = AudioGen.get_pretrained(self.params.audiocraft.model.value, device=self.params.audiocraft.device.value)
        elif "musicgen" in self.params.audiocraft.model.value:
            self.model = MusicGen.get_pretrained(self.params.audiocraft.model.value, device=self.params.audiocraft.device.value)
        else:
            raise ValueError(f"Model {self.params.audiocraft.model.value} not found")

    def process(self, prompt: Data):
        wav = self.model.generate([prompt.data])[0].cpu().numpy()
        prompt.meta["sampling_rate"] = self.model.sample_rate
        return {"wav": (wav, prompt.meta)}

    def audiocraft_model_changed(self, model):
        if model != self.current_model:
            self.setup()

    def setup_install_changed(self, install):
        if not install:
            return

        import subprocess
        import tempfile

        import requests

        req = requests.get("https://raw.githubusercontent.com/facebookresearch/audiocraft/main/requirements.txt").text

        # remove version from torch to avoid messing up the current installation
        req = req.replace("torch==2.1.0\n", "")
        try:
            import torch  # noqa
        except ImportError:
            raise ImportError(
                "Please first install torch according to these instructions: https://pytorch.org/get-started/locally/"
            )

        # remove xformers to install independently to avoid not finding a compatible version
        req = req.replace("xformers<0.0.23\n", "")

        # write requirements to a temp file and install them
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(req.encode())
            req_file = f.name
        subprocess.run(["pip", "install", "-r", req_file])

        # install xformers and torchmetrics independently
        subprocess.run(["pip", "install", "xformers", "torchmetrics"])

        # update torchvision and torchaudio if new torch was installed
        subprocess.run(["pip", "install", "-U", "torchvision", "torchaudio"])

        # install audiocraft without dependencies
        subprocess.run(["pip", "install", "audiocraft", "--no-deps"])
