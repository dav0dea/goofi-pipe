from os.path import join, exists
from os import makedirs
from typing import Any, Dict, Tuple

import cv2
import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, FloatParam, IntParam, StringParam


class ImageGeneration(Node):
    """
    This node performs AI-based image generation by interfacing with either local Stable Diffusion (SD) or cloud-based DALL-E models. It supports both text-to-image (txt2img) and image-to-image (img2img) generation depending on model capabilities and user settings. The node can automatically manage image state for iterative workflows, and optionally saves generated images to disk.

    Inputs:
    - prompt: The main text prompt describing the desired content of the generated image.
    - negative_prompt: An optional text prompt specifying aspects to avoid in the generated image.
    - base_image: An optional input image used as a starting point or reference for image-to-image generation.

    Outputs:
    - img: The generated image as a NumPy array, along with associated prompt information.
    """

    def config_params():
        return {
            "image_generation": {
                "save_image": BoolParam(False, doc="Save the images in NumPy Arrays"),
                "save_filename": StringParam("goofi", doc="Filename to save the image as"),
                "model_id": StringParam("dall-e-3", options=["stabilityai/stable-diffusion-2-1", "dall-e-2", "dall-e-3"]),
                "openai_key": StringParam("openai.key"),
                "inference_steps": IntParam(50, 5, 100),
                "guidance_scale": FloatParam(7.5, 0.1, 20),
                "use_fixers": BoolParam(True, doc="Use textural inversion (nfixer and nrealfixer) for better image quality."),
                "seed": IntParam(-1, -1, 1000000, doc="-1 for random seed"),
                "width": IntParam(512, 100, 1024),
                "height": IntParam(512, 100, 1024),
                "scheduler": StringParam(
                    list(SCHEDULER_MAPPING.keys())[0],
                    options=list(SCHEDULER_MAPPING.keys()),
                ),
                "device": "cuda",
            },
            "img2img": {
                "enabled": False,
                "strength": FloatParam(0.8, 0, 1),
                "resize_input": True,
                "reset_image": BoolParam(False, trigger=True),
            },
        }

    def config_input_slots():
        return {"prompt": DataType.STRING, "negative_prompt": DataType.STRING, "base_image": DataType.ARRAY}

    def config_output_slots():
        return {"img": DataType.ARRAY}

    def setup(self):
        if self.params.image_generation.model_id.value == "stabilityai/stable-diffusion-2-1":
            self.torch, self.diffusers = import_libs("stabilityai/stable-diffusion-2-1")
            # load StableDiffusion model
            if self.params.img2img.enabled.value:
                # TODO: make sure this works without internet access
                self.sd_pipe = self.diffusers.StableDiffusionImg2ImgPipeline.from_pretrained(
                    self.params.image_generation.model_id.value, torch_dtype=self.torch.float16
                )
            else:
                # TODO: make sure this works without internet access
                self.sd_pipe = self.diffusers.StableDiffusionPipeline.from_pretrained(
                    self.params.image_generation.model_id.value, torch_dtype=self.torch.float16
                )
            # set device
            self.sd_pipe.to(self.params.image_generation.device.value)

            # initialize scheduler
            self.image_generation_scheduler_changed(self.params.image_generation.scheduler.value)

            # load textural inversions
            if self.params.image_generation.use_fixers.value:
                self.sd_pipe.load_textual_inversion(join(self.assets_path, "nfixer.pt"))
                self.sd_pipe.load_textual_inversion(join(self.assets_path, "nrealfixer.pt"))

            # initialize last image
            if not hasattr(self, "last_img"):
                self.last_img = None
                self.reset_last_img()
        elif self.params.image_generation.model_id.value in ["dall-e-2", "dall-e-3"]:
            self.base64, self.openai = import_libs("dall-e")

            api_key = self.params.image_generation.openai_key.value
            with open(api_key, "r") as f:
                api_key = f.read().strip()
            # load Dall-E model
            if self.params.img2img.enabled.value:
                self.dalle_pipe = self.openai.OpenAI(api_key=api_key).images.edit
            else:
                self.dalle_pipe = self.openai.OpenAI(api_key=api_key).images.generate
            # initialize last image
            if not hasattr(self, "last_img"):
                self.last_img = None
                self.reset_last_img()
        else:
            raise ValueError(f"Unknown model: {self.params.image_generation.model_id.value}")

    def process(self, prompt: Data, negative_prompt: Data, base_image: Data) -> Dict[str, Tuple[np.ndarray, Dict[str, Any]]]:
        if prompt is None:
            return None
        prompt = prompt.data

        if self.params.img2img.reset_image.value:
            # reset the last image to zeros
            self.reset_last_img()

        if self.params.img2img.enabled.value:
            # update the last image
            if base_image is None:
                base_image = self.last_img
            else:
                base_image = base_image.data
            # resize
            if self.params.img2img.resize_input.value:
                # resize input image to match the last image
                base_image = cv2.resize(
                    base_image, (self.params.image_generation.width.value, self.params.image_generation.height.value)
                )
                if base_image.ndim == 3:
                    # add batch dimension
                    base_image = np.expand_dims(base_image, 0)

        # remote pipes
        if self.params.image_generation.model_id.value in ["dall-e-3", "dall-e-2"]:
            size = f"{self.params.image_generation.width.value}x{self.params.image_generation.height.value}"
            if self.params.img2img.enabled.value:
                # raise error because img2img is not working yet
                raise NotImplementedError("img2img is not working yet.")
                # convert to uint8
                base_image = (base_image * 255).astype(np.uint8)
                base_image = cv2.cvtColor(base_image, cv2.COLOR_RGB2RGBA)
                # convert to bytes
                base_image = cv2.imencode(".png", base_image)[1].tobytes()
                # run the Dall-E img2img pipeline
                response = self.dalle_pipe(
                    image=base_image,
                    prompt=prompt,
                    n=1,
                    size=size,
                    response_format="b64_json",
                )
            else:
                try:
                    # run the Dall-E txt2img pipeline
                    response = self.dalle_pipe(
                        model=self.params.image_generation.model_id.value,
                        prompt=prompt,
                        n=1,
                        size=size,
                        quality="standard",
                        response_format="b64_json",
                    )
                except self.openai.BadRequestError as e:
                    if e.response.status_code == 400:
                        raise RuntimeError(
                            f"Error code 400: the size of the image is not supported by the model."
                            f"\nYour Model: {self.params.image_generation.model_id.value}"
                            f"\n1024x1024 is minimum for Dall-E3"
                        )
                    raise e

            img = response.data[0].b64_json
            # Decode base64 to bytes
            decoded_bytes = self.base64.b64decode(img)
            # Convert bytes to numpy array using OpenCV
            img_array = cv2.imdecode(np.frombuffer(decoded_bytes, np.uint8), cv2.IMREAD_COLOR)

            # save numpy array using OpenCV
            if self.params.image_generation.save_image.value:
                makedirs(join(self.assets_path, "imgs"), exist_ok=True)

                # make sure not to overwrite
                filename = f"{join(self.assets_path, 'imgs', self.params.image_generation.save_filename.value)}"
                n = 0
                while exists(join(self.assets_path, "imgs", f"default_value_{n:02d}.png")):
                    n += 1
                # Save the image and check if it was successful
                if cv2.imwrite(join(self.assets_path, "imgs", f"default_value_{n:02d}.png"), img_array):
                    print(f"{join(self.assets_path,'imgs', f'default_value_{n:02d}.png')}")
                else:
                    print(f"Failed to save image to {filename}_{n:02d}.png")

            # Convert to float32 to display in goofi
            img_array = img_array.astype(np.float32) / 255.0
            # Ensure correct shape
            if img_array.shape != (self.params.image_generation.width.value, self.params.image_generation.height.value, 3):
                img_array = cv2.resize(
                    img_array, (self.params.image_generation.width.value, self.params.image_generation.height.value)
                )
            # Add batch dimension
            img = np.expand_dims(img_array, 0)
            # save last image
            self.last_img = np.array(img[0])
            return {
                "img": (img, {"prompt": prompt, "negative_prompt": negative_prompt if negative_prompt is not None else None})
            }

        # local pipes
        elif self.params.image_generation.model_id.value == "stabilityai/stable-diffusion-2-1":
            # set seed
            if self.params.image_generation.seed.value != -1:
                self.torch.manual_seed(self.params.image_generation.seed.value)

            # add textural inversions to the negative prompt
            negative_prompt = negative_prompt.data if negative_prompt is not None else ""
            if self.params.image_generation.use_fixers.value:
                negative_prompt = " ".join([negative_prompt, "nfixer nrealfixer"])

            with self.torch.inference_mode():
                if self.params.img2img.enabled.value:
                    if base_image is None:
                        base_image = self.last_img
                    else:
                        base_image = base_image.data

                    # run the img2img stable diffusion pipeline
                    img, _ = self.sd_pipe(
                        image=base_image,
                        strength=self.params.img2img.strength.value,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=self.params.image_generation.inference_steps.value,
                        guidance_scale=self.params.image_generation.guidance_scale.value,
                        return_dict=False,
                        output_type="np",
                    )
                else:
                    if base_image is not None:
                        raise ValueError(
                            "base_image is not supported in text2img mode. Enable img2img or disconnect base_image."
                        )

                    # run the text2img stable diffusion pipeline
                    img, _ = self.sd_pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        width=self.params.image_generation.width.value,
                        height=self.params.image_generation.height.value,
                        num_inference_steps=self.params.image_generation.inference_steps.value,
                        guidance_scale=self.params.image_generation.guidance_scale.value,
                        return_dict=False,
                        output_type="np",
                    )

            # remove the batch dimension
            img = np.array(img[0])
            # save last image
            self.last_img = img
            # save numpy array ; do not overwrite
            if self.params.image_generation.save_image.value:
                filename = f"{self.assets_path}/imgs/{prompt}"
                n = 0
                while exists(f"{filename}_{n:02d}.npy"):
                    n += 1

                np.save(f"{filename}_{n:02d}.npy", img)

            return {
                "img": (
                    img,
                    {"prompt": prompt, "negative_prompt": negative_prompt if negative_prompt is not None else None},
                )
            }

    def reset_last_img(self):
        """Reset the last image."""
        self.last_img = np.zeros((self.params.image_generation.height.value, self.params.image_generation.width.value, 3))

    def image_generation_use_fixers_changed(self, value):
        """Load the textural inversions."""
        if value:
            # load the textural inversions
            self.sd_pipe.load_textual_inversion(join(self.assets_path, "nfixer.pt"))
            self.sd_pipe.load_textual_inversion(join(self.assets_path, "nrealfixer.pt"))
        else:
            # reload the model without the textural inversions
            self.setup()

    def image_generation_scheduler_changed(self, value):
        """Change the scheduler of the Stable Diffusion pipeline."""
        scheduler_settings = dict(SCHEDULER_MAPPING[value])
        scheduler_type = scheduler_settings.pop("_sched")
        self.sd_pipe.scheduler = getattr(self.diffusers, scheduler_type).from_config(
            self.sd_pipe.scheduler.config, **scheduler_settings
        )

    def image_generation_width_changed(self, value):
        """Resize the last image to match the new width (for img2img)."""
        self.last_img = cv2.resize(self.last_img, (value, self.params.image_generation.height.value))

    def image_generation_height_changed(self, value):
        """Resize the last image to match the new height (fog img2img)."""
        self.last_img = cv2.resize(self.last_img, (self.params.image_generation.width.value, value))

    def img2img_enabled_changed(self, value):
        """Load the correct Stable Diffusion pipeline."""
        self.setup()


def import_libs(checks):
    if checks == "stabilityai/stable-diffusion-2-1":
        try:
            import torch
        except ImportError:
            raise ImportError(
                "You need to install torch to use the ImageGeneration node with Stable Diffusion: "
                "https://pytorch.org/get-started/locally/"
            )
        try:
            import diffusers
        except ImportError:
            raise ImportError(
                "You need to install diffusers to use the ImageGeneration node with Stable Diffusion: pip install diffusers"
            )
        return torch, diffusers
    elif checks == "dall-e":
        try:
            import openai
        except ImportError:
            raise ImportError("You need to install openai to use the ImageGeneration node with Dall-E: pip install openai")
        try:
            import base64
        except ImportError:
            raise ImportError("You need to import base64 to use the ImageGeneration node with Dall-E")
        return base64, openai
    else:
        raise ValueError(f"Unknown model: {checks}")


SCHEDULER_MAPPING = {
    "DPM++ 2M": dict(_sched="DPMSolverMultistepScheduler"),
    "DPM++ 2M Karras": dict(_sched="DPMSolverMultistepScheduler", use_karras_sigmas=True),
    "DPM++ 2M SDE": dict(_sched="DPMSolverMultistepScheduler", algorithm_type="sde-dpmsolver++"),
    "DPM++ 2M SDE Karras": dict(_sched="DPMSolverMultistepScheduler", use_karras_sigmas=True, algorithm_type="sde-dpmsolver++"),
    "DPM++ SDE": dict(_sched="DPMSolverSinglestepScheduler"),
    "DPM++ SDE Karras": dict(_sched="DPMSolverSinglestepScheduler", use_karras_sigmas=True),
    "DPM2": dict(_sched="KDPM2DiscreteScheduler"),
    "DPM2 Karras": dict(_sched="KDPM2DiscreteScheduler", use_karras_sigmas=True),
    "DPM2 a": dict(_sched="KDPM2AncestralDiscreteScheduler"),
    "DPM2 a Karras": dict(_sched="KDPM2AncestralDiscreteScheduler", use_karras_sigmas=True),
    "Euler": dict(_sched="EulerDiscreteScheduler"),
    "Euler a": dict(_sched="EulerAncestralDiscreteScheduler"),
    "Heun": dict(_sched="HeunDiscreteScheduler"),
    "LMS": dict(_sched="LMSDiscreteScheduler"),
    "LMS Karras": dict(_sched="LMSDiscreteScheduler", use_karras_sigmas=True),
}
