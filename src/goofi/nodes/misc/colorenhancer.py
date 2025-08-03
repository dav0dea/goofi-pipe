import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam


class ColorEnhancer(Node):
    """
    Enhances a color or grayscale image by applying a combination of contrast, brightness, gamma correction, and optional color boosting. The node expects the input image data to be normalized to the [0, 1] range for correct operation.

    Inputs:
    - image: A NumPy array representing the input image, either grayscale or color (with 3 channels), with values normalized between 0 and 1.

    Outputs:
    - enhanced_image: The processed image array after enhancement, with the same shape as the input, and values clipped to the [0, 1] range.
    """

    def config_input_slots():
        return {"image": DataType.ARRAY}

    def config_output_slots():
        return {"enhanced_image": DataType.ARRAY}

    def config_params():
        return {
            "enhancement": {
                "contrast": FloatParam(1.0, 0.1, 10.0, doc="Multiplier for contrast adjustment. 1.0 means no change."),
                "brightness": FloatParam(0.0, -0.9, 0.9, doc="Additive brightness adjustment. 0.0 means no change."),
                "gamma": FloatParam(1.0, 0.1, 5.0, doc="Gamma correction value. 1.0 means no change."),
                "color_boost": FloatParam(1.0, 0.5, 3.0, doc="Multiplier for boosting colors. 1.0 means no change."),
            }
        }

    def process(self, image: Data):
        if image is None or image.data is None:
            return None

        # Extract parameters
        contrast = self.params["enhancement"]["contrast"].value
        brightness = self.params["enhancement"]["brightness"].value
        gamma = self.params["enhancement"]["gamma"].value
        color_boost = self.params["enhancement"]["color_boost"].value

        # Find the mean intensity
        midpoint = 0.5

        # Adjust contrast around the mean value
        enhanced_image = (image.data - midpoint) * contrast + midpoint + brightness

        # Adjust gamma
        enhanced_image = np.clip(np.power(enhanced_image, gamma), 0, 1)

        # Boost colors (only if the image is colored)
        if enhanced_image.shape[-1] == 3:
            enhanced_image *= color_boost
            enhanced_image = np.clip(enhanced_image, 0, 1)

        return {"enhanced_image": (enhanced_image, {**image.meta})}
