import cv2
import numpy as np

from goofi.data import Data, DataType
from goofi.image_utils import as_uint8
from goofi.node import Node
from goofi.params import FloatParam, StringParam


class EdgeDetector(Node):
    """
    This node performs edge detection on input images using either the Canny or Sobel method. It receives an image array, converts it to grayscale if necessary, applies the selected edge detection algorithm, and outputs the resulting edge map as a normalized array.

    Inputs:
    - image: Input image array, which can be a grayscale or color image.

    Outputs:
    - edges: Array representing the detected edges, normalized to the range [0, 1].
    """

    def config_input_slots():
        return {"image": DataType.ARRAY}  # Image as input

    def config_output_slots():
        return {"edges": DataType.ARRAY}  # Edges as output

    def config_params():
        return {
            "edge": {
                "method": StringParam("sobel", options=["canny", "sobel"], doc="Edge detection method"),
                "threshold1": FloatParam(50.0, 0.0, 300.0, doc="First threshold for the hysteresis procedure for canny"),
                "threshold2": FloatParam(150.0, 0.0, 300.0, doc="Second threshold for the hysteresis procedure for canny"),
            },
        }

    def process(self, image: Data):
        if image is None or image.data is None:
            return None

        # Extract method and thresholds from parameters
        method = self.params["edge"]["method"].value
        threshold1 = self.params["edge"]["threshold1"].value
        threshold2 = self.params["edge"]["threshold2"].value

        # If the image has 3 channels (color image), convert it to grayscale
        if image.data.shape[-1] == 3:
            gray_image = cv2.cvtColor(image.data, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image.data

        # Convert the image to 8-bit (values in [0, 255]). as_uint8 passes a
        # uint8 input through and scales a float [0,1] input — so the node
        # works whether the producer emits uint8 or float (A0).
        gray_image_8bit = as_uint8(gray_image)

        # Apply the chosen edge detection method
        if method == "canny":
            edges = cv2.Canny(gray_image_8bit, threshold1, threshold2)
        elif method == "sobel":
            grad_x = cv2.Sobel(gray_image_8bit, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_image_8bit, cv2.CV_32F, 0, 1, ksize=3)
            edges = cv2.magnitude(grad_x, grad_y)
            _, edges = cv2.threshold(edges, threshold1, 255, cv2.THRESH_BINARY)

        # Emit the edge map as uint8 [0,255] (A0). Both Canny (uint8) and the
        # thresholded Sobel magnitude (float {0,255}) are already in [0,255].
        edges = np.clip(edges, 0, 255).astype(np.uint8)

        return {"edges": (edges, {**image.meta})}
