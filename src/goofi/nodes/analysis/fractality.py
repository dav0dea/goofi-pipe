import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, IntParam, StringParam


class Fractality(Node):
    """
    This node computes the fractal dimension or fractal characteristics of input data using a variety of methods, including several signal-based estimators (such as Hurst and Higuchi) and a box counting method for 2D arrays. It is applicable to both time series and image data, depending on the selected method.

    Inputs:
    - data_input: An array representing either a time series (1D or 2D) or a 2D image (grayscale or RGB).

    Outputs:
    - fractal_dimension: The calculated fractal dimension or feature value based on the chosen method.
    """

    def config_input_slots():
        return {"data_input": DataType.ARRAY}

    def config_output_slots():
        return {"fractal_dimension": DataType.ARRAY}

    def config_params():
        return {
            "method": {
                "name": StringParam(
                    "hurst",
                    options=[
                        "hurst",
                        "fractal_correlation",
                        "fractal_petrosian",
                        "fractal_psdslope",
                        "fractal_higuchi",
                        "fractal_katz",
                        "fractal_linelength",
                        "fractal_nld",
                        "box_counting_2d",
                    ],
                ),
            },
            "box_counting": {
                "scales_base": FloatParam(2.0, 1.1, 10.0, doc="Base for the logspace calculation in box_counting"),
                "scales_start": FloatParam(0.01, 0.001, 1.0, doc="Start of the logspace in box_counting"),
                "scales_num": IntParam(10, 5, 100, doc="Number of steps in the logspace for box_counting"),
            },
            "fractal_higuchi": {"k_max": IntParam(10, 1, 100, doc="Maximum k value for fractal_higuchi method")},
            "fractal_correlation": {
                "delay": IntParam(1, 1, 100, doc="Delay for fractal_correlation method"),
                "dimension": IntParam(2, 1, 100, doc="Dimension for fractal_correlation method"),
            },
        }

    def setup(self):
        import neurokit2 as nk

        self.nk = nk

    def process(self, data_input: Data):
        if data_input is None or data_input.data is None:
            return None

        method = self.params["method"]["name"].value

        # For methods in neurokit2
        if method == "fractal_katz":
            result, _ = self.nk.fractal_katz(data_input.data)
        elif method == "fractal_petrosian":
            result, _ = self.nk.fractal_petrosian(data_input.data)
        elif method == "fractal_linelength":
            result, _ = self.nk.fractal_linelength(data_input.data)
        elif method == "fractal_psdslope":
            result, _ = self.nk.fractal_psdslope(data_input.data)
        elif method == "fractal_nld":
            result, _ = self.nk.fractal_nld(data_input.data)
        elif method == "fractal_higuchi":
            result, _ = self.nk.fractal_higuchi(data_input.data, k_max=self.params["fractal_higuchi"]["k_max"].value)
        elif method == "hurst":
            result, _ = self.nk.fractal_hurst(data_input.data)
        elif method == "fractal_correlation":
            result, _ = self.nk.fractal_correlation(data_input.data)
        elif method == "box_counting_2d":
            result = self.box_counting(data_input.data)
            print(result)

        return {"fractal_dimension": (result, {})}

    def box_counting(self, data):
        # Check if data is either 2D or 3D with shape (x, y, 3)
        if len(data.shape) not in [2, 3] or (len(data.shape) == 3 and data.shape[2] != 3):
            raise ValueError("Data for box_counting should be either 2D or have a shape of (x, y, 3)")

        # Convert from RGB to grayscale if data is 3D
        if len(data.shape) == 3:
            data = self.rgb2gray(data)
        # Threshold the data to create a binary representation
        threshold = np.mean(data)
        data = (data > threshold).astype(int)

        if len(data.shape) != 2:
            raise ValueError("Data for box_counting should be 2D after any necessary conversions")

        # Find all the non-zero pixels (modify this if your image isn't binary)
        pixels = np.column_stack(np.where(data > 0))

        # Retrieve the scales parameters
        scales_base = self.params["box_counting"]["scales_base"].value
        scales_start = self.params["box_counting"]["scales_start"].value
        scales_num = self.params["box_counting"]["scales_num"].value

        scales = np.logspace(scales_start, 1, num=scales_num, endpoint=False, base=scales_base)
        Ns = []

        for scale in scales:
            H, _ = np.histogramdd(pixels, bins=(np.arange(0, data.shape[0], scale), np.arange(0, data.shape[1], scale)))
            Ns.append(np.sum(H > 0))

        coeffs = np.polyfit(np.log(scales), np.log(Ns), 1)
        return -coeffs[0]

    def rgb2gray(self, rgb):
        gray = 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]
        return gray / 255  # Normalize to [0,1]


# https://github.com/ChatzigeorgiouGroup/FractalDimension/blob/master/FractalDimension.py
