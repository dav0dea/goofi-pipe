import colorsys

import numpy as np
import webcolors
import webcolors._definitions

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import IntParam, StringParam


class TuningColors(Node):
    """
    This node converts a musical scale represented as an array of frequency ratios into corresponding HSV color values, based on the frequency and averaged consonance of each step in the scale. The hue represents mapped frequency, saturation encodes consonance, and the value (brightness) is fixed. Additionally, the node provides readable color names or HEX codes for the first few colors calculated from the scale.

    Inputs:
    - data: 1D array representing a musical scale, where the first element is the fundamental frequency in Hz and the rest are frequency ratios.

    Outputs:
    - hue: Array of hue values (float), one for each note in the scale except the fundamental, representing converted pitch.
    - saturation: Array of saturation values (float), encoding the consonance for each scale step.
    - value: Array of value (brightness) values (float), set to a constant for each scale step.
    - color_names: String containing the names or HEX codes of the first few color representations corresponding to the scale notes (excluding the fundamental).
    """

    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {"hue": DataType.ARRAY, "saturation": DataType.ARRAY, "value": DataType.ARRAY, "color_names": DataType.STRING}

    def config_params():
        return {
            "Biocolors": {
                "color_names_mode": StringParam("name", options=["name", "HEX"]),
                "n_first_colors": IntParam(3, 1, 6),
            }
        }

    def setup(self):
        from biotuner.biocolors import audible2visible, scale2freqs, wavelength_to_rgb
        from biotuner.biotuner_object import dyad_similarity
        from biotuner.metrics import tuning_cons_matrix

        self.audible2visible = audible2visible
        self.scale2freqs = scale2freqs
        self.wavelength_to_rgb = wavelength_to_rgb
        self.dyad_similarity = dyad_similarity
        self.tuning_cons_matrix = tuning_cons_matrix

    def process(self, data: Data):
        """
        Convert a musical scale into a list of HSV colors based on the scale's frequency values
        and their averaged consonance.
        """
        if data is None:
            return None

        if data.data.ndim > 1:
            raise ValueError("Data must be 1D")

        scale = data.data  # a list of frequency ratios representing the musical scale
        fund = data.data[0]  # fundamental frequency of the scale in Hz

        min_ = 0
        max_ = 1
        # convert the scale to frequency values
        scale_freqs = self.scale2freqs(scale, fund)
        # compute the averaged consonance of each step
        scale_cons, _, _ = self.tuning_cons_matrix(scale, self.dyad_similarity, ratio_type="all")
        # rescale to match RGB standards (0, 255)
        scale_cons = (np.array(scale_cons) - min_) * (1 / max_ - min_) * 255
        scale_cons = scale_cons.astype("uint8").astype(float) / 255

        hsv_all = []
        for s, cons in zip(scale_freqs, scale_cons):
            # convert freq in nanometer values
            _, _, nm, octave = self.audible2visible(s)
            # convert to RGB values
            rgb = self.wavelength_to_rgb(nm)
            # convert to HSV values
            # TODO: colorsys might be slow
            hsv = colorsys.rgb_to_hsv(rgb[0] / float(255), rgb[1] / float(255), rgb[2] / float(255))
            hsv = np.array(hsv)
            # rescale
            hsv = (hsv - 0) * (1 / (1 - 0))
            # define the saturation
            hsv[1] = cons
            # define the luminance
            hsv[2] = 200 / 255
            hsv = tuple(hsv)
            hsv_all.append(hsv)

        # hsv_all is a list of HSV color tuples, one for each scale step, with hue representing the
        # frequency value, saturation representing the consonance, and luminance set to a fixed value
        hsvs = hsv_all[1:]
        color_names = []
        color_names_mode = self.params["Biocolors"]["color_names_mode"].value
        if color_names_mode == "name":
            for hsv in hsvs:
                rgb = tuple(map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*hsv)))
                color_names.append(rgb2name(rgb))
            color_names = color_names[: self.params["Biocolors"]["n_first_colors"].value]
            color_names = " ".join(color_names)

        elif color_names_mode == "HEX":
            for hsv in hsvs:
                rgb = tuple(map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*hsv)))
                hex_value = "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])
                color_names.append(hex_value)
            color_names = color_names[: self.params["Biocolors"]["n_first_colors"].value]
            color_names = " ".join(color_names)

        # select n_first_colors

        return {
            "hue": (np.array([x[0] for x in hsvs]), data.meta),
            "saturation": (np.array([x[1] for x in hsvs]), data.meta),
            "value": (np.array([x[2] for x in hsvs]), data.meta),
            "color_names": (color_names, data.meta),
        }


def rgb2name(rgb):
    """
    Find the closest color in a dictionary of colors to an input RGB value.

    Parameters:
        rgb (Tuple[int, int, int]): RGB color tuple

    Returns:
        str: The name of the closest color in the dictionary
    """
    colors = {k: webcolors.hex_to_rgb(k) for k in webcolors._definitions._CSS3_HEX_TO_NAMES.keys()}
    closest_color = min(
        colors,
        key=lambda color: sum((a - b) ** 2 for a, b in zip(rgb, colors[color])),
    )
    return webcolors._definitions._CSS3_HEX_TO_NAMES[closest_color]
