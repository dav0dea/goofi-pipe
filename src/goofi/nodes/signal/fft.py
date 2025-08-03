from copy import deepcopy

import numpy as np
from numpy.fft import fft, fftfreq

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import StringParam


class FFT(Node):
    """
    This node computes the Fast Fourier Transform (FFT) of the input data, allowing analysis of frequency components in either time series or image data. For time series, it processes 1D or 2D arrays and outputs the magnitude and phase spectra for each frequency. For images, it processes grayscale or RGB images and outputs the magnitude and phase spectra over spatial frequencies.

    Inputs:
    - data: The input array to be transformed, either as a time series (1D or 2D array with sampling frequency in metadata) or as an image (2D grayscale or 3D RGB array).

    Outputs:
    - mag: The magnitude spectrum of the FFT of the input data.
    - phase: The phase spectrum of the FFT of the input data.
    """

    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {"mag": DataType.ARRAY, "phase": DataType.ARRAY}

    def config_params():
        return {
            "fft": {
                "input_type": StringParam("time_series", options=["time_series", "image"]),
            }
        }

    def process(self, data: Data):
        if data is None or data.data is None:
            return None

        input_type = self.params["fft"]["input_type"].value

        if input_type == "time_series":
            if data.data.ndim not in [1, 2]:
                raise ValueError("Data must be 1D or 2D")
            sfreq = data.meta["sfreq"]
            freq = fftfreq(data.data.shape[-1], 1 / sfreq)
            fft_result = fft(data.data, axis=-1)
            psd = np.abs(fft_result)
            phase = np.angle(fft_result)

            meta = deepcopy(data.meta)
            if data.data.ndim == 1:
                meta["channels"]["dim0"] = freq.tolist()
            else:  # if 2D
                meta["channels"]["dim1"] = freq.tolist()

        elif input_type == "image":
            if data.data.ndim == 2:  # Grayscale image
                fft_result = np.fft.fft2(data.data)
            elif data.data.ndim == 3 and data.data.shape[2] == 3:  # RGB image
                fft_result = np.stack([np.fft.fft2(data.data[:, :, channel]) for channel in range(3)], axis=-1)
            else:
                raise ValueError("Unsupported image shape for FFT. Supported shapes are (x, y) or (x, y, 3)")

            psd = np.abs(fft_result)
            phase = np.angle(fft_result)

            # Meta data remains the same for image since we're not really modifying the "channels" in the image case
            meta = deepcopy(data.meta)

        return {"mag": (psd, meta), "phase": (phase, meta)}
