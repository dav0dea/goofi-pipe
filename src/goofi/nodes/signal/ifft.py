import numpy as np
from numpy.fft import ifft
from goofi.data import Data, DataType
from goofi.node import Node


class IFFT(Node):
    """
    This node performs an inverse Fast Fourier Transform (IFFT) to reconstruct a time-domain signal from its magnitude (spectrum) and phase data. It accepts separate arrays for the spectrum and phase, combines them into complex frequency-domain data, and then computes the time-domain signal using the inverse FFT. If the input arrays differ in length, the shorter array is zero-padded to match the longer one before processing.

    Inputs:
    - spectrum: Array containing the magnitude (spectrum) values of the frequency-domain signal.
    - phase: Array containing the phase values corresponding to the frequency-domain signal.

    Outputs:
    - reconstructed: Array containing the reconstructed time-domain signal obtained from the IFFT, along with the metadata from the phase input.
    """

    def config_input_slots():
        return {"spectrum": DataType.ARRAY, "phase": DataType.ARRAY}

    def config_output_slots():
        return {"reconstructed": DataType.ARRAY}

    def config_params():
        return {}  # No parameters for this node

    def process(self, spectrum: Data, phase: Data):
        # Check if the input is provided
        if spectrum is None or spectrum.data is None or phase is None or phase.data is None:
            return None

        # Check if the lengths of spectrum and phase are different
        if len(spectrum.data) != len(phase.data):
            # Calculate the difference in lengths
            length_diff = abs(len(spectrum.data) - len(phase.data))

            # Check which data is shorter and pad accordingly
            if len(spectrum.data) < len(phase.data):
                # Pad the spectrum data with zeros (or another value) to match the phase data length
                padding = np.zeros(length_diff)
                spectrum.data = np.concatenate([spectrum.data, padding])
            else:
                # Pad the phase data with zeros (or another value) to match the spectrum data length
                padding = np.zeros(length_diff)
                phase.data = np.concatenate([phase.data, padding])

        # Create complex numbers from magnitude (spectrum) and phase
        complex_data = spectrum.data * np.exp(1j * phase.data)

        # Inverse Fourier Transform to get the time series
        time_series = np.real(ifft(complex_data))

        # For this example, I'm not adjusting or copying metadata, but you can do so as needed
        return {"reconstructed": (time_series, phase.meta)}
