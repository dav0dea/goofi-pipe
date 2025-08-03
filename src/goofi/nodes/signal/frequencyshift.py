import numpy as np
from goofi.data import DataType
from goofi.node import Node
from goofi.params import FloatParam


class FrequencyShift(Node):
    """
    Shifts the frequency content of an input signal by a specified amount using the FFT frequency shifting method. The node takes a time-domain signal, converts it to the frequency domain, shifts the spectrum, and converts it back to the time domain.

    Inputs:
    - data: An array representing the input signal to shift. Must include metadata with the sampling frequency ("sfreq").

    Outputs:
    - out: The frequency-shifted signal as an array, with the original metadata preserved.
    """

    @staticmethod
    def config_input_slots():
        return {"data": DataType.ARRAY}

    @staticmethod
    def config_output_slots():
        return {"out": DataType.ARRAY}

    @staticmethod
    def config_params():
        return {
            "shift": {
                "frequency_shift": FloatParam(1.0, -1000.0, 1000.0, doc="Frequency shift in Hz"),
            }
        }

    def process(self, data):
        # early exit
        if data is None or data.data is None:
            return None

        # flatten to 1D
        signal = np.asarray(data.data).flatten()
        sfreq = data.meta.get("sfreq")
        if sfreq is None:
            raise ValueError("data.meta must include sampling frequency 'sfreq'.")

        n = signal.shape[0]
        freq_shift = self.params["shift"]["frequency_shift"].value

        # compute how many FFT bins to shift
        delta_bins = int(round(freq_shift * n / sfreq))

        # perform FFT
        spectrum = np.fft.fft(signal)

        # allocate shifted spectrum
        shifted_spectrum = np.zeros_like(spectrum)

        # positive shift => move content upward
        if delta_bins > 0:
            shifted_spectrum[delta_bins:] = spectrum[:-delta_bins]
        # negative shift => move content downward
        elif delta_bins < 0:
            shifted_spectrum[:delta_bins] = spectrum[-delta_bins:]
        else:
            shifted_spectrum[:] = spectrum

        # invert back to time domain
        shifted_signal = np.fft.ifft(shifted_spectrum)
        # because spectrum shift breaks conjugate symmetry, take real part
        shifted_signal = np.real(shifted_signal)

        # trim/pad to original length (just in case)
        shifted_signal = shifted_signal[:n]

        return {"out": (shifted_signal, data.meta)}
