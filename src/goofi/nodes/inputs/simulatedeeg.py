import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, IntParam


class SimulatedEEG(Node):
    """
    This node simulates multi-channel EEG signals in real time, generating synthetic data with controllable background spectral exponents, oscillatory peaks, variances, and peak amplitudes per channel. The generated signal is output in chunks, suitable for real-time or streaming processing.

    Inputs:
    - exponents: Array of spectral exponent values, one per channel, controlling the power-law background of each channel's signal.
    - peaks: Array specifying oscillatory peak frequencies for each channel.
    - variances: Array specifying the variance of the background component for each channel.
    - peak_amplitudes: Array specifying the amplitude for each peak frequency in each channel.

    Outputs:
    - eeg_signal: Multichannel EEG signal data chunked over time, along with metadata specifying the sampling frequency.
    """

    def config_input_slots():
        return {
            "exponents": DataType.ARRAY,
            "peaks": DataType.ARRAY,
            "variances": DataType.ARRAY,  # Input to control variance per channel
            "peak_amplitudes": DataType.ARRAY,  # Input to control peak amplitudes per channel
        }

    def config_output_slots():
        return {"eeg_signal": DataType.ARRAY}

    def config_params():
        return {
            "signal": {
                "num_channels": IntParam(1, 1, 64, doc="Number of EEG channels."),
                "signal_length": IntParam(600, 10, 3600, doc="Total length of the signal in seconds."),
                "sampling_rate": IntParam(1000, 10, 10000, doc="Sampling rate of the signal in Hz."),
                "chunk_size": IntParam(10, 1, 1000, doc="Number of samples per output."),
                "default_variance": FloatParam(10.0, 0.1, 100.0, doc="Default variance for signal components."),
                "noise_factor": FloatParam(0.1, 0.0, 1.0, doc="Scale of noise added to channels."),
            },
        }

    def setup(self):
        from neurodsp.sim import sim_combined

        self.sim_combined = sim_combined

        # Retrieve parameters
        self.num_channels = self.params["signal"]["num_channels"].value
        self.signal_length = self.params["signal"]["signal_length"].value
        self.sampling_rate = self.params["signal"]["sampling_rate"].value
        self.chunk_size = self.params["signal"]["chunk_size"].value
        self.default_variance = self.params["signal"]["default_variance"].value
        self.noise_factor = self.params["signal"]["noise_factor"].value

        # Initialize the buffer for all channels
        self.buffer = None
        self.index = 0  # Initialize the buffer pointer

    def _generate_long_signal(self, exponents, peaks, variances, peak_amplitudes):
        """
        Generate multi-channel EEG signals with varying exponents, peaks, variances, and noise.

        Parameters
        ----------
        exponents : array
            Array of exponents for each channel.
        peaks : array
            Array of peak frequencies for each channel.
        variances : array
            Array of variances for each channel.
        peak_amplitudes : array
            Array of amplitudes for each peak per channel.

        Returns
        -------
        signals : ndarray
            Multi-channel simulated EEG signal.
        """
        signals = []
        for i in range(self.num_channels):
            # Set up simulation parameters for each channel
            components = {
                "sim_powerlaw": {"exponent": exponents[i]},
                "sim_oscillation": [{"freq": freq} for freq in peaks[i]],
            }

            # Use provided variance or default if not specified
            component_variances = [variances[i]] + [
                amp**2 for amp in peak_amplitudes[i]
            ]  # Square amplitude for variance scaling

            # Generate signal for the channel
            signal = self.sim_combined(self.signal_length, self.sampling_rate, components, component_variances)

            # Add noise to the signal
            noise = np.random.normal(0, self.noise_factor * np.std(signal), len(signal))
            signal += noise

            signals.append(signal)

        return np.array(signals)

    def process(self, exponents: Data, peaks: Data, variances: Data, peak_amplitudes: Data):
        """
        Process EEG signals by generating chunks based on input parameters.

        Parameters
        ----------
        exponents : Data
            Real-time array of exponents for each channel.
        peaks : Data
            Real-time array of peak frequencies for each channel.
        variances : Data
            Real-time array of variances for each channel.
        peak_amplitudes : Data
            Real-time array of peak amplitudes for each channel.

        Returns
        -------
        dict
            Contains the EEG signal chunk and metadata, including sampling frequency.
        """
        # Check for valid inputs, else use default
        if exponents is not None and exponents.data is not None:
            exponents = exponents.data
        else:
            exponents = np.full(self.num_channels, -1.0)

        if len(exponents) == 1 and self.num_channels > 1:
            exponents = np.full(self.num_channels, exponents[0])

        if peaks is not None and peaks.data is not None:
            peaks = peaks.data
        else:
            peaks = np.tile([2, 6, 12, 18, 24, 30], (self.num_channels, 1))

        # Ensure peaks are 2D (one set of peaks per channel)
        if peaks.ndim == 1:
            peaks = np.tile(peaks, (self.num_channels, 1))

        if variances is not None and variances.data is not None:
            variances = variances.data
        else:
            variances = np.full(self.num_channels, self.default_variance)

        if peak_amplitudes is not None and peak_amplitudes.data is not None:
            peak_amplitudes = peak_amplitudes.data
        else:
            peak_amplitudes = np.ones((self.num_channels, peaks.shape[1]))

        # Ensure consistency in parameters
        if peaks.shape[0] != self.num_channels:
            peaks = np.tile([2, 6, 12, 18, 24, 30], (self.num_channels, 1))
        if len(variances) != self.num_channels:
            variances = np.full(self.num_channels, self.default_variance)
        if peak_amplitudes.shape[0] != self.num_channels or peak_amplitudes.shape[1] != peaks.shape[1]:
            peak_amplitudes = np.ones((self.num_channels, peaks.shape[1]))

        # Generate a new signal buffer if none exists or if parameters changed
        if self.buffer is None or self.index == 0:
            self.buffer = self._generate_long_signal(exponents, peaks, variances, peak_amplitudes)

        # Retrieve the next chunk of data for each channel
        start_idx = self.index
        end_idx = start_idx + self.chunk_size

        # Handle buffer overflow by wrapping around
        if end_idx >= self.buffer.shape[1]:
            end_idx = self.buffer.shape[1]
            chunk = self.buffer[:, start_idx:end_idx]

            # Regenerate the buffer for seamless streaming
            self.buffer = self._generate_long_signal(exponents, peaks, variances, peak_amplitudes)
            self.index = 0
        else:
            chunk = self.buffer[:, start_idx:end_idx]
            self.index = end_idx

        # Return chunk with metadata including sampling frequency
        return {"eeg_signal": (chunk, {"sfreq": self.sampling_rate})}

        # Return chunk with metadata including sampling frequency
        return {"eeg_signal": (chunk, {"sfreq": self.sampling_rate})}

    def signal_num_channels_changed(self, _):
        """Reinitialize the stream if the number of channels changes."""
        self.setup()

    def signal_signal_length_changed(self, _):
        """Reinitialize the stream if the signal length changes."""
        self.setup()

    def signal_sampling_rate_changed(self, _):
        """Reinitialize the stream if the sampling rate changes."""
        self.setup()

    def signal_chunk_size_changed(self, _):
        """Reinitialize the stream if the chunk size changes."""
        self.setup()

    def signal_default_variance_changed(self, _):
        """Reinitialize the stream if the default variance changes."""
        self.setup()

    def signal_noise_factor_changed(self, _):
        """Reinitialize the stream if the noise factor changes."""
        self.setup()
