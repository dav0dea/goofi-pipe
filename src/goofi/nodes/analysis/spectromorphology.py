from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import IntParam, StringParam


class SpectroMorphology(Node):
    """
    This node calculates spectromorphological features on a 1D audio signal. It analyzes the input time series and extracts a selected spectral feature over time, returning the feature values aligned with their corresponding timestamps.

    Inputs:
    - data: A 1D array containing the audio signal to be analyzed, along with metadata such as the sampling frequency.

    Outputs:
    - spectro: A tuple consisting of an array of computed spectromorphological feature values and the original metadata.
    """

    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {"spectro": DataType.ARRAY}

    def config_params():
        return {
            "Parameters": {
                "method": StringParam(
                    "SpectralCentroid",
                    options=[
                        "SpectralCentroid",
                        "SpectralCrestFactor",
                        "SpectralDecrease",
                        "SpectralFlatness",
                        "SpectralFlux",
                        "SpectralKurtosis",
                        "SpectralMfccs",
                        "SpectralPitchChroma",
                        "SpectralRolloff",
                        "SpectralSkewness",
                        "SpectralSlope",
                        "SpectralSpread",
                        "SpectralTonalPowerRatio",
                    ],
                ),
                "window": IntParam(100, 10, 10000),
                "overlap": IntParam(50, 1, 1000),
            }
        }

    def process(self, data: Data):
        """
        Convert a musical scale into a list of HSV colors based on the scale's frequency values
        and their averaged consonance.
        """
        if data is None:
            return None

        if data.data.ndim > 1:
            raise ValueError("Data must be 1D")

        method = self.params["Parameters"]["method"].value
        window = self.params["Parameters"]["window"].value
        overlap = self.params["Parameters"]["overlap"].value
        v, t = computeFeatureCl_new(data.data, method, data.meta["sfreq"], window=window, overlap=overlap)
        return {"spectro": (v, data.meta)}


pyACA_mod = None


def computeFeatureCl_new(afAudioData, cFeatureName, f_s, window=4000, overlap=1):
    """Calculate spectromorphological metrics on time series.

    Parameters
    ----------
    afAudioData : array (numDataPoints, )
        Input signal.
    cFeatureName : str
        {'SpectralCentroid', 'SpectralCrestFactor', 'SpectralDecrease',
         'SpectralFlatness', 'SpectralFlux', 'SpectralKurtosis',
         'SpectralMfccs', 'SpectralPitchChroma', 'SpectralRolloff',
         'SpectralSkewness', 'SpectralSlope', 'SpectralSpread',
         'SpectralTonalPowerRatio', 'TimeAcfCoeff', 'TimeMaxAcf',
         'TimePeakEnvelope', 'TimeRms', 'TimeStd', 'TimeZeroCrossingRate'}
    f_s : int
        Sampling frequency.
    window : int
        Length of the moving window in samples.
    overlap : int
        Overlap between each moving window in samples.

    Returns
    -------
    v : array
        Vector of the spectromorphological metric.
    t : array
        Timestamps.
    """
    global pyACA_mod
    if pyACA_mod is None:
        import pyACA

        pyACA_mod = pyACA

    [v, t] = pyACA_mod.computeFeature(cFeatureName, afAudioData, f_s, None, window, overlap)
    return (v, t)
