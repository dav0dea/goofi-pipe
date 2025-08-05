from os.path import join

import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, IntParam


class Bioplanets(Node):
    """
    This node analyzes an array of peak frequencies and identifies which of the main solar system planets (Venus, Earth, Mars, Jupiter, Saturn) have spectral lines near those peaks, based on a reference planetary spectrum dataset. For each planet, it outputs the list of matched wavelengths, and also provides a summary indicating which planets have the most matched peaks.

    Inputs:
    - peaks: A 1D array of detected frequency peaks to be analyzed.

    Outputs:
    - planets: A table containing, for each planet, the array of matched wavelengths corresponding to the input peaks.
    - top_planets: A string listing the planets with the highest number of matched spectral peaks, ranked in order.
    """

    def config_input_slots():
        return {"peaks": DataType.ARRAY}

    def config_output_slots():
        return {"planets": DataType.TABLE, "top_planets": DataType.STRING}

    def config_params():
        return {
            "bioplanets": {
                "tolerance": FloatParam(0.5, 0.01, 5),
                "n_top_planets": IntParam(3, 1, 6),
            }
        }

    def setup(self):
        import pandas as pd

        # load the dataframe here to avoid loading it on startup
        self.planets_data = pd.read_csv(join(self.assets_path, "planets_peaks_prominence02.csv"))
        self.desired_planets = ["venus", "earth", "mars", "jupiter", "saturn"]

    def process(self, peaks: Data):
        if peaks is None:
            return None

        peaks.data = np.squeeze(peaks.data)
        if peaks.data.ndim > 1:
            raise ValueError("Data must be 1D")

        tolerance = self.params["bioplanets"]["tolerance"].value
        results = bioplanets_realtime(peaks.data, self.planets_data, tolerance)
        # Determine top planets based on the number of peaks
        # Filter out planets that have no peaks
        planet_peaks_count = {planet: len(peaks) for planet, peaks in results.items() if len(peaks) > 0}
        sorted_planets = sorted(planet_peaks_count, key=planet_peaks_count.get, reverse=True)
        top_planets_str = " ".join(sorted_planets[: self.params["bioplanets"]["n_top_planets"].value])

        planets = {}
        for i in self.desired_planets:
            planets[i] = Data(DataType.ARRAY, np.array(results[i]), peaks.meta)

        return {"planets": (planets, peaks.meta), "top_planets": (top_planets_str, peaks.meta)}


hertz_to_nm_fn, find_matching_spectral_lines_fn = None, None


def bioplanets_realtime(peaks, df, tolerance):
    desired_planets = ["venus", "earth", "mars", "jupiter", "saturn"]
    global hertz_to_nm_fn, find_matching_spectral_lines_fn
    if hertz_to_nm_fn is None or find_matching_spectral_lines_fn is None:
        from biotuner.bioelements import find_matching_spectral_lines, hertz_to_nm

        hertz_to_nm_fn = hertz_to_nm
        find_matching_spectral_lines_fn = find_matching_spectral_lines
    # Define the list of planets you are interested in

    peaks_ang = [hertz_to_nm_fn(x) * 10 for x in peaks]  # convert to Angstrom

    # find_matching_spectral_lines is a function you must have defined elsewhere
    res = find_matching_spectral_lines_fn(df, peaks_ang, tolerance=tolerance)

    # Filter the res DataFrame for the desired planets
    res_filtered = res[res["planet"].isin(desired_planets)]

    # Extract and output the wavelengths for these planets
    wavelengths_by_planet = {}
    for planet in desired_planets:
        # Filter wavelengths for a given planet
        wavelengths = res_filtered[res_filtered["planet"] == planet]["wavelength"].tolist()

        # Round each wavelength to two decimal places
        wavelengths = [round(w, 2) for w in wavelengths]

        # Remove duplicates by converting to a set, then back to a list
        unique_wavelengths = list(set(wavelengths))

        # Sort for readability (optional)
        unique_wavelengths.sort()

        wavelengths_by_planet[planet] = unique_wavelengths

    return wavelengths_by_planet
