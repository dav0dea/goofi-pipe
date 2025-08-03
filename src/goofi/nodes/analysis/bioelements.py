from os.path import join

import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, IntParam


class Bioelements(Node):
    """
    This node identifies chemical elements present in an input 1D frequency array by matching its spectral features against a database of known air element spectral lines. It extracts the most frequently matched elements and provides their names, spectral regions, and types as output.

    Inputs:
    - data: A 1D array representing signal frequencies to be analyzed for spectral characteristics.

    Outputs:
    - elements: A table containing the names of detected elements, their corresponding spectral regions, and their types, based on the matches found in the spectral database.
    """

    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {"elements": DataType.TABLE}

    def config_params():
        return {
            "bioelements": {
                "tolerance": FloatParam(0.5, 0.01, 5),
                "n_elements": IntParam(2, 1, 4),
            }
        }

    def setup(self):
        import pandas as pd

        # load the dataframe here to avoid loading it on startup
        self.air_elements = pd.read_csv(join(self.assets_path, "air_elements_filtered.csv"))

    def process(self, data: Data):
        if data is None:
            return None

        data.data = np.squeeze(data.data)
        if data.data.ndim > 1:
            raise ValueError("Data must be 1D")

        tolerance = self.params["bioelements"]["tolerance"].value
        elems, spec_regions, types = bioelements_realtime(data.data, self.air_elements, tolerance)

        # select the n most common elements
        elems = elems[: self.params["bioelements"]["n_elements"].value]
        spec_regions = spec_regions[: self.params["bioelements"]["n_elements"].value]
        types = types[: self.params["bioelements"]["n_elements"].value]

        # remove duplicates
        elems = list(dict.fromkeys(elems))
        spec_regions = list(dict.fromkeys(spec_regions))
        types = list(dict.fromkeys(types))
        # create single string for each variable
        elems = ", ".join(elems)
        spec_regions = ", ".join(spec_regions)
        types = ", ".join(types)
        # combine the three lists into a dictionary
        elements = {
            "element": Data(DataType.STRING, elems, {}),
            "spectral_region": Data(DataType.STRING, spec_regions, {}),
            "type": Data(DataType.STRING, types, {}),
        }
        # print(elements)
        return {"elements": (elements, data.meta)}


hertz_to_nm_fn, find_matching_spectral_lines_fn = None, None


def bioelements_realtime(data, df, tolerance):
    global hertz_to_nm_fn, find_matching_spectral_lines_fn
    if hertz_to_nm_fn is None or find_matching_spectral_lines_fn is None:
        from biotuner.bioelements import find_matching_spectral_lines, hertz_to_nm

        hertz_to_nm_fn = hertz_to_nm
        find_matching_spectral_lines_fn = find_matching_spectral_lines

    peaks_ang = [hertz_to_nm_fn(x) * 10 for x in data]
    res = find_matching_spectral_lines_fn(df, peaks_ang, tolerance=tolerance)
    elements_count = res["element"].value_counts()
    elements_final = elements_count.index.tolist()[:3]  # take the three most common elements

    elements_mapped = []
    spectrum_regions_mapped = []
    types_mapped = []

    for element in elements_final:
        res_filtered = res[res["element"] == element]

        # Here we are making sure to only get one spectrum region and type per element
        spectrum_region = res_filtered["spectrum_region"].iloc[0]
        type_ = res_filtered["type"].iloc[0]

        elements_mapped.append(element)
        spectrum_regions_mapped.append(spectrum_region)
        types_mapped.append(type_)

    return elements_mapped, spectrum_regions_mapped, types_mapped
