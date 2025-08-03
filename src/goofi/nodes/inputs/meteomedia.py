from os.path import join

import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, StringParam


class MeteoMedia(Node):
    """
    This node fetches current weather data for a given geographic location using the Tomorrow.io Realtime Weather API. It accepts either latitude and longitude coordinates or a location name as input, and outputs a table containing various weather parameters for the specified location.

    Inputs:
    - latitude: A single-value array representing the latitude coordinate of the location.
    - longitude: A single-value array representing the longitude coordinate of the location.
    - location_name: A string representing the human-readable name of the location (optional; used instead of coordinates if provided).

    Outputs:
    - weather_data_table: A table (dictionary of arrays) containing weather parameters such as temperature, humidity, wind speed, and other data returned by the API. If the API call fails, an "ERROR" field with the HTTP status code is provided.
    """

    def config_input_slots():
        return {"latitude": DataType.ARRAY, "longitude": DataType.ARRAY, "location_name": DataType.STRING}

    def config_output_slots():
        return {"weather_data_table": DataType.TABLE}

    def config_params():
        return {
            "TomorrowAPI": {
                "key": StringParam("YOUR_API_KEY", doc="you can also add a file in the assets folder : tomorrowkey.txt")
            },
            "common": {
                "autotrigger": True,
                "max_frequency": FloatParam(0.1, 0.1, 30.0),
            },
        }

    def setup(self):
        import requests

        self.requests = requests

    def process(self, latitude: Data, longitude: Data, location_name: Data):
        if latitude is None or longitude is None:
            return None
        # Extract values from Data objects
        lat_value = latitude.data
        long_value = longitude.data
        # ensure that lat and long are not empty and single values
        if len(lat_value) != 1 or len(long_value) != 1:
            return None
        # Get the API key from the parameters
        api_key = self.params["TomorrowAPI"]["key"].value
        api_key_path = join(self.assets_path, "tomorrowkey.txt")
        # check if the file exists
        try:
            with open(api_key_path, "r") as f:
                api_key = f.read()
                print("API key found in file", api_key)
        except FileNotFoundError:
            pass
        if location_name is None:
            url = (
                f"https://api.tomorrow.io/v4/weather/realtime?location={float(lat_value)},{float(long_value)}&apikey={api_key}"
            )
            headers = {"accept": "application/json"}
            response = self.requests.get(url, headers=headers)
            print(response.status_code)

        else:
            url = f"https://api.tomorrow.io/v4/weather/realtime?location={location_name}&apikey={api_key}"
            headers = {"accept": "application/json"}
            response = self.requests.get(url, headers=headers)
            print(response.status_code)
        if response.status_code == 200:
            responses = response.json()
            output_table = responses["data"]["values"]
            output_table
            print("HELLO")
            # convert all elements of the dictionary to a Data object
            for key, value in output_table.items():
                output_table[key] = Data(DataType.ARRAY, np.array(value), {})
            print(output_table)
        else:
            # Handle error or empty response
            output_table = {"ERROR": Data(DataType.ARRAY, np.array(response.status_code), {})}

        return {"weather_data_table": (output_table, {})}


# Note: Replace YOUR_API_KEY with your actual API key from Tomorrow.io.
