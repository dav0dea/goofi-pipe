from goofi.data import Data, DataType
from goofi.node import Node
import numpy as np


class Coord2loc(Node):
    """
    This node converts geographic coordinates (latitude and longitude) into human-readable location information using reverse geocoding. It determines key location details such as city, state, country, road, and village, and provides both a structured location summary and an indication of whether the coordinates represent a valid land location or are in the ocean.

    Inputs:
    - latitude: The latitude coordinate to be reverse geocoded.
    - longitude: The longitude coordinate to be reverse geocoded.

    Outputs:
    - coord_info: A table containing information about the location corresponding to the input coordinates, such as city, state, country, road, village, and the full address. If the coordinates are invalid or not associated with a land address, it returns a message indicating the location cannot be determined.
    - water_situation: An array indicating whether the coordinates correspond to a recognized land location (0) or a water/non-land location (1).
    """

    def config_input_slots():
        return {"latitude": DataType.ARRAY, "longitude": DataType.ARRAY}

    def config_output_slots():
        return {"coord_info": DataType.TABLE, "water_situation": DataType.ARRAY}

    def setup(self):
        from geopy.geocoders import Nominatim

        # list keys : ['ISO3166-2-lvl4', 'ISO3166-2-lvl6', 'ISO3166-2-lvl7', 'city',
        # 'city_district', 'country', 'country_code', 'county', 'district', 'hamlet', 'locality',
        # 'municipality', 'postcode', 'region', 'road', 'state', 'state_district', 'suburb', 'town', 'village']
        # Use a user-defined user agent to comply with Nominatim's usage policy
        self.road = ["road"]
        self.village = ["village", "hamlet"]

        self.city = ["city", "town", "locality", "municipality", "city_district"]
        self.state = ["state", "region", "state_district"]
        self.country = ["country"]
        self.geolocator = Nominatim(user_agent="my_application")

    def process(self, latitude: Data, longitude: Data):
        if latitude is None or latitude.data is None or longitude is None or longitude.data is None:
            return None
        # check if the input is within the range of the latitude and longitude
        if not (-90 <= latitude.data <= 90 and -180 <= longitude.data <= 180):
            return {"coord_info": ({"info : ": Data(DataType.STRING, "Invalid coordinates", {})}, {})}

        latitude = latitude.data
        longitude = longitude.data

        # Convert the input coordinates to a string format "latitude, longitude"
        coordinates = (latitude, longitude)
        # Use geolocator to find the location from coordinates
        location = self.geolocator.reverse(coordinates)

        # Check if location is found
        if location and location.address:

            location_info = {}
            for key in self.city:
                value = location.raw.get("address", {}).get(key)
                if value:
                    location_info["city"] = Data(DataType.STRING, value, {})
                    break
            value = None
            for key in self.state:
                value = location.raw.get("address", {}).get(key)
                if value:
                    location_info["state"] = Data(DataType.STRING, value, {})
                    break
            value = None
            for key in self.country:
                value = location.raw.get("address", {}).get(key)
                if value:
                    location_info["country"] = Data(DataType.STRING, value, {})
                    break
            value = None
            for key in self.road:
                value = location.raw.get("address", {}).get(key)
                if value:
                    location_info["road"] = Data(DataType.STRING, value, {})
                    break
            value = None
            for key in self.village:
                value = location.raw.get("address", {}).get(key)
                if value:
                    location_info["village"] = Data(DataType.STRING, value, {})
                    break
            location_info["full_address"] = Data(DataType.STRING, location.address, {})
            value = None

            return {"coord_info": (location_info, {}), "water_situation": (np.array(0), {})}
        else:
            return {
                "coord_info": ({"info": Data(DataType.STRING, "You're lost in the ocean", {})}, {}),
                "water_situation": (np.array(1), {}),
            }
