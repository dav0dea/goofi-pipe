import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, FloatParam


class Walker(Node):
    """
    Simulates a step-by-step movement of a point (walker) on the Earth's surface based on a direction (angle), speed (velocity), and whether it is moving on water (water). The node calculates the new latitude and longitude from the previous position for each input, accounting for the Earth's curvature and constraints such as pole crossings and longitude normalization.

    Inputs:
    - angle: The direction(s) of movement in degrees, as an array.
    - velocity: The distance(s) to move per step, as an array. Might be scaled up if moving through water.
    - water: An array indicating if the movement is over water (1) or not (0), which affects the speed.

    Outputs:
    - latitude: The updated latitude(s) after applying movement and corrections for the globe's boundaries.
    - longitude: The updated longitude(s) after applying movement and corrections for the globe's boundaries.
    """

    def config_input_slots():
        return {"angle": DataType.ARRAY, "velocity": DataType.ARRAY, "water": DataType.ARRAY}

    def config_output_slots():
        return {"latitude": DataType.ARRAY, "longitude": DataType.ARRAY}

    def config_params():
        return {
            "initial_coordinates": {
                "latitude": FloatParam(0.0, -90, 90),
                "longitude": FloatParam(0.0, -180, 180),
                "reset": BoolParam(False, trigger=True),
                "water_speed_factor": FloatParam(5.0, 0.1, 10.0),
            },
        }

    def setup(self):
        self.lat, self.lon = (
            self.params["initial_coordinates"]["latitude"].value,
            self.params["initial_coordinates"]["longitude"].value,
        )

    def process(self, angle: Data, velocity: Data, water: Data):
        """
        Processes the input angle and velocity to compute new geographical coordinates,
        considering the curvature of the Earth and boundaries such as the poles and the
        International Date Line.
        """
        if angle is None or velocity is None:
            return None

        if self.params["initial_coordinates"]["reset"].value:
            self.lat, self.lon = (
                self.params["initial_coordinates"]["latitude"].value,
                self.params["initial_coordinates"]["longitude"].value,
            )

        if water is None:
            water_situation = 0
        else:
            water_situation = water.data

        water_speed_factor = self.params["initial_coordinates"]["water_speed_factor"].value
        if water_situation == 1:
            velocity.data = velocity.data * water_speed_factor
        # Convert angle to radians for computation
        angle_rad = np.radians(angle.data)

        # Calculate new latitude and longitude
        self.lat = self.lat + velocity.data * np.cos(angle_rad)
        self.lon = self.lon + velocity.data * np.sin(angle_rad)

        # Handle crossing the North/South poles
        if self.lat > 90:
            self.lat = 180 - self.lat
            # self.lon += 180
        elif self.lat < -90:
            self.lat = -180 - self.lat
            # self.lon += 180

        # Normalize longitude to be within -180 to 180
        if self.lon > 180:
            self.lon -= 360
        elif self.lon < -180:
            self.lon += 360

        return {"latitude": (self.lat, {}), "longitude": (self.lon, {})}
