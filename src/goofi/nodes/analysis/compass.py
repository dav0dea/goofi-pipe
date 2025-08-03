import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node


class Compass(Node):
    """
    Computes the angular differences between two points in N-dimensional space. The node receives two N-dimensional vectors and calculates the sequence of angles between corresponding vector elements, expressing the direction from the first input vector (pole1) to the second (pole2) projected onto each pair of adjacent dimensions. The angles are given in degrees and normalized to the range [0, 360).

    Inputs:
    - pole1: First N-dimensional array representing the initial point.
    - pole2: Second N-dimensional array representing the target point.

    Outputs:
    - angles: Array of N-1 angles (in degrees) representing the orientation difference between pole1 and pole2 in each pair of adjacent dimensions.
    """

    def config_input_slots():
        return {"pole1": DataType.ARRAY, "pole2": DataType.ARRAY}

    def config_output_slots():
        return {"angles": DataType.ARRAY}

    def process(self, pole1: Data, pole2: Data):
        """
        Processes the input polar arrays to compute angles in N-dimensional space.

        Args:
            pole1 (Data): First input array (1D, length N).
            pole2 (Data): Second input array (1D, length N).

        Returns:
            dict: Angles in N-1 dimensions.
        """
        if pole1 is None or pole2 is None:
            return None

        # Extract the float values from Data objects
        pole1_val = pole1.data
        pole2_val = pole2.data

        # Ensure the inputs are valid 1D arrays of the same length
        if pole1_val.ndim != 1 or pole2_val.ndim != 1 or len(pole1_val) != len(pole2_val):
            raise ValueError("Inputs pole1 and pole2 must be 1D arrays of the same length.")

        # Calculate the vector difference
        vector_diff = pole2_val - pole1_val

        # Initialize angles array
        angles = []

        # Compute angles iteratively
        for i in range(len(vector_diff) - 1):
            y = vector_diff[i + 1]
            x = vector_diff[i]
            angle = np.arctan2(y, x)  # Use arctan2 for full circle coverage
            angles.append(angle)

        # Convert angles to degrees and normalize to [0, 360)
        angles_degrees = np.degrees(angles)
        angles_degrees = np.mod(angles_degrees, 360)  # Ensure values are in [0, 360)

        # Return the angles as a Data object
        return {"angles": (np.array(angles_degrees), {})}
