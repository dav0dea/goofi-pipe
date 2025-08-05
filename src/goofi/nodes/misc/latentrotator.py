import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, FloatParam


class LatentRotator(Node):
    """
    This node incrementally rotates a latent vector in its high-dimensional space based on an array of input angles. For each process step, a delta vector is calculated using the cosine of each angle and is added to the latent vector, enabling gradual, controlled movement within the latent space. The internal state accumulates these changes over time.

    Inputs:
    - latent_vector: The initial or current latent vector to be rotated, provided as a 1D array.
    - angles: An array of angles (one per latent vector dimension) that determines the direction and magnitude of movement in the latent space.

    Outputs:
    - rotated_vector: The resulting latent vector after applying the incremental rotation and accumulation, as a 1D array.
    """

    def config_input_slots():
        return {
            "latent_vector": DataType.ARRAY,
            "angles": DataType.ARRAY,
        }

    def config_output_slots():
        return {
            "rotated_vector": DataType.ARRAY,
        }

    def config_params():
        return {
            "Control": {
                "step_size": FloatParam(0.1, 0.01, 10.0, doc="Step size for moving in the latent space"),
                "normalize": BoolParam(False, doc="Normalize the rotated vector to unit length"),
                "reset": BoolParam(False, trigger=True, doc="Reset the latent vector to its initial state"),
            }
        }

    def setup(self):
        """
        Initialize the cumulative vector state.
        """
        self.cumulative_vector = None

    def process(self, latent_vector: Data, angles: Data):
        """
        Rotates the input latent vector based on the provided angles and accumulates the changes.

        Args:
            latent_vector (Data): The original latent vector (shape (N,)).
            angles (Data): Array of angles (N,).

        Returns:
            dict: The rotated latent vector.
        """
        if latent_vector is None or angles is None:
            return None

        if self.params.Control.reset.value:
            # Reset the cumulative vector to the initial latent vector
            self.cumulative_vector = latent_vector.data.copy()
            return {"rotated_vector": (self.cumulative_vector, {})}

        latent_vector_array = latent_vector.data
        angles_array = angles.data

        if latent_vector_array.ndim != 1:
            raise ValueError("Latent vector must be 1D.")

        n = latent_vector_array.shape[0]

        if angles_array.shape[0] != n:
            raise ValueError(f"Number of angles must be {n} for a latent vector of dimension {n}.")

        step_size = self.params.Control.step_size.value
        normalize = self.params.Control.normalize.value

        # Compute the unit vector directly influenced by each angle
        unit_vector = np.zeros(n)
        for i in range(n):
            unit_vector[i] = np.cos(angles_array[i]) * step_size

        # Initialize cumulative vector if not already set
        if self.cumulative_vector is None:
            self.cumulative_vector = latent_vector_array.copy()

        # Apply the rotation by adding the delta to the cumulative vector
        self.cumulative_vector += unit_vector

        # Normalize the rotated vector if specified
        if normalize:
            self.cumulative_vector /= np.linalg.norm(self.cumulative_vector)

        return {"rotated_vector": (self.cumulative_vector, {})}
