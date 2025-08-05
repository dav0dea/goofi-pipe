import numpy as np

from goofi.data import DataType
from goofi.node import Node
from goofi.params import BoolParam, StringParam


class RandomArray(Node):
    """
    This node generates a random array of specified dimensions using either a uniform or normal distribution. Optionally, if the array is square, it can normalize the largest eigenvalue to 1. The generated array can be reset or regenerated as needed.

    Outputs:
    - random_array: The generated random array based on the selected distribution and dimensions.
    """

    def config_params():
        return {
            "random": {
                "dimensions": "2,2",  # Default dimensions
                "distribution": StringParam("normal", options=["uniform", "normal"]),
                "normalize_eigenvalue": False,  # Default: do not normalize eigenvalue
                "reset": BoolParam(trigger=True),
            },
            "common": {
                "autotrigger": True,
            },
        }

    def config_output_slots():
        return {"random_array": DataType.ARRAY}

    def setup(self):
        self.random_array = None

        distribution = self.params.random.distribution.value
        normalize_eigenvalue = self.params.random.normalize_eigenvalue.value

        try:
            # Parse the dimensions string into a tuple of integers
            dimensions = tuple(map(int, self.params.random.dimensions.value.split(",")))

            # Generate a random array based on the specified distribution
            if distribution == "uniform":
                random_array = np.random.uniform(size=dimensions)
            elif distribution == "normal":
                random_array = np.random.normal(0, 1, size=dimensions)
            else:
                raise ValueError(f"Unsupported distribution: {distribution}. Use 'uniform' or 'normal'.")

            # Normalize the largest eigenvalue to 1 if specified
            if normalize_eigenvalue and len(dimensions) == 2 and dimensions[0] == dimensions[1]:
                eigenvalues, _ = np.linalg.eig(random_array)
                max_eigenvalue = np.max(np.abs(eigenvalues))
                if max_eigenvalue != 0:
                    random_array /= max_eigenvalue

            self.random_array = random_array

        except (ValueError, TypeError) as e:
            # Handle errors in parsing dimensions or generating the array
            raise ValueError("Invalid dimensions format or distribution type. Please provide a valid input.") from e

    def process(self):
        return {"random_array": (self.random_array, {})}

    def random_dimensions_changed(self, val):
        self.setup()

    def random_distribution_changed(self, val):
        self.setup()

    def random_normalize_eigenvalue_changed(self, val):
        self.setup()

    def random_reset_changed(self, val):
        self.setup()
