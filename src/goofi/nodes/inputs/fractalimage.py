from goofi.params import IntParam, FloatParam
from goofi.data import Data, DataType
from goofi.node import Node
import numpy as np
import time


class FractalImage(Node):
    """
    Generates a 2D fractal noise image based on the Fractal Brownian Motion (fBm) algorithm using Perlin noise. The node procedurally creates fractal patterns which can vary in complexity and visual detail. This can be used for texture synthesis, background generation, or as input to other visual processing nodes.

    Inputs:
    - complexity: If provided, overrides the default noise persistence value to modulate the fractal detail. Expects a scalar or array controlling the fBm persistence.

    Outputs:
    - image: A 2D array representing the generated fractal noise image, with values normalized to the [0, 1] range.
    """

    def config_input_slots():
        return {"complexity": DataType.ARRAY}  # No input data for this node

    def config_output_slots():
        return {"image": DataType.ARRAY}

    def config_params():
        return {
            "image": {
                "image_size": IntParam(512, 128, 2048),
                "persistence": FloatParam(
                    0.5, 0.1, 1.0, doc="The persistence value for the fractal noise (higher values result in more detail)"
                ),
                "octaves": IntParam(6, 1, 6),
                "lacunarity": FloatParam(2.0, 1.0, 4.0),
            },
            "common": {"autotrigger": True},
        }

    def process(self, complexity: Data):
        # Extract the parameters
        image_size = self.params["image"]["image_size"].value
        octaves = self.params["image"]["octaves"].value
        lacunarity = self.params["image"]["lacunarity"].value
        if complexity is None:
            persistence = self.params["image"]["persistence"].value
        else:
            persistence = complexity.data

        # Generate a new seed based on current time
        seed = int(time.time() * 1000) % 2**32  # Convert time to milliseconds and take modulo to avoid overflow

        # Generate the fractal image
        x = np.linspace(0, 5, image_size)
        y = np.linspace(0, 5, image_size)
        x, y = np.meshgrid(x, y)
        image = fbm(x, y, octaves=octaves, lacunarity=lacunarity, persistence=persistence, seed=seed)
        image = (image + 1) / 2  # Rescale values from [-1, 1] to [0, 1]
        # Return the generated fractal image
        return {"image": (image, {})}  # No metadata to include, hence the empty dictionary


def fade(t):
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def lerp(a, b, x):
    return a + x * (b - a)


def gradient(h, x, y):
    # Define the gradient vectors
    vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    # Choose the appropriate gradient vector based on hash value h
    g = vectors[h % 4]
    # Calculate the dot product
    return g[..., 0] * x + g[..., 1] * y


def perlin(x, y, seed=0):
    np.random.seed(seed)
    p = np.arange(256, dtype=int)
    np.random.shuffle(p)
    p = np.stack([p, p]).flatten()
    xi = x.astype(int)
    yi = y.astype(int)
    xf = x - xi
    yf = y - yi

    u = fade(xf)
    v = fade(yf)

    n00 = gradient(p[p[xi] + yi], xf, yf)
    n01 = gradient(p[p[xi] + yi + 1], xf, yf - 1)
    n11 = gradient(p[p[xi + 1] + yi + 1], xf - 1, yf - 1)
    n10 = gradient(p[p[xi + 1] + yi], xf - 1, yf)

    x1 = lerp(n00, n10, u)
    x2 = lerp(n01, n11, u)

    return lerp(x1, x2, v)


def fbm(x, y, octaves=6, lacunarity=2, persistence=0.5, seed=0):
    value = 0
    amplitude = 1.0
    frequency = 1.0
    for _ in range(octaves):
        value += amplitude * perlin(x * frequency, y * frequency, seed=seed)
        amplitude *= persistence
        frequency *= lacunarity
    return value
