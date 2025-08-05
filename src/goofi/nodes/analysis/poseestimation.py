from os import path

import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node


class PoseEstimation(Node):
    """
    Estimates the 3D positions of hand landmarks from an input image using a machine learning model. The node processes images to detect a single hand and outputs the detected hand pose as a set of 3D landmark coordinates along with related metadata.

    Inputs:
    - image: A 2D or 3D array representing the input image for hand pose estimation.

    Outputs:
    - pose: The detected 3D hand landmarks as an array transposed to (3, 21), with accompanying metadata including the hand handedness and landmark channel names.
    """

    def config_input_slots():
        return {"image": DataType.ARRAY}

    def config_output_slots():
        return {"pose": DataType.ARRAY}

    def setup(self):
        import mediapipe as mp
        import requests
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self.mp = mp

        model_path = path.join(self.assets_path, "hand_landmarker.task")
        if not path.exists(model_path):
            url = (
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )
            response = requests.get(url)
            # ensure the request was successful
            response.raise_for_status()
            # write the content to a file
            with open(model_path, "wb") as file:
                file.write(response.content)

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
        self.detector = vision.HandLandmarker.create_from_options(options)

    def process(self, image: Data):
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=(image.data * 255).astype(np.uint8))
        detection_result = self.detector.detect(mp_image)

        if len(detection_result.hand_landmarks) == 0:
            return None

        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in detection_result.hand_landmarks[0]]).T
        meta = {
            "handdedness": detection_result.handedness[0][0].display_name,
            "channels": {"dim0": ["x", "y", "z"], "dim1": LANDMARK_NAMES},
        }
        return {"pose": (landmarks, meta)}


LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",
    "INDEX_FINGER_MCP",
    "INDEX_FINGER_PIP",
    "INDEX_FINGER_DIP",
    "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP",
    "MIDDLE_FINGER_PIP",
    "MIDDLE_FINGER_DIP",
    "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP",
    "RING_FINGER_PIP",
    "RING_FINGER_DIP",
    "RING_FINGER_TIP",
    "PINKY_MCP",
    "PINKY_PIP",
    "PINKY_DIP",
    "PINKY_TIP",
]
