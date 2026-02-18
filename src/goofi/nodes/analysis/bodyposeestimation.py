from os import path

import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, FloatParam, IntParam, StringParam


class BodyPoseEstimation(Node):
    """
    Estimates the 3D positions of body landmarks from an input image using MediaPipe Pose Landmarker.
    The node processes images to detect human body poses and outputs 33 landmark coordinates.

    Inputs:
    - image: A 2D or 3D array representing the input image for body pose estimation.

    Outputs:
    - pose: The detected 3D body landmarks as an array (3, 33) with x, y, z coordinates.
    - world_pose: The 3D world coordinates of body landmarks in meters.
    - segmentation: Optional segmentation mask for the detected pose.
    """

    def config_input_slots():
        return {"image": DataType.ARRAY}

    def config_output_slots():
        return {
            "pose": DataType.ARRAY,
            "world_pose": DataType.ARRAY,
            "segmentation": DataType.ARRAY,
        }

    def config_params():
        return {
            "detection": {
                "model": StringParam("lite", options=["lite", "full", "heavy"], doc="Model complexity variant"),
                "num_poses": IntParam(1, 1, 10, doc="Maximum number of poses to detect"),
                "min_detection_confidence": FloatParam(0.5, 0.0, 1.0, doc="Minimum pose detection confidence"),
                "min_presence_confidence": FloatParam(0.5, 0.0, 1.0, doc="Minimum pose presence confidence"),
                "min_tracking_confidence": FloatParam(0.5, 0.0, 1.0, doc="Minimum tracking confidence"),
                "output_segmentation": BoolParam(False, doc="Output segmentation mask for detected pose"),
            }
        }

    def setup(self):
        import mediapipe as mp
        import requests
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self.mp = mp
        self.python = python
        self.vision = vision
        self.requests = requests
        self.detector = None
        self._current_model = None

    def _get_or_create_detector(self):
        """Create or recreate detector if model setting changed."""
        model_variant = self.params["detection"]["model"].value
        
        if self.detector is not None and self._current_model == model_variant:
            return self.detector

        model_name = f"pose_landmarker_{model_variant}"
        model_filename = f"{model_name}.task"
        model_path = path.join(self.assets_path, model_filename)

        if not path.exists(model_path):
            url = f"https://storage.googleapis.com/mediapipe-models/pose_landmarker/{model_name}/float16/1/{model_filename}"
            response = self.requests.get(url)
            response.raise_for_status()
            with open(model_path, "wb") as file:
                file.write(response.content)

        base_options = self.python.BaseOptions(model_asset_path=model_path)
        options = self.vision.PoseLandmarkerOptions(
            base_options=base_options,
            num_poses=self.params["detection"]["num_poses"].value,
            min_pose_detection_confidence=self.params["detection"]["min_detection_confidence"].value,
            min_pose_presence_confidence=self.params["detection"]["min_presence_confidence"].value,
            min_tracking_confidence=self.params["detection"]["min_tracking_confidence"].value,
            output_segmentation_masks=self.params["detection"]["output_segmentation"].value,
        )
        self.detector = self.vision.PoseLandmarker.create_from_options(options)
        self._current_model = model_variant
        return self.detector

    def process(self, image: Data):
        if image is None or image.data is None:
            return None

        detector = self._get_or_create_detector()

        # Convert image to MediaPipe format
        img_data = image.data
        if img_data.dtype != np.uint8:
            img_data = (img_data * 255).astype(np.uint8)
        
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=img_data)
        detection_result = detector.detect(mp_image)

        if len(detection_result.pose_landmarks) == 0:
            return None

        # Extract normalized landmarks (image coordinates)
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in detection_result.pose_landmarks[0]]).T
        pose_meta = {
            "channels": {"dim0": ["x", "y", "z"], "dim1": POSE_LANDMARK_NAMES},
        }

        # Extract world landmarks (3D coordinates in meters)
        world_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in detection_result.pose_world_landmarks[0]]).T
        world_meta = {
            "channels": {"dim0": ["x", "y", "z"], "dim1": POSE_LANDMARK_NAMES},
            "units": "meters",
        }

        # Extract segmentation mask if available
        segmentation = None
        if detection_result.segmentation_masks and len(detection_result.segmentation_masks) > 0:
            segmentation = (detection_result.segmentation_masks[0].numpy_view(), {})

        return {
            "pose": (landmarks, pose_meta),
            "world_pose": (world_landmarks, world_meta),
            "segmentation": segmentation,
        }


POSE_LANDMARK_NAMES = [
    "NOSE",
    "LEFT_EYE_INNER",
    "LEFT_EYE",
    "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER",
    "RIGHT_EYE",
    "RIGHT_EYE_OUTER",
    "LEFT_EAR",
    "RIGHT_EAR",
    "MOUTH_LEFT",
    "MOUTH_RIGHT",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_PINKY",
    "RIGHT_PINKY",
    "LEFT_INDEX",
    "RIGHT_INDEX",
    "LEFT_THUMB",
    "RIGHT_THUMB",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
    "LEFT_HEEL",
    "RIGHT_HEEL",
    "LEFT_FOOT_INDEX",
    "RIGHT_FOOT_INDEX",
]
