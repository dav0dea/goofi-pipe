from os import path

import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, FloatParam, IntParam


class FaceLandmarker(Node):
    """
    Detects face landmarks and facial expressions from an input image using MediaPipe Face Landmarker.
    The node processes images to detect 478 face mesh landmarks and optionally 52 blendshape scores
    for facial expressions.

    Inputs:
    - image: A 2D or 3D array representing the input image for face landmark detection.

    Outputs:
    - landmarks: The detected 3D face landmarks as an array (3, 478) with x, y, z coordinates.
    - blendshapes: Optional array of 52 blendshape scores representing facial expressions.
    """

    def config_input_slots():
        return {"image": DataType.ARRAY}

    def config_output_slots():
        return {
            "landmarks": DataType.ARRAY,
            "blendshapes": DataType.ARRAY,
        }

    def config_params():
        return {
            "detection": {
                "num_faces": IntParam(1, 1, 10, doc="Maximum number of faces to detect"),
                "min_detection_confidence": FloatParam(0.5, 0.0, 1.0, doc="Minimum face detection confidence"),
                "min_presence_confidence": FloatParam(0.5, 0.0, 1.0, doc="Minimum face presence confidence"),
                "min_tracking_confidence": FloatParam(0.5, 0.0, 1.0, doc="Minimum tracking confidence"),
                "output_blendshapes": BoolParam(True, doc="Output blendshape scores for facial expressions"),
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
        self._setup_detector()

    def _setup_detector(self):
        """Create the face landmarker detector."""
        model_filename = "face_landmarker.task"
        model_path = path.join(self.assets_path, model_filename)

        if not path.exists(model_path):
            url = f"https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/{model_filename}"
            response = self.requests.get(url)
            response.raise_for_status()
            with open(model_path, "wb") as file:
                file.write(response.content)

        base_options = self.python.BaseOptions(model_asset_path=model_path)
        options = self.vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=self.params["detection"]["num_faces"].value,
            min_face_detection_confidence=self.params["detection"]["min_detection_confidence"].value,
            min_face_presence_confidence=self.params["detection"]["min_presence_confidence"].value,
            min_tracking_confidence=self.params["detection"]["min_tracking_confidence"].value,
            output_face_blendshapes=self.params["detection"]["output_blendshapes"].value,
        )
        self.detector = self.vision.FaceLandmarker.create_from_options(options)

    def process(self, image: Data):
        if image is None or image.data is None:
            return None

        # Convert image to MediaPipe format
        img_data = image.data
        if img_data.dtype != np.uint8:
            img_data = (img_data * 255).astype(np.uint8)
        
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=img_data)
        detection_result = self.detector.detect(mp_image)

        if len(detection_result.face_landmarks) == 0:
            return None

        # Extract face landmarks (478 points)
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in detection_result.face_landmarks[0]]).T
        landmark_meta = {
            "channels": {"dim0": ["x", "y", "z"]},
        }

        # Extract blendshapes if available
        blendshapes = None
        if detection_result.face_blendshapes and len(detection_result.face_blendshapes) > 0:
            blendshape_data = np.array([bs.score for bs in detection_result.face_blendshapes[0]])
            blendshape_names = [bs.category_name for bs in detection_result.face_blendshapes[0]]
            blendshapes = (blendshape_data, {"channels": {"dim0": blendshape_names}})

        return {
            "landmarks": (landmarks, landmark_meta),
            "blendshapes": blendshapes,
        }


# MediaPipe Face Landmarker outputs 478 landmarks
# Key regions include:
# - Lips: landmarks around the mouth
# - Left/Right eye: landmarks around each eye
# - Left/Right eyebrow: landmarks on each eyebrow
# - Face oval: landmarks around the face contour
# - Nose: landmarks on the nose
# See: https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png

# Blendshape names (52 total) include expressions like:
# browDownLeft, browDownRight, browInnerUp, browOuterUpLeft, browOuterUpRight,
# cheekPuff, cheekSquintLeft, cheekSquintRight, eyeBlinkLeft, eyeBlinkRight,
# eyeLookDownLeft, eyeLookDownRight, eyeLookInLeft, eyeLookInRight,
# eyeLookOutLeft, eyeLookOutRight, eyeLookUpLeft, eyeLookUpRight,
# eyeSquintLeft, eyeSquintRight, eyeWideLeft, eyeWideRight,
# jawForward, jawLeft, jawOpen, jawRight, mouthClose, mouthDimpleLeft,
# mouthDimpleRight, mouthFrownLeft, mouthFrownRight, mouthFunnel,
# mouthLeft, mouthLowerDownLeft, mouthLowerDownRight, mouthPressLeft,
# mouthPressRight, mouthPucker, mouthRight, mouthRollLower, mouthRollUpper,
# mouthShrugLower, mouthShrugUpper, mouthSmileLeft, mouthSmileRight,
# mouthStretchLeft, mouthStretchRight, mouthUpperUpLeft, mouthUpperUpRight,
# noseSneerLeft, noseSneerRight, _neutral
