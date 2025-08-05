from typing import Any, Dict, Tuple

import cv2
import numpy as np
from mss import mss

from goofi.data import DataType
from goofi.node import Node
from goofi.params import BoolParam, IntParam, StringParam


class VideoStream(Node):
    """
    This node captures video frames from either a connected camera device or the desktop screen. It outputs the captured frame as a normalized RGB array, with optional mirroring and cropping applied.

    Outputs:
    - frame: The captured image frame as a NumPy array of shape (height, width, 3) in RGB format, with pixel values in the range [0, 1].
    """

    def config_params():
        return {
            "video_stream": {
                "capture_mode": StringParam("camera", ["camera", "screen"]),
                "device_index": IntParam(0, 0, 10, doc="Index of the camera or screen to capture"),
                "mirror": BoolParam(False),
                "crop_left": IntParam(0, 0, 2000, doc="Number of pixels to crop from the left"),
                "crop_top": IntParam(0, 0, 2000, doc="Number of pixels to crop from the top"),
                "crop_right": IntParam(0, 0, 2000, doc="Number of pixels to crop from the right"),
                "crop_bottom": IntParam(0, 0, 2000, doc="Number of pixels to crop from the bottom"),
            },
            "common": {"autotrigger": BoolParam(True)},
        }

    def config_output_slots():
        return {"frame": DataType.ARRAY}

    def setup(self):
        try:
            # close opencv video capture
            self.cap.release()
        except:
            pass

        self.cap = None
        if self.params.video_stream.capture_mode.value == "camera":
            self.cap = cv2.VideoCapture(self.params.video_stream.device_index.value)

    def process(self) -> Dict[str, Tuple[Any, Dict[str, Any]]]:
        frame = None

        capture_mode = self.params.video_stream.capture_mode.value
        cropping_done = False
        if capture_mode == "camera":
            if self.cap is None:
                return None

            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                print("Failed to capture frame from camera.")
                return None

        elif capture_mode == "screen":
            screen_index = self.params.video_stream.device_index.value

            with mss(display=screen_index) as sct:
                if screen_index >= len(sct.monitors):
                    print(f"Screen index {screen_index} is out of range.")

                # update monitor size with cropping values
                monitor_size = sct.monitors[screen_index].copy()
                monitor_size["top"] += self.params.video_stream.crop_top.value
                monitor_size["left"] += self.params.video_stream.crop_left.value
                monitor_size["width"] -= self.params.video_stream.crop_left.value + self.params.video_stream.crop_right.value
                monitor_size["height"] -= self.params.video_stream.crop_top.value + self.params.video_stream.crop_bottom.value

                # cropping is done by mss to avoid performance overhead
                cropping_done = True
                screen_capture = sct.grab(monitor_size)

                frame = np.array(screen_capture)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

        if frame is None:
            return None

        if not cropping_done:
            # crop the frame if still needed
            frame = frame[
                self.params.video_stream.crop_top.value : -self.params.video_stream.crop_bottom.value - 1,
                self.params.video_stream.crop_left.value : -self.params.video_stream.crop_right.value - 1,
            ]

        if self.params.video_stream.mirror.value:
            # flip the frame horizontally
            frame = cv2.flip(frame, 1)

        frame = frame.astype("float32") / 255.0

        return {"frame": (frame, {})}

    def video_stream_device_index_changed(self, value):
        # opencv video capture needs to be reinitialized, mss can be reused
        if self.params.video_stream.capture_mode.value == "camera":
            self.setup()

    def video_stream_capture_mode_changed(self, value):
        self.setup()
