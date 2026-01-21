"""
Camera module for capturing frames from the Raspberry Pi camera.
"""

import time
from typing import Optional
import numpy as np

# Try to import picamera2 (only available on Raspberry Pi)
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False

# Fallback to OpenCV for testing on non-Pi systems
import cv2


class Camera:
    """
    Camera interface that works with Raspberry Pi Camera or USB webcam.
    """

    def __init__(
        self,
        resolution: tuple[int, int] = (1280, 720),
        framerate: int = 30,
        rotation: int = 0,
        use_picamera: bool = True,
    ):
        """
        Initialize the camera.

        Args:
            resolution: (width, height) tuple
            framerate: Target framerate
            rotation: Rotation in degrees (0, 90, 180, 270)
            use_picamera: Whether to use PiCamera (False for USB webcam)
        """
        self.resolution = resolution
        self.framerate = framerate
        self.rotation = rotation
        self.use_picamera = use_picamera and PICAMERA_AVAILABLE

        self._camera = None
        self._is_running = False

    def start(self) -> None:
        """Start the camera capture."""
        if self._is_running:
            return

        if self.use_picamera:
            self._start_picamera()
        else:
            self._start_opencv()

        self._is_running = True

    def _start_picamera(self) -> None:
        """Initialize Raspberry Pi camera."""
        self._camera = Picamera2()

        config = self._camera.create_preview_configuration(
            main={"size": self.resolution, "format": "RGB888"},
            controls={"FrameRate": self.framerate},
        )
        self._camera.configure(config)
        self._camera.start()

        # Allow camera to warm up
        time.sleep(2)

    def _start_opencv(self) -> None:
        """Initialize OpenCV camera (USB webcam)."""
        self._camera = cv2.VideoCapture(0)
        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self._camera.set(cv2.CAP_PROP_FPS, self.framerate)

        if not self._camera.isOpened():
            raise RuntimeError("Failed to open camera")

        # Allow camera to warm up
        time.sleep(1)

    def capture(self) -> Optional[np.ndarray]:
        """
        Capture a single frame.

        Returns:
            numpy array of shape (height, width, 3) in RGB format,
            or None if capture failed
        """
        if not self._is_running:
            return None

        if self.use_picamera:
            frame = self._camera.capture_array()
        else:
            ret, frame = self._camera.read()
            if not ret:
                return None
            # OpenCV captures in BGR, convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply rotation if needed
        if self.rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotation == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return frame

    def stop(self) -> None:
        """Stop the camera capture."""
        if not self._is_running:
            return

        if self.use_picamera:
            self._camera.stop()
            self._camera.close()
        else:
            self._camera.release()

        self._camera = None
        self._is_running = False

    @property
    def is_running(self) -> bool:
        """Check if camera is currently running."""
        return self._is_running

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
