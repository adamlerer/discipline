"""
Cat detection using YOLOv8.

Detects cats in camera frames and returns bounding boxes.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class Detection:
    """A single cat detection."""

    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    center: tuple[int, int]  # (x, y) center point

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    def crop_from(self, frame: np.ndarray) -> np.ndarray:
        """Extract the detected region from a frame."""
        x1, y1, x2, y2 = self.bbox
        return frame[y1:y2, x1:x2].copy()


class CatDetector:
    """
    Detects cats in images using YOLOv8.
    """

    # COCO class ID for 'cat'
    CAT_CLASS_ID = 15

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        device: Optional[str] = None,
    ):
        """
        Initialize the cat detector.

        Args:
            model_path: Path to YOLOv8 model weights
            confidence_threshold: Minimum confidence for detections
            device: Device to run inference on ('cpu', 'cuda', or None for auto)
        """
        self.confidence_threshold = confidence_threshold
        self.device = device

        # Import ultralytics here to avoid loading it if not needed
        from ultralytics import YOLO

        self.model = YOLO(model_path)

        # Move to specified device if provided
        if device:
            self.model.to(device)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Detect cats in a frame.

        Args:
            frame: RGB image as numpy array (height, width, 3)

        Returns:
            List of Detection objects for each cat found
        """
        # Run inference
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            classes=[self.CAT_CLASS_ID],  # Only detect cats
            verbose=False,
        )

        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                # Get bounding box coordinates
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy

                # Get confidence
                conf = float(boxes.conf[i].cpu().numpy())

                # Calculate center point
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                detection = Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    center=(center_x, center_y),
                )
                detections.append(detection)

        return detections

    def detect_with_crops(
        self, frame: np.ndarray
    ) -> list[tuple[Detection, np.ndarray]]:
        """
        Detect cats and return both detections and cropped images.

        Args:
            frame: RGB image as numpy array

        Returns:
            List of (Detection, cropped_image) tuples
        """
        detections = self.detect(frame)
        return [(det, det.crop_from(frame)) for det in detections]
