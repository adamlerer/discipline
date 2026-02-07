"""
Cat detection using YOLOv8 with ONNX Runtime.

Detects cats in camera frames and returns bounding boxes.
Supports both ONNX models (for ARM/Pi) and PyTorch models (when available).
"""

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
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

    Automatically uses ONNX Runtime if PyTorch is not available.
    """

    # COCO class ID for 'cat'
    CAT_CLASS_ID = 15

    # YOLOv8n input size
    INPUT_SIZE = 640

    # URL for pre-exported ONNX model
    ONNX_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.onnx"

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        device: Optional[str] = None,
    ):
        """
        Initialize the cat detector.

        Args:
            model_path: Path to YOLOv8 model weights (.pt or .onnx)
            confidence_threshold: Minimum confidence for detections
            device: Device to run inference on ('cpu', 'cuda', or None for auto)
        """
        self.confidence_threshold = confidence_threshold
        self.device = device
        self._use_onnx = False
        self.model = None
        self._ort_session = None

        # Check if PyTorch works before trying ultralytics
        pytorch_available = self._check_pytorch()

        if pytorch_available:
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
                if device:
                    self.model.to(device)
            except Exception as e:
                print(f"Failed to load ultralytics ({e}), using ONNX Runtime")
                self._use_onnx = True
                self._init_onnx(model_path)
        else:
            print("PyTorch not available, using ONNX Runtime")
            self._use_onnx = True
            self._init_onnx(model_path)

    def _check_pytorch(self) -> bool:
        """Check if PyTorch is available and working."""
        try:
            import subprocess
            import sys
            # Run a quick test in a subprocess to avoid crashing the main process
            result = subprocess.run(
                [sys.executable, "-c", "import torch; print('ok')"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0 and "ok" in result.stdout
        except Exception:
            return False

    def _init_onnx(self, model_path: str) -> None:
        """Initialize ONNX Runtime session."""
        import onnxruntime as ort

        # Determine ONNX model path
        onnx_path = Path(model_path)
        if onnx_path.suffix == ".pt":
            onnx_path = onnx_path.with_suffix(".onnx")

        # Download if needed
        if not onnx_path.exists():
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            onnx_path = models_dir / "yolov8n.onnx"

            if not onnx_path.exists():
                print(f"Downloading YOLOv8n ONNX model to {onnx_path}...")
                urllib.request.urlretrieve(self.ONNX_MODEL_URL, onnx_path)
                print("Download complete.")

        # Create ONNX Runtime session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = ['CPUExecutionProvider']
        self._ort_session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_options,
            providers=providers,
        )

        # Get input details
        self._input_name = self._ort_session.get_inputs()[0].name

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        """
        Preprocess frame for YOLO inference.

        Returns:
            Preprocessed image, scale factor, and padding offset
        """
        h, w = frame.shape[:2]

        # Calculate scale to fit in INPUT_SIZE while maintaining aspect ratio
        scale = min(self.INPUT_SIZE / w, self.INPUT_SIZE / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize
        resized = cv2.resize(frame, (new_w, new_h))

        # Pad to square
        pad_w = (self.INPUT_SIZE - new_w) // 2
        pad_h = (self.INPUT_SIZE - new_h) // 2

        padded = np.full((self.INPUT_SIZE, self.INPUT_SIZE, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized

        # Convert to float and normalize
        img = padded.astype(np.float32) / 255.0

        # HWC to CHW, add batch dimension
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        return img, scale, (pad_w, pad_h)

    def _postprocess(
        self,
        output: np.ndarray,
        scale: float,
        pad: tuple[int, int],
        orig_shape: tuple[int, int],
    ) -> list[Detection]:
        """
        Postprocess YOLO output to get detections.

        Args:
            output: Raw model output [1, 84, 8400] for YOLOv8
            scale: Scale factor used in preprocessing
            pad: Padding (pad_w, pad_h) used in preprocessing
            orig_shape: Original image shape (h, w)
        """
        # YOLOv8 output shape is [1, 84, 8400]
        # 84 = 4 (bbox) + 80 (classes)
        # Transpose to [8400, 84]
        predictions = output[0].T

        # Get boxes, scores
        boxes = predictions[:, :4]  # x_center, y_center, width, height
        scores = predictions[:, 4:]  # class scores

        # Get cat class scores
        cat_scores = scores[:, self.CAT_CLASS_ID]

        # Filter by confidence
        mask = cat_scores > self.confidence_threshold
        boxes = boxes[mask]
        cat_scores = cat_scores[mask]

        if len(boxes) == 0:
            return []

        # Convert from center format to corner format
        x_center, y_center, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2

        # Remove padding and scale back to original size
        pad_w, pad_h = pad
        x1 = (x1 - pad_w) / scale
        y1 = (y1 - pad_h) / scale
        x2 = (x2 - pad_w) / scale
        y2 = (y2 - pad_h) / scale

        # Clip to image bounds
        orig_h, orig_w = orig_shape
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)

        # Apply NMS
        boxes_for_nms = np.stack([x1, y1, x2, y2], axis=1)
        indices = self._nms(boxes_for_nms, cat_scores, iou_threshold=0.45)

        # Create detections
        detections = []
        for i in indices:
            bbox = (int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i]))
            center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            detections.append(Detection(
                bbox=bbox,
                confidence=float(cat_scores[i]),
                center=center,
            ))

        return detections

    def _nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float = 0.45,
    ) -> list[int]:
        """Non-maximum suppression."""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        order = scores.argsort()[::-1]
        keep = []

        while len(order) > 0:
            i = order[0]
            keep.append(i)

            if len(order) == 1:
                break

            # Compute IoU with rest
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            union = areas[i] + areas[order[1:]] - inter
            iou = inter / union

            # Keep boxes with IoU below threshold
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Detect cats in a frame.

        Args:
            frame: RGB image as numpy array (height, width, 3)

        Returns:
            List of Detection objects for each cat found
        """
        if self._use_onnx:
            return self._detect_onnx(frame)
        else:
            return self._detect_pytorch(frame)

    def _detect_onnx(self, frame: np.ndarray) -> list[Detection]:
        """Detect using ONNX Runtime."""
        orig_shape = frame.shape[:2]

        # Preprocess
        img, scale, pad = self._preprocess(frame)

        # Run inference
        outputs = self._ort_session.run(None, {self._input_name: img})

        # Postprocess
        return self._postprocess(outputs[0], scale, pad, orig_shape)

    def _detect_pytorch(self, frame: np.ndarray) -> list[Detection]:
        """Detect using PyTorch/ultralytics."""
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
