"""
Main controller for the Discipline cat food bowl guardian system.

Ties together all components: camera, detection, identification,
bowl monitoring, and sprayer control.
"""

import os
import signal
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from .bowl_monitor import BowlMonitor
from .camera import Camera
from .cat_detector import CatDetector
from .cat_identifier import CatIdentifier
from .logger import DisciplineLogger
from .sound_player import SoundPlayer
from .sprayer import Sprayer


class DisciplineSystem:
    """
    Main controller that orchestrates all components.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the Discipline system.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Initialize logger first
        log_cfg = self.config.get("logging", {})
        self.logger = DisciplineLogger(
            log_file=log_cfg.get("file", "logs/discipline.log"),
            level=log_cfg.get("level", "INFO"),
            max_size_mb=log_cfg.get("max_size_mb", 10),
            backup_count=log_cfg.get("backup_count", 5),
        )

        # Initialize components
        self.camera: Optional[Camera] = None
        self.detector: Optional[CatDetector] = None
        self.identifier: Optional[CatIdentifier] = None
        self.monitor: Optional[BowlMonitor] = None
        self.sprayer: Optional[Sprayer] = None
        self.sound_player: Optional[SoundPlayer] = None
        self.web_app = None  # Will be DisciplineWebApp if enabled

        # Runtime state
        self._running = False
        self._frame_count = 0
        self._start_time: Optional[float] = None

        # Runtime toggles (can override config during runtime)
        self._spray_enabled: Optional[bool] = None  # None = use config default
        self._sound_enabled: Optional[bool] = None  # None = use config default

        # Store current frame and detections for web streaming
        self._current_frame: Optional[np.ndarray] = None
        self._current_detections: list = []
        self._current_violations: set = set()  # Set of cat names currently in violation
        self._frame_lock = threading.Lock()

        # Event log for web interface
        self._recent_events: list = []
        self._max_events = 100

        # Debug settings
        debug_cfg = self.config.get("debug", {})
        self.show_video = debug_cfg.get("show_video", False)
        self.save_detections = debug_cfg.get("save_detections", False)
        self.detections_dir = Path(debug_cfg.get("detections_dir", "logs/detections"))

        # Labeling settings
        labeling_cfg = self.config.get("labeling", {})
        self._labeling_enabled = labeling_cfg.get("enabled", False)
        self._capture_interval_s = labeling_cfg.get("capture_interval_s", 1.0)
        self._max_unlabeled = labeling_cfg.get("max_unlabeled", 500)
        self._min_detection_confidence = labeling_cfg.get("min_detection_confidence", 0.6)
        self._last_capture_time = 0.0

        # Data directories for labeling (absolute paths based on config file location)
        project_root = self.config_path.parent.resolve()
        self._unlabeled_dir = project_root / "data" / "unlabeled"
        self._training_dir = project_root / "data" / "training"

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def initialize(self) -> None:
        """Initialize all system components."""
        self.logger.log_startup(self.config)

        # Camera
        cam_cfg = self.config.get("camera", {})
        resolution = cam_cfg.get("resolution", {})
        self.camera = Camera(
            resolution=(
                resolution.get("width", 1280),
                resolution.get("height", 720),
            ),
            framerate=cam_cfg.get("framerate", 30),
            rotation=cam_cfg.get("rotation", 0),
        )

        # Cat detector
        det_cfg = self.config.get("detection", {})
        self.detector = CatDetector(
            model_path=det_cfg.get("model", "yolov8n.pt"),
            confidence_threshold=det_cfg.get("confidence_threshold", 0.5),
        )

        # Cat identifier
        id_cfg = self.config.get("identification", {})
        self.identifier = CatIdentifier(
            model_path=id_cfg.get("model_path", "models/cat_classifier.pkl"),
            confidence_threshold=id_cfg.get("confidence_threshold", 0.7),
        )

        if not self.identifier.is_trained:
            self.logger.warning(
                "Cat classifier not trained. Run train_classifier.py first."
            )

        # Bowl monitor
        mon_cfg = self.config.get("monitoring", {})
        self.monitor = BowlMonitor(
            bowls_config=self.config.get("bowls", {}),
            cats_config=self.config.get("cats", {}),
            eating_threshold_s=mon_cfg.get("eating_threshold_s", 2.0),
            leave_threshold_s=mon_cfg.get("leave_threshold_s", 5.0),
            proximity_threshold=mon_cfg.get("proximity_threshold", 150),
        )

        # Sprayer
        spray_cfg = self.config.get("spray", {})
        self.sprayer = Sprayer(
            gpio_pin=spray_cfg.get("gpio_pin", 17),
            duration_ms=spray_cfg.get("duration_ms", 500),
            cooldown_s=spray_cfg.get("cooldown_s", 10),
            enabled=spray_cfg.get("enabled", True),
        )

        # Sound player
        sound_cfg = self.config.get("sound", {})
        if sound_cfg.get("enabled", False):
            self.sound_player = SoundPlayer(
                sound_file=sound_cfg.get("file", "sounds/deterrent.mp3"),
                volume=sound_cfg.get("volume", 0.7),
                duration_ms=sound_cfg.get("duration_ms", 2000),
                cooldown_s=sound_cfg.get("cooldown_s", 5),
                enabled=True,
            )
            self.sound_player.initialize()
            self.logger.info("Sound player initialized")

        # Web app
        web_cfg = self.config.get("web", {})
        if web_cfg.get("enabled", False):
            from .web.app import DisciplineWebApp

            self.web_app = DisciplineWebApp(
                system=self,
                host=web_cfg.get("host", "0.0.0.0"),
                port=web_cfg.get("port", 5000),
            )
            self.web_app.start()
            self.logger.info(
                f"Web interface started at http://{web_cfg.get('host', '0.0.0.0')}:{web_cfg.get('port', 5000)}"
            )

        # Create detections directory if saving
        if self.save_detections:
            self.detections_dir.mkdir(parents=True, exist_ok=True)

        # Create labeling directories if enabled
        if self._labeling_enabled:
            self._unlabeled_dir.mkdir(parents=True, exist_ok=True)
            (self._training_dir / "abbi").mkdir(parents=True, exist_ok=True)
            (self._training_dir / "ilana").mkdir(parents=True, exist_ok=True)
            self.logger.info("Labeling enabled - capturing cat images for training")

        self.logger.info("System initialized successfully")

    def start(self) -> None:
        """Start the monitoring system."""
        if self._running:
            return

        self.initialize()

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self._running = True
        self._start_time = time.time()
        self._frame_count = 0

        self.logger.info("Starting camera capture")
        self.camera.start()

        self.logger.info("Initializing sprayer")
        self.sprayer.initialize()

        self.logger.info("System running - monitoring for cats")
        self._main_loop()

    def _main_loop(self) -> None:
        """Main processing loop."""
        while self._running:
            try:
                # Capture frame
                frame = self.camera.capture()
                if frame is None:
                    self.logger.warning("Failed to capture frame")
                    time.sleep(0.1)
                    continue

                self._frame_count += 1

                # Process frame
                self._process_frame(frame)

                # Show video if debug mode
                if self.show_video:
                    self._show_debug_frame(frame)

                # Small delay to control frame rate
                time.sleep(0.033)  # ~30 FPS

            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                time.sleep(1)

    def _process_frame(self, frame: np.ndarray) -> None:
        """
        Process a single frame.

        Args:
            frame: RGB image from camera
        """
        # Detect cats
        detections = self.detector.detect(frame)

        # Identify each detected cat
        identified = []
        if detections:
            for detection in detections:
                # Crop cat from frame
                cat_image = detection.crop_from(frame)

                # Identify which cat it is
                identification = self.identifier.identify(cat_image)

                identified.append((detection, identification))

                # Log detection
                bowl_at = None
                for bowl_name, bowl in self.monitor.bowls.items():
                    if bowl.contains_point(detection.center):
                        bowl_at = bowl_name
                        break

                self.logger.log_detection(
                    cat_name=identification.cat_name,
                    confidence=identification.confidence,
                    position=detection.center,
                    bowl=bowl_at,
                )

        # Capture images for labeling if enabled
        if self._labeling_enabled and detections:
            self._capture_for_labeling(frame, identified)

        if not detections:
            with self._frame_lock:
                self._current_frame = frame.copy()
                self._current_detections = []
                self._current_violations = set()
            return

        # Update bowl monitor and check for violations
        violations = self.monitor.update(identified)

        # Track which cats are currently in violation (at wrong bowl)
        current_violations = set()
        for cat_name, track in self.monitor.cat_tracks.items():
            if track.current_bowl:
                allowed_bowl = self.monitor.cat_allowed_bowls.get(cat_name)
                if allowed_bowl and track.current_bowl != allowed_bowl:
                    current_violations.add(cat_name)

        # Store current frame and detections for web streaming
        with self._frame_lock:
            self._current_frame = frame.copy()
            self._current_detections = identified.copy()
            self._current_violations = current_violations

        # Handle violations
        for violation in violations:
            should_deter = self.monitor.should_spray(violation)

            # Check runtime toggle for spray
            spray_enabled = self._spray_enabled if self._spray_enabled is not None else self.sprayer.enabled
            # Check runtime toggle for sound
            sound_enabled = self._sound_enabled if self._sound_enabled is not None else (
                self.sound_player.enabled if self.sound_player else False
            )

            sprayed = False
            sounded = False

            self.logger.log_violation(
                cat_name=violation.cat_name,
                bowl_name=violation.bowl_name,
                owner_present=violation.owner_present,
                sprayed=should_deter and spray_enabled,
            )

            # Trigger spray if enabled
            if should_deter and spray_enabled and self.sprayer.can_spray():
                self.sprayer.spray()
                sprayed = True
                self.logger.log_spray(
                    cat_name=violation.cat_name,
                    bowl_name=violation.bowl_name,
                    duration_ms=self.sprayer.duration_ms,
                )

            # Trigger sound if enabled
            if should_deter and sound_enabled and self.sound_player and self.sound_player.can_play():
                self.sound_player.play()
                sounded = True
                self.logger.log_sound(
                    cat_name=violation.cat_name,
                    bowl_name=violation.bowl_name,
                    duration_ms=self.sound_player.duration_ms,
                )

            # Add to recent events for web interface
            self._add_event({
                "type": "violation",
                "cat": violation.cat_name,
                "bowl": violation.bowl_name,
                "owner_present": violation.owner_present,
                "sprayed": sprayed,
                "sounded": sounded,
            })

        # Save detection image if configured
        if self.save_detections and detections:
            self._save_detection_image(frame, identified)

    def _add_event(self, event: dict) -> None:
        """Add an event to the recent events log."""
        event["timestamp"] = datetime.now().isoformat()
        self._recent_events.insert(0, event)
        # Keep only the most recent events
        if len(self._recent_events) > self._max_events:
            self._recent_events = self._recent_events[: self._max_events]

    def set_spray_enabled(self, enabled: bool) -> None:
        """Set runtime spray enabled state."""
        self._spray_enabled = enabled
        self.logger.log_deterrent_toggle("spray", enabled, source="web")
        self._add_event({
            "type": "toggle",
            "deterrent": "spray",
            "enabled": enabled,
        })

    def set_sound_enabled(self, enabled: bool) -> None:
        """Set runtime sound enabled state."""
        self._sound_enabled = enabled
        self.logger.log_deterrent_toggle("sound", enabled, source="web")
        self._add_event({
            "type": "toggle",
            "deterrent": "sound",
            "enabled": enabled,
        })

    def get_spray_enabled(self) -> bool:
        """Get current spray enabled state."""
        if self._spray_enabled is not None:
            return self._spray_enabled
        return self.sprayer.enabled if self.sprayer else False

    def get_sound_enabled(self) -> bool:
        """Get current sound enabled state."""
        if self._sound_enabled is not None:
            return self._sound_enabled
        return self.sound_player.enabled if self.sound_player else False

    def get_deterrent_state(self) -> dict:
        """Get current deterrent state for web API."""
        return {
            "spray": {
                "enabled": self.get_spray_enabled(),
                "stats": self.sprayer.get_stats() if self.sprayer else {},
            },
            "sound": {
                "enabled": self.get_sound_enabled(),
                "stats": self.sound_player.get_stats() if self.sound_player else {},
                "available": self.sound_player is not None,
            },
        }

    def get_annotated_frame(self) -> Optional[np.ndarray]:
        """
        Get the current frame with bounding boxes and cat labels.

        Returns:
            Annotated frame in BGR format for MJPEG streaming, or None
        """
        with self._frame_lock:
            if self._current_frame is None:
                return None

            frame = self._current_frame.copy()
            detections = self._current_detections.copy()
            violations = self._current_violations.copy()

        # Convert to BGR for OpenCV
        annotated = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Note: Bowl zones are drawn via SVG overlay in the web UI

        # Draw bounding boxes for detected cats
        for detection, identification in detections:
            x1, y1, x2, y2 = detection.bbox
            is_violation = identification.cat_name in violations

            # Color based on cat identity
            if identification.cat_name == "abbi":
                color = (0, 255, 0)  # Green for Abbi
            elif identification.cat_name == "ilana":
                color = (255, 0, 0)  # Blue for Ilana (BGR)
            else:
                color = (0, 255, 255)  # Yellow for unknown

            # Draw semi-transparent red fill if in violation
            if is_violation:
                overlay = annotated.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)  # Red fill
                cv2.addWeighted(overlay, 0.3, annotated, 0.7, 0, annotated)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{identification.cat_name} ({identification.confidence:.0%})"
            if is_violation:
                label = f"VIOLATION: {label}"

            # Position label inside box at top (with background for visibility)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            label_y = y1 + text_h + 5
            # Draw background rectangle
            cv2.rectangle(annotated, (x1, y1), (x1 + text_w + 4, label_y + 2), (0, 0, 0), -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 2, label_y - 2),
                font,
                font_scale,
                (0, 0, 255) if is_violation else color,
                thickness,
            )

        return annotated

    def get_recent_events(self, limit: int = 50) -> list:
        """Get recent events for web interface."""
        return self._recent_events[:limit]

    def get_system_status(self) -> dict:
        """Get full system status for web API."""
        runtime = 0
        if self._start_time:
            runtime = time.time() - self._start_time

        return {
            "running": self._running,
            "runtime_s": round(runtime, 1),
            "frames_processed": self._frame_count,
            "violations_detected": len(self.monitor.violations) if self.monitor else 0,
            "deterrents": self.get_deterrent_state(),
            "cats_config": self.config.get("cats", {}),
            "bowls_config": self.config.get("bowls", {}),
        }

    def get_bowl_positions(self) -> dict:
        """Get current bowl positions."""
        return self.config.get("bowls", {})

    def update_bowl_position(self, bowl_name: str, x: int, y: int, radius: int) -> bool:
        """
        Update a bowl's position and save to config.

        Args:
            bowl_name: "abbi" or "ilana"
            x: Center x coordinate
            y: Center y coordinate
            radius: Detection zone radius

        Returns:
            True if successful
        """
        if bowl_name not in ("abbi", "ilana"):
            return False

        # Update in-memory config
        if "bowls" not in self.config:
            self.config["bowls"] = {}
        if bowl_name not in self.config["bowls"]:
            self.config["bowls"][bowl_name] = {}

        self.config["bowls"][bowl_name]["x"] = x
        self.config["bowls"][bowl_name]["y"] = y
        self.config["bowls"][bowl_name]["radius"] = radius

        # Update the bowl monitor if it exists
        if self.monitor and bowl_name in self.monitor.bowls:
            self.monitor.bowls[bowl_name].x = x
            self.monitor.bowls[bowl_name].y = y
            self.monitor.bowls[bowl_name].radius = radius

        # Save to config file
        self._save_config()

        self.logger.info(f"Updated bowl position: {bowl_name} at ({x}, {y}) radius {radius}")
        return True

    def _save_config(self) -> None:
        """Save current config to file."""
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

    def _capture_for_labeling(
        self,
        frame: np.ndarray,
        identified: list,
    ) -> None:
        """
        Capture cropped cat images for labeling.

        Args:
            frame: RGB image from camera
            identified: List of (detection, identification) tuples
        """
        current_time = time.time()

        # Check if enough time has passed since last capture
        if current_time - self._last_capture_time < self._capture_interval_s:
            return

        for detection, identification in identified:
            # Only capture if detection confidence is high enough
            if detection.confidence < self._min_detection_confidence:
                continue

            # Crop cat from frame
            cat_image = detection.crop_from(frame)

            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            detection_id = str(uuid.uuid4())[:8]
            filename = f"{timestamp}_{detection_id}.jpg"
            filepath = self._unlabeled_dir / filename

            # Convert to BGR and save
            cv2.imwrite(str(filepath), cv2.cvtColor(cat_image, cv2.COLOR_RGB2BGR))
            self._last_capture_time = current_time

            self.logger.debug(f"Captured image for labeling: {filename}")

        # Prune old images if over max
        self._prune_unlabeled_images()

    def _prune_unlabeled_images(self) -> None:
        """Remove oldest unlabeled images if over max limit."""
        if not self._unlabeled_dir.exists():
            return

        images = sorted(
            self._unlabeled_dir.glob("*.jpg"),
            key=lambda p: p.stat().st_mtime,
        )

        while len(images) > self._max_unlabeled:
            oldest = images.pop(0)
            oldest.unlink()
            self.logger.debug(f"Pruned old unlabeled image: {oldest.name}")

    def get_unlabeled_images(self, with_predictions: bool = False) -> list:
        """
        Get list of unlabeled images with metadata and optionally classifier predictions.

        Args:
            with_predictions: If True, run classifier on each image (slower)

        Returns:
            List of dicts with filename, timestamp, and optionally predicted cat
        """
        if not self._unlabeled_dir.exists():
            return []

        images = []
        for filepath in sorted(
            self._unlabeled_dir.glob("*.jpg"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,  # Most recent first
        ):
            stat = filepath.stat()
            image_data = {
                "filename": filepath.name,
                "timestamp": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "size": stat.st_size,
                "prediction": None,
                "confidence": None,
            }

            # Run classifier if available and requested
            if with_predictions and self.identifier and self.identifier.is_trained:
                try:
                    img = cv2.imread(str(filepath))
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        result = self.identifier.identify(img_rgb)
                        image_data["prediction"] = result.cat_name
                        image_data["confidence"] = round(result.confidence, 2)
                except Exception:
                    pass  # Skip prediction on error

            images.append(image_data)

        return images

    def label_image(self, filename: str, cat_name: str) -> bool:
        """
        Move an image from unlabeled to the appropriate training folder.

        Args:
            filename: Name of the image file
            cat_name: "abbi" or "ilana"

        Returns:
            True if successful, False otherwise
        """
        if cat_name not in ("abbi", "ilana"):
            return False

        source = self._unlabeled_dir / filename
        if not source.exists():
            return False

        dest_dir = self._training_dir / cat_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / filename

        source.rename(dest)
        self.logger.info(f"Labeled image {filename} as {cat_name}")
        return True

    def skip_image(self, filename: str) -> bool:
        """
        Delete an unlabeled image (skip it).

        Args:
            filename: Name of the image file

        Returns:
            True if successful, False otherwise
        """
        filepath = self._unlabeled_dir / filename
        if not filepath.exists():
            return False

        filepath.unlink()
        self.logger.debug(f"Skipped image: {filename}")
        return True

    def get_labeled_images(self, cat_name: str) -> list:
        """
        Get list of labeled images for a specific cat.

        Args:
            cat_name: "abbi" or "ilana"

        Returns:
            List of dicts with filename and timestamp
        """
        cat_dir = self._training_dir / cat_name
        if not cat_dir.exists():
            return []

        images = []
        for filepath in sorted(
            cat_dir.glob("*.jpg"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            stat = filepath.stat()
            images.append({
                "filename": filepath.name,
                "cat": cat_name,
                "timestamp": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "size": stat.st_size,
            })

        return images

    def get_labeled_image_path(self, cat_name: str, filename: str) -> Optional[Path]:
        """
        Get the full path to a labeled image.

        Args:
            cat_name: "abbi" or "ilana"
            filename: Name of the image file

        Returns:
            Absolute path to the image or None if not found
        """
        if cat_name not in ("abbi", "ilana"):
            return None

        filepath = self._training_dir / cat_name / filename
        if filepath.exists():
            return filepath.resolve()
        return None

    def swap_image_label(self, cat_name: str, filename: str) -> bool:
        """
        Swap an image's label from one cat to the other.

        Args:
            cat_name: Current cat ("abbi" or "ilana")
            filename: Name of the image file

        Returns:
            True if successful
        """
        if cat_name not in ("abbi", "ilana"):
            return False

        new_cat = "ilana" if cat_name == "abbi" else "abbi"

        source = self._training_dir / cat_name / filename
        if not source.exists():
            return False

        dest_dir = self._training_dir / new_cat
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / filename

        source.rename(dest)
        self.logger.info(f"Swapped image {filename} from {cat_name} to {new_cat}")
        return True

    def delete_labeled_images(self, images: list) -> dict:
        """
        Delete multiple labeled images.

        Args:
            images: List of {"cat": "abbi"/"ilana", "filename": "..."}

        Returns:
            Dict with deleted count and any errors
        """
        deleted = 0
        errors = []

        for img in images:
            cat_name = img.get("cat")
            filename = img.get("filename")

            if cat_name not in ("abbi", "ilana"):
                errors.append(f"Invalid cat: {cat_name}")
                continue

            filepath = self._training_dir / cat_name / filename
            if filepath.exists():
                try:
                    filepath.unlink()
                    deleted += 1
                    self.logger.info(f"Deleted labeled image: {cat_name}/{filename}")
                except Exception as e:
                    errors.append(f"Failed to delete {filename}: {e}")
            else:
                errors.append(f"Not found: {cat_name}/{filename}")

        return {"deleted": deleted, "errors": errors}

    def get_labeling_stats(self) -> dict:
        """
        Get counts of unlabeled and labeled images.

        Returns:
            Dict with counts for unlabeled, abbi, ilana
        """
        unlabeled_count = len(list(self._unlabeled_dir.glob("*.jpg"))) if self._unlabeled_dir.exists() else 0
        abbi_count = len(list((self._training_dir / "abbi").glob("*.jpg"))) if (self._training_dir / "abbi").exists() else 0
        ilana_count = len(list((self._training_dir / "ilana").glob("*.jpg"))) if (self._training_dir / "ilana").exists() else 0

        return {
            "unlabeled": unlabeled_count,
            "abbi": abbi_count,
            "ilana": ilana_count,
            "enabled": self._labeling_enabled,
        }

    def get_unlabeled_image_path(self, filename: str) -> Optional[Path]:
        """
        Get the full path to an unlabeled image.

        Args:
            filename: Name of the image file

        Returns:
            Absolute path to the image or None if not found
        """
        filepath = self._unlabeled_dir / filename
        if filepath.exists():
            return filepath.resolve()  # Return absolute path
        return None

    def _show_debug_frame(self, frame: np.ndarray) -> None:
        """Display frame with debug overlays."""
        # Convert to BGR for OpenCV display
        display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Draw bowl zones
        for bowl_name, bowl in self.monitor.bowls.items():
            color = (0, 255, 0) if bowl_name == "abbi" else (255, 0, 0)
            cv2.circle(display, (bowl.x, bowl.y), bowl.radius, color, 2)
            label = f"{bowl_name.upper()} BOWL"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = bowl.x - text_size[0] // 2
            cv2.putText(
                display,
                label,
                (text_x, bowl.y - bowl.radius - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

        # Draw cat tracks
        for name, track in self.monitor.cat_tracks.items():
            if track.position:
                cv2.circle(display, track.position, 10, (0, 255, 255), -1)
                cv2.putText(
                    display,
                    name,
                    (track.position[0] + 15, track.position[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

        # Show frame
        cv2.imshow("Discipline - Cat Monitor", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self._running = False

    def _save_detection_image(
        self,
        frame: np.ndarray,
        detections: list,
    ) -> None:
        """Save an annotated detection image."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = self.detections_dir / f"detection_{timestamp}.jpg"

        # Draw bounding boxes
        annotated = frame.copy()
        for detection, identification in detections:
            x1, y1, x2, y2 = detection.bbox
            color = (0, 255, 0) if identification.cat_name == "abbi" else (255, 0, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{identification.cat_name} ({identification.confidence:.2f})"
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        # Convert to BGR and save
        cv2.imwrite(str(filename), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.stop()

    def stop(self) -> None:
        """Stop the monitoring system."""
        self._running = False

        # Calculate stats
        runtime = 0
        if self._start_time:
            runtime = time.time() - self._start_time

        stats = {
            "runtime_s": round(runtime, 1),
            "frames_processed": self._frame_count,
            "violations_detected": len(self.monitor.violations) if self.monitor else 0,
            "sprays_triggered": self.sprayer.spray_count if self.sprayer else 0,
            "sounds_triggered": self.sound_player.play_count if self.sound_player else 0,
        }

        self.logger.log_shutdown(stats)

        # Cleanup
        if self.camera and self.camera.is_running:
            self.camera.stop()

        if self.sprayer:
            self.sprayer.cleanup()

        if self.sound_player:
            self.sound_player.cleanup()

        if self.web_app:
            self.web_app.stop()

        if self.show_video:
            cv2.destroyAllWindows()


def main():
    """Entry point for the discipline system."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Discipline - Cat Food Bowl Guardian"
    )
    parser.add_argument(
        "-c",
        "--config",
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with video display",
    )
    parser.add_argument(
        "--no-spray",
        action="store_true",
        help="Disable sprayer (test mode)",
    )

    args = parser.parse_args()

    # Create and run system
    system = DisciplineSystem(config_path=args.config)

    # Override settings from command line
    if args.debug:
        system.show_video = True

    if args.no_spray:
        system.config["spray"]["enabled"] = False

    try:
        system.start()
    except KeyboardInterrupt:
        pass
    finally:
        system.stop()


if __name__ == "__main__":
    main()
