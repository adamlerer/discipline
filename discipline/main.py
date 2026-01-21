"""
Main controller for the Discipline cat food bowl guardian system.

Ties together all components: camera, detection, identification,
bowl monitoring, and sprayer control.
"""

import signal
import sys
import time
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

        # Runtime state
        self._running = False
        self._frame_count = 0
        self._start_time: Optional[float] = None

        # Debug settings
        debug_cfg = self.config.get("debug", {})
        self.show_video = debug_cfg.get("show_video", False)
        self.save_detections = debug_cfg.get("save_detections", False)
        self.detections_dir = Path(debug_cfg.get("detections_dir", "logs/detections"))

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

        # Create detections directory if saving
        if self.save_detections:
            self.detections_dir.mkdir(parents=True, exist_ok=True)

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

        if not detections:
            return

        # Identify each detected cat
        identified = []
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

        # Update bowl monitor and check for violations
        violations = self.monitor.update(identified)

        # Handle violations
        for violation in violations:
            should_spray = self.monitor.should_spray(violation)

            self.logger.log_violation(
                cat_name=violation.cat_name,
                bowl_name=violation.bowl_name,
                owner_present=violation.owner_present,
                sprayed=should_spray,
            )

            if should_spray and self.sprayer.can_spray():
                self.sprayer.spray()
                self.logger.log_spray(
                    cat_name=violation.cat_name,
                    bowl_name=violation.bowl_name,
                    duration_ms=self.sprayer.duration_ms,
                )

        # Save detection image if configured
        if self.save_detections and detections:
            self._save_detection_image(frame, identified)

    def _show_debug_frame(self, frame: np.ndarray) -> None:
        """Display frame with debug overlays."""
        # Convert to BGR for OpenCV display
        display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Draw bowl zones
        for bowl_name, bowl in self.monitor.bowls.items():
            color = (0, 255, 0) if bowl_name == "abbi" else (255, 0, 0)
            cv2.circle(display, (bowl.x, bowl.y), bowl.radius, color, 2)
            cv2.putText(
                display,
                bowl_name.upper(),
                (bowl.x - 30, bowl.y - bowl.radius - 10),
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
        }

        self.logger.log_shutdown(stats)

        # Cleanup
        if self.camera and self.camera.is_running:
            self.camera.stop()

        if self.sprayer:
            self.sprayer.cleanup()

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
