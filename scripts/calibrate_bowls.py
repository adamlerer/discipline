#!/usr/bin/env python3
"""
Calibrate bowl positions in the camera frame.

This interactive tool helps you mark the positions of Abbi's and Ilana's
food bowls in the camera view. The positions are saved to config.yaml.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import yaml

from discipline.camera import Camera


class BowlCalibrator:
    """Interactive bowl position calibrator."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

        self.bowls = {
            "abbi": {"x": 200, "y": 300, "radius": 100},
            "ilana": {"x": 500, "y": 300, "radius": 100},
        }

        # Load existing bowl config if present
        if "bowls" in self.config:
            for name, cfg in self.config["bowls"].items():
                if name in self.bowls:
                    self.bowls[name].update(cfg)

        self.selected_bowl = "abbi"
        self.dragging = False
        self.drag_start = None

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f) or {}
        return {}

    def _save_config(self) -> None:
        """Save configuration to YAML file."""
        self.config["bowls"] = self.bowls

        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

        print(f"Configuration saved to {self.config_path}")

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for bowl positioning."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if clicking on a bowl
            for name, bowl in self.bowls.items():
                dist = np.sqrt((x - bowl["x"]) ** 2 + (y - bowl["y"]) ** 2)
                if dist < bowl["radius"]:
                    self.selected_bowl = name
                    self.dragging = True
                    self.drag_start = (x, y)
                    return

            # Click on empty space - move selected bowl
            self.bowls[self.selected_bowl]["x"] = x
            self.bowls[self.selected_bowl]["y"] = y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging:
                self.bowls[self.selected_bowl]["x"] = x
                self.bowls[self.selected_bowl]["y"] = y

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.drag_start = None

        elif event == cv2.EVENT_MOUSEWHEEL:
            # Scroll to adjust radius
            bowl = self.bowls[self.selected_bowl]
            if flags > 0:
                bowl["radius"] = min(300, bowl["radius"] + 10)
            else:
                bowl["radius"] = max(20, bowl["radius"] - 10)

    def run(self):
        """Run the calibration interface."""
        print("Bowl Calibration Tool")
        print("=" * 40)
        print()
        print("Instructions:")
        print("  - Click on a bowl circle to select it")
        print("  - Click anywhere to move the selected bowl")
        print("  - Drag to reposition a bowl")
        print("  - Scroll wheel to adjust bowl zone radius")
        print("  - Press '1' to select Abbi's bowl")
        print("  - Press '2' to select Ilana's bowl")
        print("  - Press 's' to save and exit")
        print("  - Press 'q' to quit without saving")
        print()

        camera = Camera(resolution=(1280, 720))

        window_name = "Bowl Calibration"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        try:
            camera.start()
            print("Camera started. Position the bowls...")

            while True:
                frame = camera.capture()
                if frame is None:
                    continue

                # Convert to BGR for display
                display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Draw bowl zones
                for name, bowl in self.bowls.items():
                    # Bowl color (green for Abbi, blue for Ilana)
                    if name == "abbi":
                        color = (0, 255, 0)
                    else:
                        color = (255, 0, 0)

                    # Highlight selected bowl
                    thickness = 3 if name == self.selected_bowl else 2

                    # Draw circle
                    cv2.circle(
                        display,
                        (bowl["x"], bowl["y"]),
                        bowl["radius"],
                        color,
                        thickness,
                    )

                    # Draw center point
                    cv2.circle(
                        display,
                        (bowl["x"], bowl["y"]),
                        5,
                        color,
                        -1,
                    )

                    # Draw label
                    label = f"{name.upper()}"
                    if name == self.selected_bowl:
                        label += " [SELECTED]"

                    cv2.putText(
                        display,
                        label,
                        (bowl["x"] - 50, bowl["y"] - bowl["radius"] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                    )

                    # Show radius
                    cv2.putText(
                        display,
                        f"r={bowl['radius']}",
                        (bowl["x"] - 20, bowl["y"] + bowl["radius"] + 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                    )

                # Draw instructions
                instructions = [
                    "1/2: Select bowl | Click: Move | Scroll: Resize",
                    "S: Save | Q: Quit",
                ]
                for i, text in enumerate(instructions):
                    cv2.putText(
                        display,
                        text,
                        (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

                cv2.imshow(window_name, display)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    print("\nCalibration cancelled.")
                    break

                elif key == ord("s"):
                    self._save_config()
                    print("\nCalibration saved!")
                    break

                elif key == ord("1"):
                    self.selected_bowl = "abbi"
                    print("Selected: Abbi's bowl")

                elif key == ord("2"):
                    self.selected_bowl = "ilana"
                    print("Selected: Ilana's bowl")

        finally:
            camera.stop()
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate bowl positions in camera view"
    )
    parser.add_argument(
        "-c",
        "--config",
        default="config.yaml",
        help="Path to configuration file",
    )

    args = parser.parse_args()

    calibrator = BowlCalibrator(config_path=args.config)
    calibrator.run()


if __name__ == "__main__":
    main()
