#!/usr/bin/env python3
"""
Capture training images for cat identification.

This script helps you capture images of each cat for training the
cat identifier. It automatically detects cats in the frame and saves
cropped images to the training directory.
"""

import argparse
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from discipline.camera import Camera
from discipline.cat_detector import CatDetector


def main():
    parser = argparse.ArgumentParser(
        description="Capture training images for cat identification"
    )
    parser.add_argument(
        "--cat",
        required=True,
        choices=["abbi", "ilana"],
        help="Name of the cat to capture images for",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="Number of images to capture (default: 50)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/training",
        help="Output directory for training images",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Minimum delay between captures in seconds",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir) / args.cat
    output_dir.mkdir(parents=True, exist_ok=True)

    # Count existing images
    existing = list(output_dir.glob("*.jpg"))
    start_index = len(existing)

    print(f"Capturing training images for: {args.cat.upper()}")
    print(f"Output directory: {output_dir}")
    print(f"Existing images: {start_index}")
    print(f"Target count: {args.count}")
    print()
    print("Instructions:")
    print("  - Position the camera to see the feeding area")
    print(f"  - Make sure only {args.cat.upper()} is in the frame")
    print("  - The system will automatically capture when a cat is detected")
    print("  - Press 'q' to quit, 's' to manually save current frame")
    print()
    print("Starting in 3 seconds...")
    time.sleep(3)

    # Initialize components
    camera = Camera(resolution=(1280, 720))
    detector = CatDetector(confidence_threshold=0.5)

    captured = 0
    last_capture_time = 0

    try:
        camera.start()
        print("Camera started. Looking for cats...")

        while captured < args.count:
            frame = camera.capture()
            if frame is None:
                continue

            # Detect cats
            detections = detector.detect(frame)

            # Convert for display
            display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Draw detections
            for det in detections:
                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    display,
                    f"Cat ({det.confidence:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            # Show status
            status = f"Captured: {captured}/{args.count} | Press 's' to save, 'q' to quit"
            cv2.putText(
                display,
                status,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow(f"Capturing: {args.cat}", display)
            key = cv2.waitKey(1) & 0xFF

            # Check for quit
            if key == ord("q"):
                print("\nCapture cancelled by user")
                break

            # Auto-capture if cat detected and enough time has passed
            current_time = time.time()
            should_capture = False

            if key == ord("s"):
                # Manual capture
                should_capture = len(detections) > 0
                if not should_capture:
                    print("No cat detected - cannot capture")

            elif (
                len(detections) > 0
                and current_time - last_capture_time >= args.delay
            ):
                # Auto capture
                should_capture = True

            if should_capture and detections:
                # Save the largest detection (closest cat)
                det = max(detections, key=lambda d: d.width * d.height)
                cat_image = det.crop_from(frame)

                # Save image
                filename = output_dir / f"{args.cat}_{start_index + captured:04d}.jpg"
                cv2.imwrite(
                    str(filename),
                    cv2.cvtColor(cat_image, cv2.COLOR_RGB2BGR),
                )

                captured += 1
                last_capture_time = current_time
                print(f"Captured {captured}/{args.count}: {filename.name}")

    finally:
        camera.stop()
        cv2.destroyAllWindows()

    print()
    print(f"Capture complete! Saved {captured} images to {output_dir}")

    if captured >= args.count:
        print(f"\nYou now have enough images for {args.cat}.")
        print("Run 'python scripts/train_classifier.py' to train the classifier.")


if __name__ == "__main__":
    main()
