#!/usr/bin/env python3
"""
Download the YOLOv8 model for cat detection.

This script downloads the YOLOv8 nano model which is optimized for
fast inference on Raspberry Pi.
"""

from pathlib import Path


def main():
    print("Downloading YOLOv8 model...")

    # Import ultralytics - this will download the model automatically
    from ultralytics import YOLO

    # Download YOLOv8 nano model (smallest/fastest)
    model = YOLO("yolov8n.pt")

    print(f"Model downloaded successfully!")
    print(f"Model location: {Path('yolov8n.pt').absolute()}")

    # Test the model
    print("\nTesting model...")
    print(f"Model type: {type(model)}")
    print(f"Model task: {model.task}")

    print("\nModel ready for cat detection!")


if __name__ == "__main__":
    main()
