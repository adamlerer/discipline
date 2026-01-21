#!/usr/bin/env python3
"""
Train the cat classifier to distinguish between Abbi and Ilana.

This script loads training images from data/training/ and trains
a classifier to identify which cat is in an image.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from discipline.cat_identifier import CatIdentifier


def load_training_images(data_dir: Path) -> dict[str, list[np.ndarray]]:
    """Load training images from directory structure."""
    images = {}

    for cat_dir in data_dir.iterdir():
        if not cat_dir.is_dir():
            continue

        cat_name = cat_dir.name
        cat_images = []

        for img_path in cat_dir.glob("*.jpg"):
            img = cv2.imread(str(img_path))
            if img is not None:
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cat_images.append(img)

        if cat_images:
            images[cat_name] = cat_images
            print(f"Loaded {len(cat_images)} images for {cat_name}")

    return images


def main():
    parser = argparse.ArgumentParser(
        description="Train the cat classifier"
    )
    parser.add_argument(
        "--data-dir",
        default="data/training",
        help="Directory containing training images",
    )
    parser.add_argument(
        "--model-path",
        default="models/cat_classifier.pkl",
        help="Output path for trained model",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=20,
        help="Minimum images per cat required for training",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Training data directory not found: {data_dir}")
        print("Run 'python scripts/capture_training_images.py' first to collect images.")
        sys.exit(1)

    print("Loading training images...")
    images = load_training_images(data_dir)

    if len(images) < 2:
        print(f"Error: Need training images for at least 2 cats.")
        print(f"Found cats: {list(images.keys())}")
        print("Run 'python scripts/capture_training_images.py --cat <name>' for each cat.")
        sys.exit(1)

    # Check minimum images
    for cat_name, cat_images in images.items():
        if len(cat_images) < args.min_images:
            print(f"Warning: Only {len(cat_images)} images for {cat_name}")
            print(f"Recommended: at least {args.min_images} images per cat")

    print()
    print("Training classifier...")

    # Create and train identifier
    identifier = CatIdentifier(model_path=args.model_path)
    metrics = identifier.train(images)

    print()
    print("Training complete!")
    print(f"  Accuracy: {metrics['accuracy']:.1%} (+/- {metrics['accuracy_std']:.1%})")
    print(f"  Samples: {metrics['n_samples']}")
    print(f"  Model saved to: {args.model_path}")

    # Test predictions
    print()
    print("Testing predictions on random samples...")

    for cat_name, cat_images in images.items():
        # Test on a few random images
        test_indices = np.random.choice(
            len(cat_images),
            size=min(3, len(cat_images)),
            replace=False,
        )

        correct = 0
        for idx in test_indices:
            result = identifier.identify(cat_images[idx])
            if result.cat_name == cat_name:
                correct += 1

        print(f"  {cat_name}: {correct}/{len(test_indices)} correct")

    print()
    print("Classifier is ready! You can now run the main system.")


if __name__ == "__main__":
    main()
