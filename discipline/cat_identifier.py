"""
Cat identification module.

Distinguishes between Abbi and Ilana using a trained classifier.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class Identification:
    """Result of cat identification."""

    cat_name: str  # "abbi", "ilana", or "unknown"
    confidence: float
    probabilities: dict[str, float]  # Probability for each cat


class CatIdentifier:
    """
    Identifies which cat (Abbi or Ilana) is in an image.

    Uses a simple CNN or feature-based classifier trained on images of each cat.
    """

    def __init__(
        self,
        model_path: str = "models/cat_classifier.pkl",
        confidence_threshold: float = 0.7,
    ):
        """
        Initialize the cat identifier.

        Args:
            model_path: Path to the trained classifier model
            confidence_threshold: Minimum confidence to make identification
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.scaler = None
        self.cat_names = ["abbi", "ilana"]

        if self.model_path.exists():
            self.load_model()

    def load_model(self) -> None:
        """Load the trained classifier from disk."""
        with open(self.model_path, "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.scaler = data.get("scaler")
            self.cat_names = data.get("cat_names", ["abbi", "ilana"])

    def save_model(self) -> None:
        """Save the trained classifier to disk."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "scaler": self.scaler,
                    "cat_names": self.cat_names,
                },
                f,
            )

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract features from a cat image for classification.

        Uses color histograms and basic shape features.

        Args:
            image: RGB image of a cat

        Returns:
            Feature vector as numpy array
        """
        # Resize to standard size
        image = cv2.resize(image, (128, 128))

        features = []

        # Color histogram features (in HSV space for better color representation)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        for i, channel in enumerate(cv2.split(hsv)):
            hist = cv2.calcHist([channel], [0], None, [32], [0, 256])
            hist = hist.flatten() / hist.sum()  # Normalize
            features.extend(hist)

        # Grayscale texture features using LBP-like patterns
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Simple edge density features
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.mean(edges) / 255.0
        features.append(edge_density)

        # Gradient orientation histogram
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx) * 180 / np.pi
        angle_hist, _ = np.histogram(angle.flatten(), bins=16, range=(-180, 180))
        angle_hist = angle_hist / (angle_hist.sum() + 1e-7)
        features.extend(angle_hist)

        # Mean color values
        for channel in cv2.split(image):
            features.append(np.mean(channel) / 255.0)
            features.append(np.std(channel) / 255.0)

        return np.array(features, dtype=np.float32)

    def train(
        self, training_images: dict[str, list[np.ndarray]]
    ) -> dict[str, float]:
        """
        Train the classifier on labeled cat images.

        Args:
            training_images: Dict mapping cat name to list of images

        Returns:
            Training metrics (accuracy, etc.)
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler

        # Extract features from all images
        X = []
        y = []

        self.cat_names = list(training_images.keys())

        for cat_idx, (cat_name, images) in enumerate(training_images.items()):
            for image in images:
                features = self.extract_features(image)
                X.append(features)
                y.append(cat_idx)

        X = np.array(X)
        y = np.array(y)

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train classifier
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_scaled, y)

        # Cross-validation score
        scores = cross_val_score(self.model, X_scaled, y, cv=5)

        # Save the model
        self.save_model()

        return {
            "accuracy": float(np.mean(scores)),
            "accuracy_std": float(np.std(scores)),
            "n_samples": len(y),
        }

    def identify(self, image: np.ndarray) -> Identification:
        """
        Identify which cat is in the image.

        Args:
            image: RGB image of a cat (cropped from detection)

        Returns:
            Identification result
        """
        if self.model is None:
            return Identification(
                cat_name="unknown",
                confidence=0.0,
                probabilities={name: 0.0 for name in self.cat_names},
            )

        # Extract features
        features = self.extract_features(image)
        features = features.reshape(1, -1)

        # Scale features
        if self.scaler is not None:
            features = self.scaler.transform(features)

        # Get prediction probabilities
        probs = self.model.predict_proba(features)[0]

        # Create probability dict
        probabilities = {
            name: float(prob) for name, prob in zip(self.cat_names, probs)
        }

        # Get best prediction
        best_idx = np.argmax(probs)
        best_prob = probs[best_idx]

        if best_prob >= self.confidence_threshold:
            cat_name = self.cat_names[best_idx]
        else:
            cat_name = "unknown"

        return Identification(
            cat_name=cat_name,
            confidence=float(best_prob),
            probabilities=probabilities,
        )

    @property
    def is_trained(self) -> bool:
        """Check if the classifier has been trained."""
        return self.model is not None
