"""
Bowl monitoring module.

Tracks which cats are at which bowls and determines if any violations occur.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .cat_detector import Detection
from .cat_identifier import Identification


class BowlState(Enum):
    """State of a bowl."""

    EMPTY = "empty"  # No cat present
    OCCUPIED = "occupied"  # A cat is eating
    RECENTLY_LEFT = "recently_left"  # Cat just left, still their bowl


@dataclass
class Bowl:
    """Represents a food bowl and its zone."""

    name: str  # "abbi" or "ilana"
    x: int  # Center X coordinate in frame
    y: int  # Center Y coordinate in frame
    radius: int  # Detection zone radius

    state: BowlState = BowlState.EMPTY
    current_cat: Optional[str] = None  # Name of cat currently at bowl
    owner_eating_since: Optional[float] = None  # Timestamp when owner started eating
    last_occupied: Optional[float] = None  # Last time bowl was occupied

    def contains_point(self, point: tuple[int, int]) -> bool:
        """Check if a point is within the bowl's zone."""
        px, py = point
        distance = ((px - self.x) ** 2 + (py - self.y) ** 2) ** 0.5
        return distance <= self.radius

    def distance_to(self, point: tuple[int, int]) -> float:
        """Calculate distance from point to bowl center."""
        px, py = point
        return ((px - self.x) ** 2 + (py - self.y) ** 2) ** 0.5


@dataclass
class Violation:
    """A violation event - wrong cat at wrong bowl."""

    timestamp: float
    cat_name: str  # The offending cat
    bowl_name: str  # The bowl they're at
    owner_present: bool  # Whether the bowl's owner is also present


@dataclass
class CatTrack:
    """Tracks a cat's position and state over time."""

    cat_name: str
    current_bowl: Optional[str] = None
    at_bowl_since: Optional[float] = None
    last_seen: float = field(default_factory=time.time)
    position: Optional[tuple[int, int]] = None


class BowlMonitor:
    """
    Monitors bowl zones and tracks cat positions to detect violations.
    """

    def __init__(
        self,
        bowls_config: dict,
        cats_config: dict,
        eating_threshold_s: float = 2.0,
        leave_threshold_s: float = 5.0,
        proximity_threshold: int = 150,
    ):
        """
        Initialize the bowl monitor.

        Args:
            bowls_config: Bowl configuration from config.yaml
            cats_config: Cat configuration from config.yaml
            eating_threshold_s: Time at bowl before considered "eating"
            leave_threshold_s: Time after leaving before bowl is "available"
            proximity_threshold: Distance threshold for being "at" a bowl
        """
        self.eating_threshold = eating_threshold_s
        self.leave_threshold = leave_threshold_s
        self.proximity_threshold = proximity_threshold

        # Create bowl objects
        self.bowls: dict[str, Bowl] = {}
        for name, cfg in bowls_config.items():
            self.bowls[name] = Bowl(
                name=name,
                x=cfg["x"],
                y=cfg["y"],
                radius=cfg.get("radius", proximity_threshold),
            )

        # Map cats to their allowed bowls
        self.cat_allowed_bowls: dict[str, str] = {}
        for cat_name, cfg in cats_config.items():
            self.cat_allowed_bowls[cat_name] = cfg.get("allowed_bowl", cat_name)

        # Track cats
        self.cat_tracks: dict[str, CatTrack] = {}

        # Recent violations
        self.violations: list[Violation] = []

    def update(
        self,
        detections: list[tuple[Detection, Identification]],
    ) -> list[Violation]:
        """
        Update bowl states with new detections and check for violations.

        Args:
            detections: List of (Detection, Identification) tuples

        Returns:
            List of new violations detected
        """
        current_time = time.time()
        new_violations = []

        # Track which bowls have cats this frame
        bowls_with_cats: dict[str, list[str]] = {name: [] for name in self.bowls}

        # Process each detection
        for detection, identification in detections:
            cat_name = identification.cat_name
            if cat_name == "unknown":
                continue

            center = detection.center

            # Update cat track
            if cat_name not in self.cat_tracks:
                self.cat_tracks[cat_name] = CatTrack(cat_name=cat_name)

            track = self.cat_tracks[cat_name]
            track.last_seen = current_time
            track.position = center

            # Find which bowl (if any) this cat is at
            cat_at_bowl = None
            min_distance = float("inf")

            for bowl_name, bowl in self.bowls.items():
                distance = bowl.distance_to(center)
                if distance < min_distance and distance <= bowl.radius:
                    min_distance = distance
                    cat_at_bowl = bowl_name

            if cat_at_bowl:
                bowls_with_cats[cat_at_bowl].append(cat_name)

                # Update track
                if track.current_bowl != cat_at_bowl:
                    track.current_bowl = cat_at_bowl
                    track.at_bowl_since = current_time
                else:
                    # Check if cat has been at bowl long enough to be "eating"
                    if track.at_bowl_since:
                        eating_duration = current_time - track.at_bowl_since
                        if eating_duration >= self.eating_threshold:
                            # Check for violation
                            allowed_bowl = self.cat_allowed_bowls.get(cat_name)
                            if allowed_bowl and cat_at_bowl != allowed_bowl:
                                # This cat is at the wrong bowl!
                                bowl = self.bowls[cat_at_bowl]
                                owner_present = self._is_owner_at_bowl(cat_at_bowl)

                                violation = Violation(
                                    timestamp=current_time,
                                    cat_name=cat_name,
                                    bowl_name=cat_at_bowl,
                                    owner_present=owner_present,
                                )
                                new_violations.append(violation)
                                self.violations.append(violation)
            else:
                # Cat not at any bowl
                if track.current_bowl:
                    # Cat just left a bowl
                    track.current_bowl = None
                    track.at_bowl_since = None

        # Update bowl states
        for bowl_name, bowl in self.bowls.items():
            cats_at_bowl = bowls_with_cats[bowl_name]

            if cats_at_bowl:
                bowl.state = BowlState.OCCUPIED
                bowl.current_cat = cats_at_bowl[0]  # Primary cat
                bowl.last_occupied = current_time

                # Track if owner is eating
                if bowl_name in cats_at_bowl:
                    if bowl.owner_eating_since is None:
                        bowl.owner_eating_since = current_time
            else:
                # No cats at this bowl
                if bowl.state == BowlState.OCCUPIED:
                    bowl.state = BowlState.RECENTLY_LEFT
                    bowl.current_cat = None

                if bowl.state == BowlState.RECENTLY_LEFT:
                    if bowl.last_occupied:
                        time_since_left = current_time - bowl.last_occupied
                        if time_since_left >= self.leave_threshold:
                            bowl.state = BowlState.EMPTY
                            bowl.owner_eating_since = None

        return new_violations

    def _is_owner_at_bowl(self, bowl_name: str) -> bool:
        """Check if the owner of a bowl is currently at their bowl."""
        # Find which cat owns this bowl
        owner_cat = None
        for cat_name, allowed_bowl in self.cat_allowed_bowls.items():
            if allowed_bowl == bowl_name:
                owner_cat = cat_name
                break

        if owner_cat is None:
            return False

        # Check if owner has an active track at their bowl
        if owner_cat in self.cat_tracks:
            track = self.cat_tracks[owner_cat]
            if track.current_bowl == bowl_name:
                return True

        return False

    def should_spray(self, violation: Violation) -> bool:
        """
        Determine if a violation should trigger a spray.

        Only spray Ilana when she's at Abbi's bowl.

        Args:
            violation: The violation to evaluate

        Returns:
            True if spray should be triggered
        """
        # Only spray Ilana for eating from Abbi's bowl
        if violation.cat_name == "ilana" and violation.bowl_name == "abbi":
            return True

        return False

    def get_status(self) -> dict:
        """Get current monitoring status for logging/display."""
        return {
            "bowls": {
                name: {
                    "state": bowl.state.value,
                    "current_cat": bowl.current_cat,
                }
                for name, bowl in self.bowls.items()
            },
            "cat_tracks": {
                name: {
                    "at_bowl": track.current_bowl,
                    "position": track.position,
                }
                for name, track in self.cat_tracks.items()
            },
            "total_violations": len(self.violations),
        }

    def reset(self) -> None:
        """Reset all tracking state."""
        for bowl in self.bowls.values():
            bowl.state = BowlState.EMPTY
            bowl.current_cat = None
            bowl.owner_eating_since = None
            bowl.last_occupied = None

        self.cat_tracks.clear()
        self.violations.clear()
