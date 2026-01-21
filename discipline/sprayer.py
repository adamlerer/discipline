"""
Water sprayer control module.

Controls the solenoid valve via GPIO to spray water.
"""

import threading
import time
from typing import Optional

# Try to import RPi.GPIO (only available on Raspberry Pi)
try:
    import RPi.GPIO as GPIO

    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False


class Sprayer:
    """
    Controls the water sprayer via a relay connected to GPIO.
    """

    def __init__(
        self,
        gpio_pin: int = 17,
        duration_ms: int = 500,
        cooldown_s: float = 10.0,
        enabled: bool = True,
        simulate: bool = False,
    ):
        """
        Initialize the sprayer controller.

        Args:
            gpio_pin: BCM GPIO pin number connected to relay
            duration_ms: Default spray duration in milliseconds
            cooldown_s: Minimum time between sprays
            enabled: Whether spraying is enabled (False for test mode)
            simulate: If True, only simulate without GPIO (for testing)
        """
        self.gpio_pin = gpio_pin
        self.duration_ms = duration_ms
        self.cooldown_s = cooldown_s
        self.enabled = enabled
        self.simulate = simulate or not GPIO_AVAILABLE

        self._last_spray_time: Optional[float] = None
        self._spray_lock = threading.Lock()
        self._is_spraying = False
        self._initialized = False

        # Spray statistics
        self.spray_count = 0
        self.total_spray_duration_ms = 0

    def initialize(self) -> None:
        """Initialize GPIO for sprayer control."""
        if self._initialized:
            return

        if not self.simulate:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.gpio_pin, GPIO.OUT)
            GPIO.output(self.gpio_pin, GPIO.LOW)  # Ensure relay is off

        self._initialized = True

    def cleanup(self) -> None:
        """Clean up GPIO resources."""
        if self._initialized and not self.simulate:
            GPIO.output(self.gpio_pin, GPIO.LOW)
            GPIO.cleanup(self.gpio_pin)
        self._initialized = False

    def can_spray(self) -> bool:
        """Check if spraying is currently allowed (not in cooldown)."""
        if not self.enabled:
            return False

        if self._is_spraying:
            return False

        if self._last_spray_time is not None:
            elapsed = time.time() - self._last_spray_time
            if elapsed < self.cooldown_s:
                return False

        return True

    def spray(self, duration_ms: Optional[int] = None) -> bool:
        """
        Activate the sprayer for the specified duration.

        Args:
            duration_ms: Spray duration (uses default if not specified)

        Returns:
            True if spray was triggered, False if blocked by cooldown/disabled
        """
        if not self._initialized:
            self.initialize()

        with self._spray_lock:
            if not self.can_spray():
                return False

            self._is_spraying = True
            self._last_spray_time = time.time()

        duration = duration_ms or self.duration_ms

        # Run spray in a separate thread to not block
        spray_thread = threading.Thread(
            target=self._do_spray,
            args=(duration,),
            daemon=True,
        )
        spray_thread.start()

        return True

    def _do_spray(self, duration_ms: int) -> None:
        """Actually perform the spray (runs in separate thread)."""
        try:
            if self.simulate:
                print(f"[SIMULATE] Spraying for {duration_ms}ms")
            else:
                # Activate relay (turn on solenoid)
                GPIO.output(self.gpio_pin, GPIO.HIGH)

            # Wait for spray duration
            time.sleep(duration_ms / 1000.0)

            if not self.simulate:
                # Deactivate relay (turn off solenoid)
                GPIO.output(self.gpio_pin, GPIO.LOW)

            # Update statistics
            self.spray_count += 1
            self.total_spray_duration_ms += duration_ms

        finally:
            with self._spray_lock:
                self._is_spraying = False

    def spray_sync(self, duration_ms: Optional[int] = None) -> bool:
        """
        Activate the sprayer synchronously (blocking).

        Args:
            duration_ms: Spray duration (uses default if not specified)

        Returns:
            True if spray was triggered, False if blocked
        """
        if not self._initialized:
            self.initialize()

        if not self.can_spray():
            return False

        with self._spray_lock:
            self._is_spraying = True
            self._last_spray_time = time.time()

        duration = duration_ms or self.duration_ms

        try:
            if self.simulate:
                print(f"[SIMULATE] Spraying for {duration}ms")
            else:
                GPIO.output(self.gpio_pin, GPIO.HIGH)

            time.sleep(duration / 1000.0)

            if not self.simulate:
                GPIO.output(self.gpio_pin, GPIO.LOW)

            self.spray_count += 1
            self.total_spray_duration_ms += duration

            return True

        finally:
            with self._spray_lock:
                self._is_spraying = False

    def test_spray(self, duration_ms: int = 200) -> bool:
        """
        Test spray with a short duration, bypassing cooldown.

        Args:
            duration_ms: Test spray duration

        Returns:
            True if test spray was triggered
        """
        if not self._initialized:
            self.initialize()

        if self._is_spraying:
            return False

        # Bypass cooldown for testing
        original_cooldown = self.cooldown_s
        self.cooldown_s = 0

        try:
            return self.spray_sync(duration_ms)
        finally:
            self.cooldown_s = original_cooldown

    @property
    def cooldown_remaining(self) -> float:
        """Get remaining cooldown time in seconds."""
        if self._last_spray_time is None:
            return 0.0

        elapsed = time.time() - self._last_spray_time
        remaining = self.cooldown_s - elapsed

        return max(0.0, remaining)

    def get_stats(self) -> dict:
        """Get sprayer statistics."""
        return {
            "enabled": self.enabled,
            "spray_count": self.spray_count,
            "total_spray_duration_ms": self.total_spray_duration_ms,
            "is_spraying": self._is_spraying,
            "cooldown_remaining": self.cooldown_remaining,
            "simulate_mode": self.simulate,
        }

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False
