"""
Sound player module for deterrent sounds.

Plays audio files as a deterrent, mirroring the Sprayer pattern.
"""

import threading
import time
from pathlib import Path
from typing import Optional

# Try to import pygame for audio playback
try:
    import pygame.mixer

    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


class SoundPlayer:
    """
    Plays deterrent sounds with cooldown management.
    """

    def __init__(
        self,
        sound_file: str = "sounds/deterrent.mp3",
        volume: float = 0.7,
        duration_ms: int = 2000,
        cooldown_s: float = 5.0,
        enabled: bool = True,
        simulate: bool = False,
    ):
        """
        Initialize the sound player.

        Args:
            sound_file: Path to the sound file (MP3 or WAV)
            volume: Volume level (0.0 to 1.0)
            duration_ms: Maximum playback duration in milliseconds
            cooldown_s: Minimum time between plays
            enabled: Whether sound playback is enabled
            simulate: If True, only simulate without actual playback
        """
        self.sound_file = Path(sound_file)
        self.volume = max(0.0, min(1.0, volume))
        self.duration_ms = duration_ms
        self.cooldown_s = cooldown_s
        self.enabled = enabled
        self.simulate = simulate or not PYGAME_AVAILABLE

        self._last_play_time: Optional[float] = None
        self._play_lock = threading.Lock()
        self._is_playing = False
        self._initialized = False

        # Sound statistics
        self.play_count = 0
        self.total_play_duration_ms = 0

    def initialize(self) -> bool:
        """
        Initialize the audio system.

        Returns:
            True if initialization succeeded, False otherwise
        """
        if self._initialized:
            return True

        if self.simulate:
            self._initialized = True
            return True

        # Check if sound file exists
        if not self.sound_file.exists():
            print(f"[WARNING] Sound file not found: {self.sound_file}")
            self.simulate = True
            self._initialized = True
            return True

        try:
            pygame.mixer.init()
            pygame.mixer.music.set_volume(self.volume)
            self._initialized = True
            return True
        except Exception as e:
            print(f"[WARNING] Failed to initialize audio: {e}")
            self.simulate = True
            self._initialized = True
            return True

    def cleanup(self) -> None:
        """Clean up audio resources."""
        if self._initialized and not self.simulate:
            try:
                pygame.mixer.music.stop()
                pygame.mixer.quit()
            except Exception:
                pass
        self._initialized = False

    def can_play(self) -> bool:
        """Check if playing is currently allowed (not in cooldown)."""
        if not self.enabled:
            return False

        if self._is_playing:
            return False

        if self._last_play_time is not None:
            elapsed = time.time() - self._last_play_time
            if elapsed < self.cooldown_s:
                return False

        return True

    def play(self, duration_ms: Optional[int] = None) -> bool:
        """
        Play the deterrent sound.

        Args:
            duration_ms: Playback duration (uses default if not specified)

        Returns:
            True if sound was triggered, False if blocked by cooldown/disabled
        """
        if not self._initialized:
            self.initialize()

        with self._play_lock:
            if not self.can_play():
                return False

            self._is_playing = True
            self._last_play_time = time.time()

        duration = duration_ms or self.duration_ms

        # Run playback in a separate thread to not block
        play_thread = threading.Thread(
            target=self._do_play,
            args=(duration,),
            daemon=True,
        )
        play_thread.start()

        return True

    def _do_play(self, duration_ms: int) -> None:
        """Actually perform the playback (runs in separate thread)."""
        try:
            if self.simulate:
                print(f"[SIMULATE] Playing sound for {duration_ms}ms")
            else:
                try:
                    pygame.mixer.music.load(str(self.sound_file))
                    pygame.mixer.music.play()
                except Exception as e:
                    print(f"[ERROR] Failed to play sound: {e}")

            # Wait for playback duration
            time.sleep(duration_ms / 1000.0)

            if not self.simulate:
                try:
                    pygame.mixer.music.stop()
                except Exception:
                    pass

            # Update statistics
            self.play_count += 1
            self.total_play_duration_ms += duration_ms

        finally:
            with self._play_lock:
                self._is_playing = False

    def play_sync(self, duration_ms: Optional[int] = None) -> bool:
        """
        Play the sound synchronously (blocking).

        Args:
            duration_ms: Playback duration (uses default if not specified)

        Returns:
            True if sound was triggered, False if blocked
        """
        if not self._initialized:
            self.initialize()

        if not self.can_play():
            return False

        with self._play_lock:
            self._is_playing = True
            self._last_play_time = time.time()

        duration = duration_ms or self.duration_ms

        try:
            if self.simulate:
                print(f"[SIMULATE] Playing sound for {duration}ms")
            else:
                try:
                    pygame.mixer.music.load(str(self.sound_file))
                    pygame.mixer.music.play()
                except Exception as e:
                    print(f"[ERROR] Failed to play sound: {e}")

            time.sleep(duration / 1000.0)

            if not self.simulate:
                try:
                    pygame.mixer.music.stop()
                except Exception:
                    pass

            self.play_count += 1
            self.total_play_duration_ms += duration

            return True

        finally:
            with self._play_lock:
                self._is_playing = False

    def test_play(self, duration_ms: int = 500) -> bool:
        """
        Test play with a short duration, bypassing cooldown.

        Args:
            duration_ms: Test play duration

        Returns:
            True if test play was triggered
        """
        if not self._initialized:
            self.initialize()

        if self._is_playing:
            return False

        # Bypass cooldown for testing
        original_cooldown = self.cooldown_s
        self.cooldown_s = 0

        try:
            return self.play_sync(duration_ms)
        finally:
            self.cooldown_s = original_cooldown

    def set_volume(self, volume: float) -> None:
        """Set the playback volume."""
        self.volume = max(0.0, min(1.0, volume))
        if self._initialized and not self.simulate:
            try:
                pygame.mixer.music.set_volume(self.volume)
            except Exception:
                pass

    @property
    def cooldown_remaining(self) -> float:
        """Get remaining cooldown time in seconds."""
        if self._last_play_time is None:
            return 0.0

        elapsed = time.time() - self._last_play_time
        remaining = self.cooldown_s - elapsed

        return max(0.0, remaining)

    def get_stats(self) -> dict:
        """Get sound player statistics."""
        return {
            "enabled": self.enabled,
            "play_count": self.play_count,
            "total_play_duration_ms": self.total_play_duration_ms,
            "is_playing": self._is_playing,
            "cooldown_remaining": self.cooldown_remaining,
            "simulate_mode": self.simulate,
            "volume": self.volume,
            "sound_file": str(self.sound_file),
        }

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False
