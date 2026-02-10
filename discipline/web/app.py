"""
Flask web application for the Discipline admin interface.

Provides live video streaming, status monitoring, and deterrent controls.
"""

import threading
import time
from typing import TYPE_CHECKING, Generator, Optional

import cv2
import os
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

if TYPE_CHECKING:
    from ..main import DisciplineSystem


class DisciplineWebApp:
    """
    Web interface for the Discipline cat monitoring system.
    """

    def __init__(
        self,
        system: "DisciplineSystem",
        host: str = "0.0.0.0",
        port: int = 5000,
    ):
        """
        Initialize the web application.

        Args:
            system: Reference to the main DisciplineSystem
            host: Host to bind to
            port: Port to listen on
        """
        self.system = system
        self.host = host
        self.port = port

        self._app = Flask(
            __name__,
            template_folder="templates",
            static_folder="static",
        )
        self._setup_routes()

        self._server_thread: Optional[threading.Thread] = None
        self._running = False

    def _setup_routes(self) -> None:
        """Set up Flask routes."""

        @self._app.route("/")
        def index():
            """Serve the admin dashboard."""
            return render_template("dashboard.html")

        @self._app.route("/video_feed")
        def video_feed():
            """MJPEG video stream with overlays."""
            return Response(
                self._generate_frames(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        @self._app.route("/api/status")
        def api_status():
            """Get system status."""
            return jsonify(self.system.get_system_status())

        @self._app.route("/api/spray/toggle", methods=["POST"])
        def api_spray_toggle():
            """Toggle spray on/off."""
            data = request.get_json() or {}
            enabled = data.get("enabled")

            if enabled is None:
                # Toggle current state
                enabled = not self.system.get_spray_enabled()

            self.system.set_spray_enabled(enabled)
            return jsonify({
                "success": True,
                "spray_enabled": enabled,
            })

        @self._app.route("/api/sound/toggle", methods=["POST"])
        def api_sound_toggle():
            """Toggle sound on/off."""
            data = request.get_json() or {}
            enabled = data.get("enabled")

            if enabled is None:
                # Toggle current state
                enabled = not self.system.get_sound_enabled()

            self.system.set_sound_enabled(enabled)
            return jsonify({
                "success": True,
                "sound_enabled": enabled,
            })

        @self._app.route("/api/test/spray", methods=["POST"])
        def api_test_spray():
            """Test spray (bypasses cooldown)."""
            if self.system.sprayer:
                success = self.system.sprayer.test_spray()
                return jsonify({
                    "success": success,
                    "message": "Spray test triggered" if success else "Spray busy",
                })
            return jsonify({
                "success": False,
                "message": "Sprayer not available",
            })

        @self._app.route("/api/test/sound", methods=["POST"])
        def api_test_sound():
            """Test sound (bypasses cooldown)."""
            if self.system.sound_player:
                success = self.system.sound_player.test_play()
                return jsonify({
                    "success": success,
                    "message": "Sound test triggered" if success else "Sound busy",
                })
            return jsonify({
                "success": False,
                "message": "Sound player not available",
            })

        @self._app.route("/api/events")
        def api_events():
            """Get recent events."""
            limit = request.args.get("limit", 50, type=int)
            events = self.system.get_recent_events(limit)
            return jsonify(events)

        # Labeling API endpoints
        @self._app.route("/api/labeling/images")
        def api_labeling_images():
            """Get list of unlabeled images."""
            images = self.system.get_unlabeled_images()
            return jsonify(images)

        @self._app.route("/api/labeling/image/<filename>")
        def api_labeling_image(filename: str):
            """Serve an unlabeled image file."""
            filepath = self.system.get_unlabeled_image_path(filename)
            if filepath is None:
                return jsonify({"error": "Image not found"}), 404
            return send_file(filepath, mimetype="image/jpeg")

        @self._app.route("/api/labeling/label", methods=["POST"])
        def api_labeling_label():
            """Label an image and move to training folder."""
            data = request.get_json() or {}
            filename = data.get("filename")
            cat_name = data.get("cat_name")

            if not filename or not cat_name:
                return jsonify({
                    "success": False,
                    "error": "Missing filename or cat_name",
                }), 400

            if cat_name not in ("abbi", "ilana"):
                return jsonify({
                    "success": False,
                    "error": "Invalid cat_name (must be 'abbi' or 'ilana')",
                }), 400

            success = self.system.label_image(filename, cat_name)
            return jsonify({
                "success": success,
                "message": f"Labeled as {cat_name}" if success else "Image not found",
            })

        @self._app.route("/api/labeling/skip", methods=["POST"])
        def api_labeling_skip():
            """Skip (delete) an unlabeled image."""
            data = request.get_json() or {}
            filename = data.get("filename")

            if not filename:
                return jsonify({
                    "success": False,
                    "error": "Missing filename",
                }), 400

            success = self.system.skip_image(filename)
            return jsonify({
                "success": success,
                "message": "Image skipped" if success else "Image not found",
            })

        @self._app.route("/api/labeling/stats")
        def api_labeling_stats():
            """Get labeling statistics."""
            stats = self.system.get_labeling_stats()
            return jsonify(stats)

        @self._app.route("/api/labeling/labeled/<cat_name>")
        def api_labeled_images(cat_name: str):
            """Get list of labeled images for a cat."""
            if cat_name not in ("abbi", "ilana"):
                return jsonify({"error": "Invalid cat name"}), 400
            images = self.system.get_labeled_images(cat_name)
            return jsonify(images)

        @self._app.route("/api/labeling/labeled/<cat_name>/<filename>")
        def api_labeled_image(cat_name: str, filename: str):
            """Serve a labeled image file."""
            filepath = self.system.get_labeled_image_path(cat_name, filename)
            if filepath is None:
                return jsonify({"error": "Image not found"}), 404
            return send_file(filepath, mimetype="image/jpeg")

        @self._app.route("/api/labeling/swap", methods=["POST"])
        def api_swap_label():
            """Swap an image's label to the other cat."""
            data = request.get_json() or {}
            cat_name = data.get("cat")
            filename = data.get("filename")

            if not cat_name or not filename:
                return jsonify({
                    "success": False,
                    "error": "Missing cat or filename",
                }), 400

            success = self.system.swap_image_label(cat_name, filename)
            new_cat = "ilana" if cat_name == "abbi" else "abbi"

            return jsonify({
                "success": success,
                "message": f"Moved to {new_cat}" if success else "Failed to swap",
                "new_cat": new_cat if success else None,
            })

        @self._app.route("/api/labeling/delete", methods=["POST"])
        def api_delete_labeled():
            """Delete multiple labeled images."""
            data = request.get_json() or {}
            images = data.get("images", [])

            if not images:
                return jsonify({
                    "success": False,
                    "error": "No images specified",
                }), 400

            result = self.system.delete_labeled_images(images)
            return jsonify({
                "success": True,
                "deleted": result["deleted"],
                "errors": result["errors"],
            })

        # Bowl configuration endpoints
        @self._app.route("/api/bowls")
        def api_bowls():
            """Get bowl positions."""
            bowls = self.system.get_bowl_positions()
            return jsonify(bowls)

        @self._app.route("/api/sound/upload", methods=["POST"])
        def api_sound_upload():
            """Upload a custom sound file."""
            if 'file' not in request.files:
                return jsonify({
                    "success": False,
                    "error": "No file provided",
                }), 400

            file = request.files['file']
            if file.filename == '':
                return jsonify({
                    "success": False,
                    "error": "No file selected",
                }), 400

            # Check file extension
            allowed_extensions = {'.wav', '.mp3', '.ogg'}
            ext = os.path.splitext(file.filename)[1].lower()
            if ext not in allowed_extensions:
                return jsonify({
                    "success": False,
                    "error": f"Invalid file type. Allowed: {', '.join(allowed_extensions)}",
                }), 400

            # Save to sounds directory
            sounds_dir = Path(self.system.config_path).parent / "sounds"
            sounds_dir.mkdir(exist_ok=True)

            filename = secure_filename(file.filename)
            filepath = sounds_dir / filename
            file.save(filepath)

            # Update config to use new sound
            self.system.config["sound"]["file"] = f"sounds/{filename}"
            self.system._save_config()

            # Reinitialize sound player if exists
            if self.system.sound_player:
                self.system.sound_player.sound_file = filepath

            return jsonify({
                "success": True,
                "message": f"Uploaded {filename}",
                "filename": filename,
            })

        @self._app.route("/api/sound/file")
        def api_sound_file():
            """Serve the current sound file for preview."""
            sound_path = self.system.config.get("sound", {}).get("file")
            if not sound_path:
                return jsonify({"error": "No sound file configured"}), 404

            filepath = Path(self.system.config_path).resolve().parent / sound_path
            if not filepath.exists():
                return jsonify({"error": "Sound file not found"}), 404

            # Determine mimetype based on extension
            ext = filepath.suffix.lower()
            mimetypes = {
                ".wav": "audio/wav",
                ".mp3": "audio/mpeg",
                ".ogg": "audio/ogg",
            }
            mimetype = mimetypes.get(ext, "audio/wav")

            return send_file(filepath, mimetype=mimetype)

        @self._app.route("/api/bowls/<bowl_name>", methods=["POST"])
        def api_update_bowl(bowl_name: str):
            """Update a bowl's position."""
            data = request.get_json() or {}
            x = data.get("x")
            y = data.get("y")
            radius = data.get("radius")

            if x is None or y is None or radius is None:
                return jsonify({
                    "success": False,
                    "error": "Missing x, y, or radius",
                }), 400

            success = self.system.update_bowl_position(
                bowl_name,
                int(x),
                int(y),
                int(radius),
            )

            return jsonify({
                "success": success,
                "message": f"Updated {bowl_name} bowl" if success else "Invalid bowl name",
            })

    def _generate_frames(self) -> Generator[bytes, None, None]:
        """Generate MJPEG frames for video streaming."""
        while self._running and self.system._running:
            frame = self.system.get_annotated_frame()

            if frame is not None:
                # Encode frame as JPEG
                _, buffer = cv2.imencode(
                    ".jpg",
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 70],
                )
                frame_bytes = buffer.tobytes()

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )

            # Control frame rate
            time.sleep(0.033)  # ~30 FPS

    def start(self) -> None:
        """Start the web server in a background thread."""
        if self._running:
            return

        self._running = True
        self._server_thread = threading.Thread(
            target=self._run_server,
            daemon=True,
        )
        self._server_thread.start()

    def _run_server(self) -> None:
        """Run the Flask server (in background thread)."""
        # Disable Flask's reloader and debugger for production
        self._app.run(
            host=self.host,
            port=self.port,
            debug=False,
            use_reloader=False,
            threaded=True,
        )

    def stop(self) -> None:
        """Stop the web server."""
        self._running = False
        # Note: Flask's development server doesn't have a clean shutdown
        # In production, use a proper WSGI server like gunicorn
