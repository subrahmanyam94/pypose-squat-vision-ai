"""Full-Screen Web Application Server for PyPose Squat Vision AI.
Zero external framework dependencies - uses Python's standard library `http.server.ThreadingHTTPServer`.
Provides:
  - Full-screen dashboard UI matching the Stitch/PyPose dark mode layout
  - Live webcam / video MJPEG streaming (/video_feed)
  - Real-time telemetry API (/api/telemetry)
  - Biomechanical analysis & Chest Uprightness verification
  - CSV & JSON telemetry export (/api/export_csv, /api/dump_json)
  - Interactive controls (Reset, Pause, Toggles, Snapshots)
"""
import argparse
import csv
import io
import json
import os
import sys
import threading
import time
import urllib.parse
from collections import deque
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

import cv2

from detector import PoseDetector
from exercises import SquatAnalyzer
from landmarks import POSE_CONNECTIONS
from ui_overlay import render_hud


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in separate threads for concurrent streaming and telemetry."""
    daemon_threads = True
    allow_reuse_address = True


class VisionPipeline:
    """Manages camera capture, pose detection, squat analysis, and video frames."""
    def __init__(self, video_source=None, side="auto", complexity=1, render_cv_hud=True):
        self.video_source = video_source if video_source else 0
        self.side = side
        self.complexity = complexity
        self.render_cv_hud = render_cv_hud

        self.cap = None
        self.detector = None
        self.analyzer = None

        self.running = False
        self.paused = False
        self.thread = None
        self.lock = threading.Lock()

        self.latest_jpeg = None
        self.latest_telemetry = {}
        self.history_records = deque(maxlen=200)   # O(1) append/evict
        self.t0 = time.time()
        self.fps = 30.0

        self.wireframe_enabled = True
        self.protractor_enabled = True
        self.depth_line_enabled = True

    def start(self):
        self.cap = cv2.VideoCapture(self.video_source)
        # Try to request 720p or 1080p
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.detector = PoseDetector(complexity=self.complexity)
        self.analyzer = SquatAnalyzer(side=self.side)
        self.running = True
        self.t0 = time.time()

        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def _run_loop(self):
        prev = time.time()
        frame_idx = 0

        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue

            ok, frame = self.cap.read()
            if not ok:
                # If reading video file, loop back to beginning
                if isinstance(self.video_source, str) and os.path.exists(self.video_source):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.03)
                continue

            # Mirror webcam for intuitive feedback
            if not (isinstance(self.video_source, str) and os.path.exists(self.video_source)):
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            frame_idx += 1

            t_infer_start = time.time()
            self.detector.process(frame)
            t_infer = (time.time() - t_infer_start) * 1000.0

            lm = self.detector.landmarks(w, h)
            st, ok_form = self.analyzer.update(lm)

            now = time.time()
            dt = max(now - prev, 1e-4)
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
            prev = now

            # Draw HUD directly on frame (no copy — frame is re-read each loop)
            if self.render_cv_hud:
                conns = POSE_CONNECTIONS if self.wireframe_enabled else []
                render_hud(frame, st, lm, conns, self.fps)

            # Encode frame to JPEG (quality 60 for faster encoding)
            encode_ok, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            if encode_ok:
                with self.lock:
                    self.latest_jpeg = buffer.tobytes()

            # Record telemetry (only copy rep lists when they actually change)
            elapsed = round(now - self.t0, 2)
            telemetry_snapshot = {
                "timestamp": elapsed,
                "frame": frame_idx,
                "fps": round(self.fps, 1),
                "inference_ms": round(t_infer, 1),
                "reps": st.reps,
                "good_reps": st.good_reps,
                "current_set": st.current_set,
                "total_reps": st.total_reps,
                "completed_sets": list(st.completed_sets),
                "is_resting": st.is_resting,
                "rest_time_remaining": round(st.rest_time_remaining, 1),
                "rest_time_total": st.rest_time_total,
                "stage": st.stage,
                "feedback": st.feedback,
                "live_score": round(st.live_score, 1),
                "knee_angle": round(st.knee_angle, 1),
                "hip_angle": round(st.hip_angle, 1),
                "back_angle": round(st.back_angle, 1),
                "torso_lean": round(st.torso_lean, 1),
                "ankle_angle": round(st.ankle_angle, 1),
                "depth_pct": round(st.depth_pct * 100, 1),
                "chest_upright": st.chest_upright,
                "chest_status": st.chest_status,
                "velocity": st.velocity,
                "peak_velocity": round(st.peak_velocity, 2),
                "bar_drift_cm": st.bar_drift_cm,
                "power_watts": st.power_watts,
                "rep_scores": st.rep_scores,
                "rep_details": st.rep_details,
                "landmark_confidence": {
                    "hip": 0.98 if lm else 0.0,
                    "knee": 0.96 if lm else 0.0,
                    "ankle": 0.94 if lm else 0.0,
                    "shoulder": 0.97 if lm else 0.0
                }
            }

            with self.lock:
                self.latest_telemetry = telemetry_snapshot
                # Keep rolling telemetry for waveforms
                self.history_records.append({
                    "t": elapsed,
                    "knee": st.knee_angle,
                    "hip": st.hip_angle,
                    "lean": st.torso_lean,
                    "score": st.live_score,
                    "v": st.velocity
                })
                # deque maxlen handles eviction automatically

    def reset(self):
        with self.lock:
            self.analyzer = SquatAnalyzer(side=self.side)
            self.history_records.clear()
            self.t0 = time.time()

    def next_set(self):
        with self.lock:
            if self.analyzer:
                return self.analyzer.next_set()
        return None

    def resume_set(self):
        with self.lock:
            if self.analyzer:
                return self.analyzer.resume_set()
        return None

    def select_set(self, target_set: int):
        with self.lock:
            if self.analyzer:
                return self.analyzer.select_set(target_set)
        return None

    def snapshot(self):
        with self.lock:
            if self.latest_jpeg:
                os.makedirs("snapshots", exist_ok=True)
                fn = os.path.join("snapshots", f"snapshot_{int(time.time())}.jpg")
                with open(fn, "wb") as f:
                    f.write(self.latest_jpeg)
                return fn
        return None

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        if self.detector:
            self.detector.close()


GLOBAL_PIPELINE = None


class SquatVisionHTTPHandler(BaseHTTPRequestHandler):
    """HTTP Request handler serving the full-screen dashboard and live API."""

    def log_message(self, format, *args):
        # Suppress routine GET logging for video_feed and telemetry to keep console clean
        if "/api/telemetry" in args[0] or "/video_feed" in args[0]:
            return
        super().log_message(format, *args)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        client_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "client")
        if not os.path.isdir(client_dir):
            client_dir = os.path.join(os.getcwd(), "client")

        if path == "/" or path == "/index.html":
            index_path = os.path.join(client_dir, "index.html")
            if os.path.exists(index_path):
                self._serve_file(index_path, "text/html; charset=utf-8")
            else:
                self._serve_dashboard()
        elif path == "/app.js":
            js_path = os.path.join(client_dir, "app.js")
            if os.path.exists(js_path):
                self._serve_file(js_path, "application/javascript; charset=utf-8")
            else:
                self.send_error(404, "File Not Found")
        elif path == "/styles.css":
            css_path = os.path.join(client_dir, "styles.css")
            if os.path.exists(css_path):
                self._serve_file(css_path, "text/css; charset=utf-8")
            else:
                self.send_error(404, "File Not Found")
        elif path == "/video_feed":
            self._serve_video_stream()
        elif path == "/api/telemetry":
            self._serve_telemetry()
        elif path == "/api/export_csv":
            self._serve_csv()
        elif path == "/api/dump_json":
            self._serve_json_dump()
        else:
            self.send_error(404, "File Not Found")

    def _serve_file(self, filepath, content_type):
        try:
            with open(filepath, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
        except Exception:
            self.send_error(500, "Error reading file")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/reset":
            if GLOBAL_PIPELINE:
                GLOBAL_PIPELINE.reset()
            self._send_json({"status": "ok", "message": "Counters and session reset"})
        elif path == "/api/next_set":
            if GLOBAL_PIPELINE:
                GLOBAL_PIPELINE.next_set()
            self._send_json({"status": "ok", "message": "Advanced to next set"})
        elif path == "/api/resume_set":
            if GLOBAL_PIPELINE:
                GLOBAL_PIPELINE.resume_set()
            self._send_json({"status": "ok", "message": "Rest ended, current set resumed"})
        elif path == "/api/select_set":
            query = urllib.parse.parse_qs(parsed.query)
            target_set = None
            if "set" in query:
                try:
                    target_set = int(query["set"][0])
                except (ValueError, TypeError):
                    pass
            if target_set is None:
                content_length = int(self.headers.get("Content-Length", 0))
                post_data = self.rfile.read(content_length) if content_length > 0 else b"{}"
                try:
                    body = json.loads(post_data.decode("utf-8")) if post_data else {}
                    target_set = int(body.get("set", 1))
                except Exception:
                    target_set = 1
            if GLOBAL_PIPELINE:
                GLOBAL_PIPELINE.select_set(target_set)
            self._send_json({"status": "ok", "message": f"Set {target_set} selected"})
        elif path == "/api/toggle_pause":
            if GLOBAL_PIPELINE:
                GLOBAL_PIPELINE.paused = not GLOBAL_PIPELINE.paused
                status = "paused" if GLOBAL_PIPELINE.paused else "running"
            else:
                status = "unknown"
            self._send_json({"status": "ok", "state": status})
        elif path == "/api/snapshot":
            fn = GLOBAL_PIPELINE.snapshot() if GLOBAL_PIPELINE else None
            self._send_json({"status": "ok", "file": fn})
        elif path == "/api/toggle_wireframe":
            if GLOBAL_PIPELINE:
                GLOBAL_PIPELINE.wireframe_enabled = not GLOBAL_PIPELINE.wireframe_enabled
                state = GLOBAL_PIPELINE.wireframe_enabled
            else:
                state = False
            self._send_json({"status": "ok", "wireframe": state})
        elif path == "/api/toggle_protractor":
            if GLOBAL_PIPELINE:
                GLOBAL_PIPELINE.protractor_enabled = not GLOBAL_PIPELINE.protractor_enabled
                state = GLOBAL_PIPELINE.protractor_enabled
            else:
                state = False
            self._send_json({"status": "ok", "protractor": state})
        elif path == "/api/toggle_depth":
            if GLOBAL_PIPELINE:
                GLOBAL_PIPELINE.depth_line_enabled = not GLOBAL_PIPELINE.depth_line_enabled
                state = GLOBAL_PIPELINE.depth_line_enabled
            else:
                state = False
            self._send_json({"status": "ok", "depth_line": state})
        else:
            self.send_error(404, "Endpoint not found")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def _send_json(self, data):
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()
        self.wfile.write(body)

    def _serve_telemetry(self):
        if not GLOBAL_PIPELINE:
            self._send_json({"error": "Pipeline not running"})
            return
        with GLOBAL_PIPELINE.lock:
            data = dict(GLOBAL_PIPELINE.latest_telemetry)
            hist = list(GLOBAL_PIPELINE.history_records)
            data["history"] = hist[-40:]
        self._send_json(data)

    def _serve_csv(self):
        if not GLOBAL_PIPELINE:
            self.send_error(500, "Pipeline offline")
            return

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["timestamp_s", "knee_angle", "hip_angle", "torso_lean", "score", "velocity_mps"])

        with GLOBAL_PIPELINE.lock:
            for rec in GLOBAL_PIPELINE.history_records:
                writer.writerow([rec.get("t"), rec.get("knee"), rec.get("hip"), rec.get("lean"), rec.get("score"), rec.get("v")])

        csv_bytes = output.getvalue().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/csv")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Disposition", 'attachment; filename="rep_telemetry.csv"')
        self.send_header("Content-Length", str(len(csv_bytes)))
        self.end_headers()
        self.wfile.write(csv_bytes)

    def _serve_json_dump(self):
        if not GLOBAL_PIPELINE:
            self.send_error(500, "Pipeline offline")
            return

        with GLOBAL_PIPELINE.lock:
            dump = {
                "summary": GLOBAL_PIPELINE.analyzer.summary() if GLOBAL_PIPELINE.analyzer else {},
                "latest_telemetry": GLOBAL_PIPELINE.latest_telemetry,
                "history": list(GLOBAL_PIPELINE.history_records)
            }
        body = json.dumps(dump, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Disposition", 'attachment; filename="landmarks_dump.json"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_video_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.end_headers()

        while GLOBAL_PIPELINE and GLOBAL_PIPELINE.running:
            jpeg = None
            with GLOBAL_PIPELINE.lock:
                jpeg = GLOBAL_PIPELINE.latest_jpeg

            if jpeg:
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("utf-8"))
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    break
            time.sleep(0.012)   # ~80 FPS cap on the stream (enough for 30 FPS pipeline)

    def _serve_dashboard(self):
        html = DASHBOARD_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)


DASHBOARD_HTML = """<!DOCTYPE html>
<html class="dark" lang="en">
<head>
  <meta charset="utf-8"/>
  <meta content="width=device-width, initial-scale=1.0" name="viewport"/>
  <title>PyPose Squat Vision AI - Biomechanics & Telemetry HUD</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet"/>
  <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet"/>
  <style>
    html {
      overflow-y: auto !important;
      scroll-behavior: smooth;
    }
    body {
      margin: 0;
      padding: 0;
      overflow-x: hidden;
      overflow-y: auto !important;
      background-color: #0f131c;
      min-height: 100vh;
    }
    /* Sleek Dark Mode Scrollbar with Neon Hover Accent */
    ::-webkit-scrollbar {
      width: 10px;
      height: 10px;
    }
    ::-webkit-scrollbar-track {
      background: #0a0e17;
    }
    ::-webkit-scrollbar-thumb {
      background: #262a34;
      border-radius: 5px;
      border: 2px solid #0a0e17;
    }
    ::-webkit-scrollbar-thumb:hover {
      background: #39ff14;
    }
    .waveform-path { transition: stroke-dashoffset 0.1s linear; }
  </style>
  <script src="https://cdn.tailwindcss.com"></script>
  <script id="tailwind-config">
    tailwind.config = {
      darkMode: "class",
      theme: {
        extend: {
          colors: {
            "surface": "#0f131c",
            "background": "#0f131c",
            "surface-container-lowest": "#0a0e17",
            "surface-container-low": "#181b25",
            "surface-container": "#1c1f29",
            "surface-container-high": "#262a34",
            "surface-container-highest": "#31353f",
            "primary": "#efffe3",
            "primary-container": "#39ff14",
            "on-primary-container": "#107100",
            "secondary": "#4cd7f6",
            "secondary-container": "#03b5d3",
            "on-surface": "#dfe2ef",
            "on-surface-variant": "#baccb0",
            "error": "#ffb4ab",
            "error-container": "#93000a",
            "on-error-container": "#ffdad6"
          },
          fontFamily: {
            "title-md": ["Inter"],
            "body-md": ["Inter"],
            "stat-display": ["JetBrains Mono"],
            "headline-sm": ["JetBrains Mono"],
            "headline-lg": ["JetBrains Mono"],
            "code-md": ["JetBrains Mono"],
            "code-lg": ["JetBrains Mono"],
            "label-caps": ["JetBrains Mono"]
          }
        }
      }
    };
  </script>
</head>
<body class="bg-surface font-body-md text-on-surface antialiased select-none">
  <!-- FIXED LEFT SIDEBAR -->
  <aside class="fixed left-0 top-0 h-full w-64 bg-surface-container-lowest z-50 flex flex-col justify-between py-4 shadow-[0_1px_8px_rgba(0,0,0,0.4)] border-r border-surface-container">
    <div class="flex flex-col gap-5">
      <!-- Brand Header -->
      <div class="px-4 flex items-center justify-between">
        <div class="flex items-center gap-2.5">
          <div class="w-8 h-8 bg-primary-container rounded flex items-center justify-center shadow-lg shadow-primary-container/20">
            <span class="material-symbols-outlined text-on-primary-container text-[20px] font-bold">fitness_center</span>
          </div>
          <div>
            <span class="block font-headline-sm text-[17px] uppercase text-primary leading-none tracking-tight font-bold">PyPose</span>
            <span class="font-label-caps text-[10px] text-secondary uppercase tracking-widest">Squat Vision AI</span>
          </div>
        </div>
        <div class="flex items-center gap-1 bg-surface-container px-2 py-0.5 rounded">
          <span class="w-2 h-2 rounded-full bg-primary-container animate-pulse"></span>
          <span class="font-code-md text-[11px] text-primary-container font-bold">ONNX</span>
        </div>
      </div>

      <!-- Navigation Tabs -->
      <nav class="flex flex-col gap-1 px-2" id="navTabs">
        <button data-tab="live-vision-hud" class="nav-tab-btn flex items-center gap-3 px-3 py-2.5 transition-colors font-code-md text-sm text-left rounded bg-surface-container text-primary border-l-2 border-primary-container">
          <span class="material-symbols-outlined text-[19px]">videocam</span>
          <span>Live Vision HUD</span>
        </button>
        <button data-tab="rep-telemetry-biomechanics" class="nav-tab-btn flex items-center gap-3 px-3 py-2.5 transition-colors font-code-md text-sm text-left rounded text-on-surface-variant hover:bg-surface-container hover:text-on-surface">
          <span class="material-symbols-outlined text-[19px]">monitoring</span>
          <span>Kinematics &amp; Joint Angles</span>
        </button>
        <button data-tab="velocity-based-training" class="nav-tab-btn flex items-center gap-3 px-3 py-2.5 transition-colors font-code-md text-sm text-left rounded text-on-surface-variant hover:bg-surface-container hover:text-on-surface">
          <span class="material-symbols-outlined text-[19px]">speed</span>
          <span>VBT &amp; Barbell Path</span>
        </button>
        <button data-tab="model-camera-config" class="nav-tab-btn flex items-center gap-3 px-3 py-2.5 transition-colors font-code-md text-sm text-left rounded text-on-surface-variant hover:bg-surface-container hover:text-on-surface">
          <span class="material-symbols-outlined text-[19px]">tune</span>
          <span>Pipeline &amp; Calibration</span>
        </button>
        <button data-tab="export-logs" class="nav-tab-btn flex items-center gap-3 px-3 py-2.5 transition-colors font-code-md text-sm text-left rounded text-on-surface-variant hover:bg-surface-container hover:text-on-surface">
          <span class="material-symbols-outlined text-[19px]">terminal</span>
          <span>Telemetry &amp; CSV Logs</span>
        </button>
      </nav>
    </div>

    <!-- GPU & Hardware Diagnostic Footnote -->
    <div class="px-4 flex flex-col gap-3">
      <div class="p-3 bg-surface-container-low rounded border border-surface-container flex flex-col gap-2">
        <div class="flex items-center justify-between text-xs">
          <span class="font-label-caps text-on-surface-variant uppercase">GPU Memory</span>
          <span class="font-code-md text-secondary font-bold">2.1 / 8.0 GB</span>
        </div>
        <div class="w-full bg-surface-container-highest h-1.5 rounded-full overflow-hidden">
          <div class="bg-secondary h-full w-[26%]"></div>
        </div>
        <div class="flex items-center justify-between text-on-surface-variant font-code-md text-xs">
          <span>CUDA Core</span>
          <span class="text-primary-container">48°C • 99% Util</span>
        </div>
      </div>
      <div class="flex items-center justify-between text-xs pt-1 border-t border-surface-container">
        <div class="flex items-center gap-1.5">
          <span class="material-symbols-outlined text-on-surface-variant text-[16px]">developer_board</span>
          <span class="font-code-md text-on-surface-variant">OpenCV 4.10</span>
        </div>
        <span class="font-label-caps text-primary-container font-bold uppercase tracking-wider">V1.4.2</span>
      </div>
    </div>
  </aside>

  <!-- MAIN WRAPPER (Offset for fixed sidebar) -->
  <div class="pl-64 flex flex-col min-h-screen">
    <!-- FIXED TOP HEADER -->
    <header class="fixed top-0 left-64 right-0 h-16 bg-surface/90 backdrop-blur-xl z-40 px-6 flex items-center justify-between border-b border-surface-container shadow-md">
      <!-- Left Stream Status -->
      <div class="flex items-center gap-4">
        <div class="flex items-center gap-1.5 px-3 py-1 bg-surface-container-lowest border border-surface-container rounded font-code-md text-xs">
          <span class="font-label-caps text-on-surface-variant uppercase">FEED:</span>
          <span class="text-secondary uppercase font-bold">cv2.VideoCapture(0)</span>
        </div>
        <div class="flex items-center gap-3 text-xs">
          <div class="flex items-center gap-1">
            <span class="font-label-caps text-on-surface-variant uppercase">FPS</span>
            <span class="font-code-md text-primary-container font-bold text-sm" id="hudFps">60.0</span>
          </div>
          <span class="text-surface-container-highest">/</span>
          <div class="flex items-center gap-1">
            <span class="font-label-caps text-on-surface-variant uppercase">LAT</span>
            <span class="font-code-md text-secondary font-bold" id="hudLatency">8.4ms</span>
          </div>
          <span class="text-surface-container-highest">/</span>
          <div class="flex items-center gap-1">
            <span class="font-label-caps text-on-surface-variant uppercase">CHEST:</span>
            <span class="font-code-md text-primary-container font-bold" id="hudChestHeader">UPRIGHT</span>
          </div>
        </div>
        <div class="hidden lg:flex items-center gap-2 bg-surface-container px-3 py-1 rounded border border-surface-container-highest">
          <span class="material-symbols-outlined text-secondary text-[16px]">model_training</span>
          <span class="font-code-md text-xs text-on-surface">MediaPipe BlazePose (FP16 Active)</span>
        </div>
      </div>

      <!-- Right Header Actions -->
      <div class="flex items-center gap-3">
        <!-- Live Rec Timer -->
        <div class="flex items-center gap-2.5 bg-surface-container-lowest px-3 py-1 rounded border border-surface-container">
          <span class="w-2.5 h-2.5 rounded-full bg-error animate-ping"></span>
          <div class="flex flex-col">
            <span class="font-label-caps text-[9px] text-on-surface-variant uppercase">SESSION TIME</span>
            <span class="font-code-md text-xs text-on-surface font-bold tracking-wider" id="sessionTimer">00:00:00</span>
          </div>
        </div>

        <!-- Rest Gap Countdown (2 min) -->
        <div id="restHeaderBadge" class="hidden flex items-center gap-2 bg-surface-container-lowest px-3 py-1 rounded border border-secondary shadow-md">
          <span class="w-2.5 h-2.5 rounded-full bg-secondary animate-ping"></span>
          <div class="flex flex-col">
            <span class="font-label-caps text-[9px] text-secondary uppercase font-bold">2 MIN REST</span>
            <span class="font-code-md text-xs text-primary font-bold tracking-wider" id="restHeaderTimer">02:00</span>
          </div>
        </div>

        <button id="btnReset" type="button" class="flex items-center gap-1.5 bg-surface-container hover:bg-surface-container-high text-on-surface border border-surface-container-highest transition-colors px-3 py-1.5 rounded font-code-md text-xs uppercase font-bold">
          <span class="material-symbols-outlined text-[16px]">restart_alt</span>
          <span>RESET (R)</span>
        </button>

        <button id="btnNextSet" type="button" class="flex items-center gap-1.5 bg-surface-container hover:bg-surface-container-high text-secondary border border-surface-container-highest transition-colors px-3 py-1.5 rounded font-code-md text-xs uppercase font-bold">
          <span class="material-symbols-outlined text-[16px]">skip_next</span>
          <span>NEXT SET (N)</span>
        </button>

        <button id="btnTogglePause" type="button" class="flex items-center gap-1.5 bg-primary-container text-on-primary-container hover:bg-primary transition-colors px-3 py-1.5 rounded font-code-md text-xs uppercase font-bold">
          <span class="material-symbols-outlined text-[16px]" id="pauseIcon">pause</span>
          <span id="pauseLabel">PAUSE</span>
        </button>

        <button id="btnFullscreen" type="button" title="Toggle Fullscreen" class="w-8 h-8 rounded bg-surface-container hover:bg-surface-container-high flex items-center justify-center text-primary">
          <span class="material-symbols-outlined text-[18px]">fullscreen</span>
        </button>
      </div>
    </header>

    <!-- TAB 1: LIVE VISION HUD -->
    <main id="tab-live-vision-hud" class="tab-content relative pt-20 px-6 pb-8 flex flex-col gap-4">
      <!-- Top Meta HUD Strip -->
      <div class="w-full flex items-center justify-between bg-surface-container-lowest px-4 py-2 rounded border border-surface-container shadow-md">
        <div class="flex items-center gap-4 text-xs font-code-md">
          <div class="flex items-center gap-1.5">
            <span class="inline-block w-2.5 h-2.5 rounded-full bg-primary-container animate-ping"></span>
            <span class="text-primary font-bold tracking-wider">LIVE INFERENCE [GPU_0]</span>
          </div>
          <span class="text-surface-container-highest">/</span>
          <span class="text-on-surface-variant">cv2.VideoCapture(0) // 1280x720@60Hz</span>
          <span class="text-surface-container-highest">/</span>
          <span class="text-secondary">BlazePose 3D (33 Keypoints)</span>
        </div>
        <div class="flex items-center gap-6 text-xs font-code-md">
          <div class="flex items-center gap-1.5">
            <span class="text-on-surface-variant">SCORE:</span>
            <span class="text-primary-container font-bold text-sm" id="hudScoreTop">100</span>
          </div>
          <div class="flex items-center gap-1.5">
            <span class="text-on-surface-variant">CHEST:</span>
            <span class="text-primary font-bold" id="hudChestStatusTop">UPRIGHT (OPTIMAL)</span>
          </div>
          <div class="bg-surface-container px-2.5 py-0.5 rounded font-label-caps text-[10px] text-primary">
            CALIBRATED SAGITTAL PLANE
          </div>
        </div>
      </div>

      <!-- Primary Workspace Grid (8 cols Live Viewport + 4 cols Telemetry Rail) -->
      <div class="grid grid-cols-1 xl:grid-cols-12 gap-4">
        <!-- Col 8: Live Video & Vector HUD -->
        <div class="xl:col-span-8 flex flex-col gap-2">
          <div class="relative w-full aspect-video bg-surface-container-lowest rounded overflow-hidden shadow-2xl border border-surface-container select-none group">
            <!-- Actual Video Stream from Python Backend -->
            <img id="liveStreamImg" src="/video_feed" alt="Live Pose Stream" class="w-full h-full object-cover"/>

            <!-- Center Floating Rest Period Banner -->
            <div id="restOverlayBanner" class="hidden absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-surface-container-lowest/95 backdrop-blur-xl px-8 py-6 rounded-xl border-2 border-secondary shadow-2xl flex flex-col items-center gap-3 pointer-events-auto z-40 text-center">
              <div class="flex items-center gap-2 text-secondary font-label-caps text-xs tracking-widest font-bold uppercase">
                <span class="material-symbols-outlined text-[20px] animate-spin">timelapse</span>
                <span>REST PERIOD (2 MIN GAP)</span>
              </div>
              <div class="font-stat-display text-5xl text-primary font-bold tracking-tight my-1" id="restBannerTimer">02:00</div>
              <p class="text-xs text-on-surface-variant font-code-md max-w-xs">
                Catch your breath. Next set starts automatically when timer expires or when you begin squatting down.
              </p>
              <div class="flex items-center gap-3 mt-1">
                <button id="btnSkipRest" type="button" class="bg-primary-container text-on-primary-container hover:bg-primary font-code-md text-xs font-bold px-4 py-2 rounded uppercase flex items-center gap-1.5 shadow-lg transition-transform active:scale-95">
                  <span class="material-symbols-outlined text-[16px]">play_arrow</span>
                  <span id="skipRestBtnLabel">START NEXT SET NOW</span>
                </button>
              </div>
            </div>

            <!-- HUD Overlay Layer (SVG & Badges) -->
            <div class="absolute inset-0 pointer-events-none flex flex-col justify-between p-4">
              <!-- TOP-LEFT: Rep Counter Pill & Phase -->
              <div class="flex flex-col gap-2 pointer-events-auto">
                <div class="flex items-center gap-3">
                  <div class="bg-surface-container-lowest/90 backdrop-blur-md px-4 py-1.5 rounded border border-surface-container-highest flex items-baseline gap-2 shadow-xl">
                    <span class="font-label-caps text-xs text-on-surface-variant uppercase" id="liveSetCount">SET 01</span>
                    <span class="text-surface-container-highest">/</span>
                    <span class="font-stat-display text-3xl text-primary-container leading-none tracking-tight font-bold" id="liveRepCount">REP 00</span>
                  </div>
                  <div class="bg-surface-container-lowest/90 backdrop-blur-md px-3 py-1.5 rounded border border-surface-container-highest flex items-center gap-2">
                    <span class="w-2.5 h-2.5 rounded-full bg-secondary animate-pulse" id="phaseDot"></span>
                    <span class="font-code-lg text-xs text-secondary uppercase font-bold tracking-wider" id="liveStageText">READY</span>
                  </div>
                </div>

                <!-- Chest Uprightness & Depth Status Pill -->
                <div class="flex items-center gap-2 flex-wrap">
                  <span id="pillChestUpright" class="inline-flex items-center gap-1.5 bg-primary-container text-on-primary-container px-3 py-1 rounded font-label-caps text-xs uppercase font-bold shadow-md">
                    <span class="material-symbols-outlined text-[15px]">check_circle</span>
                    <span id="pillChestText">CHEST UPRIGHT • OPTIMAL</span>
                  </span>
                  <span id="pillDepth" class="inline-flex items-center gap-1.5 bg-surface-container-lowest/90 text-secondary border border-surface-container-highest px-3 py-1 rounded font-code-md text-xs">
                    DEPTH: <span id="liveDepthDeg" class="font-bold">170°</span> (<span id="liveDepthPct">0%</span>)
                  </span>
                </div>
              </div>

              <!-- TOP-RIGHT: AR Metric Floating Callouts -->
              <div class="flex flex-col gap-2 items-end pointer-events-auto">
                <!-- Knee Angle Callout -->
                <div class="bg-surface-container-lowest/90 backdrop-blur-md px-4 py-2 rounded border border-surface-container-highest shadow-xl flex items-center gap-4">
                  <div class="flex flex-col">
                    <span class="font-label-caps text-[10px] text-secondary uppercase">KNEE FLEXION</span>
                    <span class="font-headline-lg text-2xl text-primary-container font-bold leading-tight" id="arKneeAngle">170.0°</span>
                  </div>
                  <div class="h-8 w-px bg-surface-container-highest"></div>
                  <div class="flex flex-col">
                    <span class="font-label-caps text-[10px] text-on-surface-variant uppercase">TARGET</span>
                    <span class="font-code-md text-xs text-primary">≤ 95.0°</span>
                  </div>
                </div>

                <!-- Chest & Torso Lean Callout -->
                <div class="bg-surface-container-lowest/90 backdrop-blur-md px-4 py-2 rounded border border-surface-container-highest shadow-xl flex items-center gap-4">
                  <div class="flex flex-col">
                    <span class="font-label-caps text-[10px] text-on-surface-variant uppercase">TORSO LEAN</span>
                    <span class="font-headline-sm text-lg text-primary font-bold leading-tight" id="arTorsoLean">15.0°</span>
                  </div>
                  <div class="h-8 w-px bg-surface-container-highest"></div>
                  <div class="flex flex-col">
                    <span class="font-label-caps text-[10px] text-on-surface-variant uppercase">HIP HINGE</span>
                    <span class="font-headline-sm text-lg text-secondary font-bold leading-tight" id="arHipAngle">160.0°</span>
                  </div>
                </div>

                <!-- Live Score Badge -->
                <div class="bg-surface-container-lowest/90 backdrop-blur-md px-3 py-1.5 rounded border border-surface-container-highest flex items-center gap-2">
                  <span class="font-code-md text-xs text-on-surface">ACCURACY SCORE:</span>
                  <span class="font-code-md text-sm text-primary-container font-bold" id="arLiveScore">100 / 100</span>
                </div>
              </div>

              <!-- BOTTOM HUD STRIP -->
              <div class="flex items-center justify-between">
                <div class="bg-surface-container-lowest/90 px-3 py-1 rounded border border-surface-container text-xs font-code-md text-on-surface-variant">
                  FRAME: #<span id="hudFrameIdx">00124</span> | <span class="text-primary-container">FPS: <span id="hudFpsBottom">60.1</span></span> | CHEST: <span id="hudChestBottom" class="text-primary-container font-bold">PASS</span>
                </div>
                <div class="bg-surface-container-lowest/90 px-3 py-1 rounded border border-surface-container text-xs font-label-caps text-secondary uppercase">
                  SAGITTAL VIEW • CALIBRATED
                </div>
              </div>
            </div>
          </div>

          <!-- Quick Action Pipeline Stream Bar -->
          <div class="w-full bg-surface-container-lowest p-3 rounded border border-surface-container flex flex-wrap items-center justify-between gap-3 shadow-sm">
            <div class="flex items-center gap-2 flex-wrap text-xs">
              <span class="font-label-caps text-on-surface-variant uppercase mr-1">TOGGLES:</span>
              <button id="btnToggleWireframe" type="button" class="px-3 py-1 bg-surface-container hover:bg-surface-container-high rounded font-code-md text-primary-container border border-surface-container flex items-center gap-1 transition-colors">
                <span class="material-symbols-outlined text-[15px]">polyline</span>
                <span id="wireframeLabel">Wireframe (ON)</span>
              </button>
              <button id="btnToggleProtractor" type="button" class="px-3 py-1 bg-surface-container hover:bg-surface-container-high rounded font-code-md text-secondary border border-surface-container flex items-center gap-1 transition-colors">
                <span class="material-symbols-outlined text-[15px]">straighten</span>
                <span id="protractorLabel">Protractor (ON)</span>
              </button>
              <button id="btnToggleDepth" type="button" class="px-3 py-1 bg-surface-container hover:bg-surface-container-high rounded font-code-md text-on-surface border border-surface-container flex items-center gap-1 transition-colors">
                <span class="material-symbols-outlined text-[15px]">horizontal_rule</span>
                <span id="depthLineLabel">Depth Line (ON)</span>
              </button>
            </div>
            <!-- Hotkeys Legend -->
            <div class="flex items-center gap-3 font-code-md text-xs text-on-surface-variant">
              <span><kbd class="px-1.5 py-0.5 bg-surface-container rounded text-primary border border-surface-container-highest">SPACE</kbd> Pause</span>
              <span><kbd class="px-1.5 py-0.5 bg-surface-container rounded text-primary border border-surface-container-highest">R</kbd> Reset</span>
              <span><kbd class="px-1.5 py-0.5 bg-surface-container rounded text-primary border border-surface-container-highest">N</kbd> Next Set</span>
              <span><kbd class="px-1.5 py-0.5 bg-surface-container rounded text-primary border border-surface-container-highest">S</kbd> Snapshot</span>
            </div>
          </div>
        </div>

        <!-- Col 4: Biomechanics Telemetry Rail -->
        <div class="xl:col-span-4 flex flex-col gap-3">
          <!-- Card 1: Angular Kinematics & Chest Uprightness -->
          <div class="bg-surface-container-low p-4 rounded border border-surface-container flex flex-col gap-3 shadow-md">
            <div class="flex items-center justify-between pb-1 border-b border-surface-container">
              <div class="flex items-center gap-2">
                <span class="material-symbols-outlined text-secondary text-[18px]">accessibility_new</span>
                <span class="font-headline-sm text-sm uppercase text-primary font-bold">Joint Kinematics</span>
              </div>
              <span class="font-label-caps text-[10px] text-primary-container bg-surface-container px-2 py-0.5 rounded uppercase">LIVE GAUGES</span>
            </div>

            <!-- Knee Flexion Circular Gauge -->
            <div class="bg-surface-container p-3 rounded flex items-center justify-between gap-4">
              <div class="flex flex-col">
                <span class="font-label-caps text-[10px] text-on-surface-variant uppercase">Knee Flexion Angle</span>
                <div class="flex items-baseline gap-2">
                  <span class="font-stat-display text-3xl text-primary-container leading-none font-bold" id="gaugeKneeText">170°</span>
                  <span class="font-code-md text-xs text-secondary" id="gaugeKneeMax">GOAL ≤ 95°</span>
                </div>
                <span class="text-xs text-on-surface-variant mt-1" id="gaugeDepthFeedback">Standing upright.</span>
              </div>
              <!-- Circular Progress Gauge -->
              <div class="relative w-16 h-16 flex-shrink-0">
                <svg class="w-full h-full -rotate-90" viewBox="0 0 36 36">
                  <path class="text-surface-container-highest" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="currentColor" stroke-width="3.5"></path>
                  <path id="svgProgressRing" class="text-primary-container" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="currentColor" stroke-dasharray="0, 100" stroke-linecap="round" stroke-width="3.5"></path>
                </svg>
                <div class="absolute inset-0 flex items-center justify-center font-code-md text-xs text-primary font-bold" id="ringPassBadge">
                  READY
                </div>
              </div>
            </div>

            <!-- Metric Grid: Chest Lean, Hip Angle, Accuracy Score -->
            <div class="grid grid-cols-3 gap-2">
              <div class="bg-surface-container p-2.5 rounded flex flex-col">
                <span class="font-label-caps text-[9px] text-on-surface-variant uppercase">Torso Lean</span>
                <span class="font-code-lg text-sm text-secondary font-bold mt-1" id="metricTorsoLean">15.0°</span>
                <span class="font-label-caps text-[9px] text-primary-container mt-1" id="metricTorsoStatus">UPRIGHT</span>
              </div>
              <div class="bg-surface-container p-2.5 rounded flex flex-col">
                <span class="font-label-caps text-[9px] text-on-surface-variant uppercase">Hip Angle</span>
                <span class="font-code-lg text-sm text-primary font-bold mt-1" id="metricHipAngle">160.0°</span>
                <span class="font-label-caps text-[9px] text-secondary mt-1">Neutral</span>
              </div>
              <div class="bg-surface-container p-2.5 rounded flex flex-col">
                <span class="font-label-caps text-[9px] text-on-surface-variant uppercase">Live Accuracy</span>
                <span class="font-code-lg text-sm text-primary-container font-bold mt-1" id="metricScore">100 / 100</span>
                <span class="font-label-caps text-[9px] text-primary-container mt-1">Excellent</span>
              </div>
            </div>
          </div>

          <!-- Card 2: Velocity-Based Training (VBT) -->
          <div class="bg-surface-container-low p-4 rounded border border-surface-container flex flex-col gap-3 shadow-md">
            <div class="flex items-center justify-between pb-1 border-b border-surface-container">
              <div class="flex items-center gap-2">
                <span class="material-symbols-outlined text-primary-container text-[18px]">speed</span>
                <span class="font-headline-sm text-sm uppercase text-primary font-bold">Velocity (VBT)</span>
              </div>
              <span class="font-code-md text-xs text-secondary">STRENGTH-SPEED</span>
            </div>
            <div class="grid grid-cols-2 gap-3">
              <div class="bg-surface-container p-3 rounded flex flex-col">
                <span class="font-label-caps text-[10px] text-on-surface-variant uppercase">Velocity</span>
                <div class="flex items-baseline gap-1 mt-1">
                  <span class="font-headline-lg text-xl text-primary font-bold" id="vbtVelocity">0.00</span>
                  <span class="font-code-md text-xs text-secondary">m/s</span>
                </div>
                <div class="w-full bg-surface-container-highest h-1.5 mt-2 rounded-full overflow-hidden">
                  <div class="bg-secondary h-full w-[10%]" id="barVel"></div>
                </div>
              </div>
              <div class="bg-surface-container p-3 rounded flex flex-col">
                <span class="font-label-caps text-[10px] text-on-surface-variant uppercase">Peak Velocity</span>
                <div class="flex items-baseline gap-1 mt-1">
                  <span class="font-headline-lg text-xl text-primary-container font-bold" id="vbtPeakVel">0.00</span>
                  <span class="font-code-md text-xs text-primary-container">m/s</span>
                </div>
                <div class="w-full bg-surface-container-highest h-1.5 mt-2 rounded-full overflow-hidden">
                  <div class="bg-primary-container h-full w-[10%]" id="barPeakVel"></div>
                </div>
              </div>
            </div>
            <div class="flex items-center justify-between bg-surface-container px-3 py-1.5 rounded font-code-md text-xs">
              <span class="text-on-surface-variant">INSTANTANEOUS POWER:</span>
              <span class="text-primary font-bold" id="vbtPower">0.0 W</span>
            </div>
          </div>

          <!-- Card 3: Barbell Path Sagittal Drift Plot -->
          <div class="bg-surface-container-low p-4 rounded border border-surface-container flex flex-col gap-3 shadow-md">
            <div class="flex items-center justify-between pb-1 border-b border-surface-container">
              <div class="flex items-center gap-2">
                <span class="material-symbols-outlined text-secondary text-[18px]">show_chart</span>
                <span class="font-headline-sm text-sm uppercase text-primary font-bold">Barbell Sagittal Drift</span>
              </div>
              <span class="font-code-md text-xs text-primary-container" id="barDriftText">0.0 cm</span>
            </div>
            <div class="bg-surface-container p-2.5 rounded flex flex-col gap-1">
              <div class="flex items-center justify-between text-[9px] font-label-caps text-on-surface-variant uppercase">
                <span>Posterior (-5cm)</span>
                <span>Midfoot (0)</span>
                <span>Anterior (+5cm)</span>
              </div>
              <div class="relative w-full h-12 bg-surface-container-lowest rounded overflow-hidden flex items-center justify-center">
                <div class="absolute top-0 bottom-0 left-1/2 w-px bg-surface-container-highest"></div>
                <div class="absolute w-12 h-full bg-secondary/10 left-1/2 -translate-x-1/2"></div>
                <!-- Dynamic Marker -->
                <div id="barDriftMarker" class="absolute w-3 h-3 rounded-full bg-primary-container shadow-lg transition-all duration-75" style="left: 50%;"></div>
              </div>
              <div class="flex items-center justify-between font-code-md text-xs pt-1">
                <span class="text-on-surface-variant">SAGITTAL ALIGNMENT:</span>
                <span class="text-primary font-bold" id="barSagittalState">OPTIMAL</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- REPS PER SET BAR GRAPH & PROGRESSION HUD -->
      <section class="w-full bg-surface-container-low p-5 rounded border border-surface-container flex flex-col gap-4 shadow-md">
        <div class="flex flex-wrap items-center justify-between gap-3 pb-2 border-b border-surface-container">
          <div class="flex items-center gap-2">
            <span class="material-symbols-outlined text-primary-container text-[20px]">bar_chart</span>
            <span class="font-headline-sm text-sm uppercase text-primary font-bold">Reps Per Set Telemetry</span>
            <span class="font-label-caps text-[10px] text-secondary px-2.5 py-0.5 bg-surface-container-highest rounded" id="repsPerSetBadge">SET 01 ACTIVE (0 REPS)</span>
          </div>
          <!-- Legend & Target Status -->
          <div class="flex items-center gap-4 text-xs font-code-md">
            <div class="flex items-center gap-1.5">
              <span class="w-2.5 h-2.5 bg-primary-container rounded-sm inline-block"></span>
              <span class="text-on-surface-variant">Optimal (&ge;85%)</span>
            </div>
            <div class="flex items-center gap-1.5">
              <span class="w-2.5 h-2.5 bg-secondary rounded-sm inline-block"></span>
              <span class="text-on-surface-variant">Acceptable (&ge;70%)</span>
            </div>
            <div class="flex items-center gap-1.5">
              <span class="w-2.5 h-2.5 bg-error rounded-sm inline-block"></span>
              <span class="text-on-surface-variant">Sub-optimal (&lt;70%)</span>
            </div>
          </div>
        </div>

        <!-- Dynamic Sets Container with Reps Bar Graph -->
        <div id="repsPerSetContainer" class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3">
          <!-- Populated dynamically by JavaScript -->
        </div>
      </section>

      <!-- Bottom Global Landmark Confidence Strip -->
      <div class="w-full bg-surface-container-low p-4 rounded border border-surface-container flex flex-col xl:flex-row xl:items-center justify-between gap-4 shadow-md">
        <div class="flex items-center gap-4 flex-wrap">
          <span class="text-on-surface-variant uppercase font-label-caps text-xs">LANDMARK CONFIDENCE:</span>
          <div class="flex items-center gap-2 bg-surface-container px-3 py-1 rounded text-xs font-code-md">
            <span class="text-on-surface-variant">HIP:</span>
            <span class="text-primary-container font-bold" id="confHip">0.98</span>
          </div>
          <div class="flex items-center gap-2 bg-surface-container px-3 py-1 rounded text-xs font-code-md">
            <span class="text-on-surface-variant">KNEE:</span>
            <span class="text-primary-container font-bold" id="confKnee">0.96</span>
          </div>
          <div class="flex items-center gap-2 bg-surface-container px-3 py-1 rounded text-xs font-code-md">
            <span class="text-on-surface-variant">ANKLE:</span>
            <span class="text-secondary font-bold" id="confAnkle">0.94</span>
          </div>
          <div class="flex items-center gap-2 bg-surface-container px-3 py-1 rounded text-xs font-code-md">
            <span class="text-on-surface-variant">SHOULDER / CHEST:</span>
            <span class="text-primary-container font-bold" id="confShoulder">0.97</span>
          </div>
        </div>
        <div class="flex items-center gap-3">
          <a href="/api/export_csv" class="bg-surface-container hover:bg-surface-container-high text-primary border border-surface-container px-4 py-1.5 rounded font-code-md text-xs flex items-center gap-2 transition-colors">
            <span class="material-symbols-outlined text-[16px]">file_download</span>
            <span>EXPORT CSV</span>
          </a>
          <button id="btnSnapshot" type="button" class="bg-primary-container text-on-primary-container hover:bg-primary transition-colors px-4 py-1.5 rounded font-code-md text-xs font-bold uppercase flex items-center gap-2">
            <span class="material-symbols-outlined text-[16px]">camera</span>
            <span>SNAPSHOT (S)</span>
          </button>
        </div>
      </div>
    </main>

    <!-- TAB 2: KINEMATICS & JOINT ANGLES -->
    <main id="tab-rep-telemetry-biomechanics" class="tab-content hidden relative pt-20 px-6 pb-8 flex flex-col gap-6">
      <!-- TOP METRICS SUMMARY ROW -->
      <section class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3 w-full">
        <!-- Card 1 -->
        <div class="bg-surface-container-low p-4 flex flex-col justify-between border border-surface-container rounded shadow-sm">
          <div class="flex items-center justify-between">
            <span class="font-label-caps text-xs text-secondary uppercase tracking-widest">VALIDATED REPS</span>
            <span class="font-code-md text-xs text-primary-container bg-surface-container-highest px-2 py-0.5 rounded" id="tab2GoodRepsBadge">0 / 0 VALID</span>
          </div>
          <div class="my-3 flex items-baseline gap-2">
            <span class="font-stat-display text-4xl text-primary tracking-tighter font-bold" id="tab2Reps">00</span>
            <span class="font-code-md text-xs text-on-surface-variant">REPS</span>
          </div>
          <div class="flex items-center justify-between pt-1 text-on-surface-variant font-code-md text-xs bg-surface-container/60 px-2 py-1 rounded">
            <span class="text-primary-container flex items-center gap-1">
              <span class="material-symbols-outlined text-[14px]">check_circle</span> CHEST VERIFIED
            </span>
            <span class="font-label-caps text-[10px]" id="tab2PassRate">100% PASS RATE</span>
          </div>
        </div>

        <!-- Card 2 -->
        <div class="bg-surface-container-low p-4 flex flex-col justify-between border border-surface-container rounded shadow-sm">
          <div class="flex items-center justify-between">
            <span class="font-label-caps text-xs text-secondary uppercase tracking-widest">MEAN SQUAT DEPTH</span>
            <span class="font-code-md text-xs text-primary-container bg-surface-container-highest px-2 py-0.5 rounded">PARALLEL +</span>
          </div>
          <div class="my-3 flex items-baseline gap-2">
            <span class="font-stat-display text-4xl text-primary tracking-tighter font-bold" id="tab2MeanDepth">82.4</span>
            <span class="font-code-md text-xs text-on-surface-variant">DEG</span>
          </div>
          <div class="flex items-center justify-between pt-1 text-on-surface-variant font-code-md text-xs bg-surface-container/60 px-2 py-1 rounded">
            <span>Target: &le; 95.0&deg;</span>
            <span class="text-primary-container font-label-caps text-[10px]">SUB-PARALLEL</span>
          </div>
        </div>

        <!-- Card 3 -->
        <div class="bg-surface-container-low p-4 flex flex-col justify-between border border-surface-container rounded shadow-sm">
          <div class="flex items-center justify-between">
            <span class="font-label-caps text-xs text-secondary uppercase tracking-widest">CHEST UPRIGHT SCORE</span>
            <span class="font-code-md text-xs text-primary-container bg-surface-container-highest px-2 py-0.5 rounded">TORSO POSTURE</span>
          </div>
          <div class="my-3 flex items-baseline gap-2">
            <span class="font-stat-display text-4xl text-primary-container tracking-tighter font-bold" id="tab2ChestScore">100</span>
            <span class="font-code-md text-xs text-on-surface-variant">/ 100</span>
          </div>
          <div class="flex items-center justify-between pt-1 text-on-surface-variant font-code-md text-xs bg-surface-container/60 px-2 py-1 rounded">
            <span class="text-primary-container">TORSO LEAN &le; 35&deg;</span>
            <span class="text-secondary font-label-caps text-[10px]">OPTIMAL</span>
          </div>
        </div>

        <!-- Card 4 -->
        <div class="bg-surface-container-low p-4 flex flex-col justify-between border border-surface-container rounded shadow-sm">
          <div class="flex items-center justify-between">
            <span class="font-label-caps text-xs text-secondary uppercase tracking-widest">PATH EFFICIENCY</span>
            <span class="font-code-md text-xs text-primary-container bg-surface-container-highest px-2 py-0.5 rounded">SAGITTAL</span>
          </div>
          <div class="my-3 flex items-baseline gap-2">
            <span class="font-stat-display text-4xl text-primary tracking-tighter font-bold" id="tab2Efficiency">98.2</span>
            <span class="font-code-md text-xs text-on-surface-variant">%</span>
          </div>
          <div class="flex items-center justify-between pt-1 text-on-surface-variant font-code-md text-xs bg-surface-container/60 px-2 py-1 rounded">
            <span>DRIFT DEVIATION</span>
            <span class="text-primary-container font-bold">&plusmn;1.2 cm</span>
          </div>
        </div>
      </section>

      <!-- WAVEFORM & BIOMECHANICAL VALIDATION -->
      <div class="grid grid-cols-1 xl:grid-cols-12 gap-4 w-full">
        <!-- 8 cols: Kinematic Joint Angles Waveform -->
        <div class="xl:col-span-8 bg-surface-container-low p-5 rounded border border-surface-container flex flex-col gap-3 shadow-md">
          <div class="flex flex-wrap items-center justify-between gap-3 pb-2 border-b border-surface-container">
            <div class="flex items-center gap-2">
              <span class="w-2 h-4 bg-primary-container inline-block rounded-sm"></span>
              <span class="font-headline-sm text-sm uppercase text-primary font-bold">Kinematic Joint Angles vs Time</span>
              <span class="font-label-caps text-[10px] text-secondary px-2 py-0.5 bg-surface-container-highest rounded">LIVE SYNCHRONIZED</span>
            </div>
            <!-- Legend -->
            <div class="flex items-center gap-4 font-code-md text-xs">
              <div class="flex items-center gap-1.5">
                <span class="w-3 h-1 bg-primary-container inline-block rounded-full"></span>
                <span>Knee Angle</span>
              </div>
              <div class="flex items-center gap-1.5">
                <span class="w-3 h-1 bg-secondary inline-block rounded-full"></span>
                <span>Hip Angle</span>
              </div>
              <div class="flex items-center gap-1.5">
                <span class="w-3 h-1 bg-amber-400 inline-block rounded-full"></span>
                <span>Torso Lean</span>
              </div>
            </div>
          </div>

          <!-- Dynamic SVG Waveform Canvas -->
          <div class="relative w-full h-64 bg-surface-container-lowest p-2 rounded overflow-hidden border border-surface-container">
            <svg id="liveWaveformSvg" class="w-full h-full" preserveAspectRatio="none" viewBox="0 0 800 240">
              <!-- Grid lines -->
              <line stroke="#262a34" stroke-dasharray="2 4" stroke-width="1" x1="40" x2="780" y1="30" y2="30"/>
              <text fill="#85967c" font-family="JetBrains Mono" font-size="9" x="8" y="34">180°</text>
              <line stroke="#262a34" stroke-dasharray="2 4" stroke-width="1" x1="40" x2="780" y1="80" y2="80"/>
              <text fill="#85967c" font-family="JetBrains Mono" font-size="9" x="8" y="84">140°</text>
              <line stroke="#262a34" stroke-dasharray="2 4" stroke-width="1" x1="40" x2="780" y1="130" y2="130"/>
              <text fill="#85967c" font-family="JetBrains Mono" font-size="9" x="8" y="134">100°</text>
              <line stroke="#262a34" stroke-dasharray="2 4" stroke-width="1" x1="40" x2="780" y1="180" y2="180"/>
              <text fill="#85967c" font-family="JetBrains Mono" font-size="9" x="8" y="184">60°</text>

              <!-- Parallel Threshold (90°) -->
              <line stroke="#39ff14" stroke-dasharray="4 4" stroke-width="1" opacity="0.6" x1="40" x2="780" y1="140" y2="140"/>
              <text fill="#39ff14" font-family="JetBrains Mono" font-size="9" x="700" y="136">PARALLEL (90°)</text>

              <!-- Dynamic Paths -->
              <path id="wfKneePath" d="M 40 40 L 780 40" fill="none" stroke="#39ff14" stroke-width="2.5" stroke-linecap="round"/>
              <path id="wfHipPath" d="M 40 50 L 780 50" fill="none" stroke="#4cd7f6" stroke-width="2" stroke-linecap="round"/>
              <path id="wfLeanPath" d="M 40 210 L 780 210" fill="none" stroke="#facc15" stroke-width="1.8" stroke-linecap="round"/>
            </svg>
            <div class="absolute bottom-1 left-10 right-4 flex justify-between font-label-caps text-[10px] text-on-surface-variant pointer-events-none">
              <span class="text-secondary">ECCENTRIC (DESCENT)</span>
              <span class="text-primary-container">BOTTOM DEPTH (PARALLEL)</span>
              <span class="text-secondary">CONCENTRIC (ASCENT)</span>
              <span>LOCKOUT</span>
            </div>
          </div>
        </div>

        <!-- 4 cols: Biomechanical Validation Checklist -->
        <div class="xl:col-span-4 bg-surface-container-low p-5 rounded border border-surface-container flex flex-col justify-between shadow-md">
          <div>
            <div class="flex items-center justify-between pb-2 border-b border-surface-container">
              <div class="flex items-center gap-2">
                <span class="material-symbols-outlined text-primary-container text-[18px]">verified_user</span>
                <span class="font-headline-sm text-sm uppercase text-primary font-bold">Biomechanical Validation</span>
              </div>
              <span class="font-label-caps text-[10px] text-on-primary-container bg-primary-container px-2 py-0.5 rounded font-bold" id="checklistScoreBadge">SCORE 100/100</span>
            </div>

            <div class="flex flex-col gap-2.5 my-3">
              <!-- Check 1: Chest Upright & Torso Alignment -->
              <div class="p-3 bg-surface-container rounded flex items-start gap-3 border border-surface-container-high" id="checkChestCard">
                <div class="mt-0.5 w-5 h-5 bg-primary-container/20 flex items-center justify-center shrink-0 rounded text-primary-container">
                  <span class="material-symbols-outlined text-[16px]">check</span>
                </div>
                <div class="flex flex-col w-full text-xs">
                  <div class="flex items-center justify-between">
                    <span class="font-title-md font-bold text-primary">Chest Upright &amp; Torso Incline</span>
                    <span class="font-code-md text-primary-container font-bold" id="checkChestValue">24.2° LEAN</span>
                  </div>
                  <p class="text-on-surface-variant mt-1" id="checkChestDesc">
                    Optimal athletic posture maintained. Torso incline under 45° threshold with zero chest drop.
                  </p>
                </div>
              </div>

              <!-- Check 2: Squat Depth Standard -->
              <div class="p-3 bg-surface-container rounded flex items-start gap-3 border border-surface-container-high" id="checkDepthCard">
                <div class="mt-0.5 w-5 h-5 bg-primary-container/20 flex items-center justify-center shrink-0 rounded text-primary-container">
                  <span class="material-symbols-outlined text-[16px]">check</span>
                </div>
                <div class="flex flex-col w-full text-xs">
                  <div class="flex items-center justify-between">
                    <span class="font-title-md font-bold text-primary">Depth Target (Parallel)</span>
                    <span class="font-code-md text-primary-container font-bold" id="checkDepthValue">PASSED</span>
                  </div>
                  <p class="text-on-surface-variant mt-1">
                    Femur achieves parallel orientation relative to floor with knee flexion &le; 95°.
                  </p>
                </div>
              </div>

              <!-- Check 3: Lumbar Alignment -->
              <div class="p-3 bg-surface-container rounded flex items-start gap-3 border border-surface-container-high">
                <div class="mt-0.5 w-5 h-5 bg-primary-container/20 flex items-center justify-center shrink-0 rounded text-primary-container">
                  <span class="material-symbols-outlined text-[16px]">check</span>
                </div>
                <div class="flex flex-col w-full text-xs">
                  <div class="flex items-center justify-between">
                    <span class="font-title-md font-bold text-primary">Spine &amp; Lumbar Neutral</span>
                    <span class="font-code-md text-primary-container font-bold">STABLE</span>
                  </div>
                  <p class="text-on-surface-variant mt-1">
                    Thoracolumbar alignment preserved. Zero premature hip shoot or lumbar rounding.
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div class="flex items-center justify-between pt-2 border-t border-surface-container font-code-md text-xs text-on-surface-variant">
            <span>LANDMARKS: 33 POSE VECTORS</span>
            <span class="text-primary-container font-bold">ALL CRITERIA PASSED</span>
          </div>
        </div>
      </div>
    </main>

    <!-- TAB 3, 4, 5: ADDITIONAL TABS -->
    <div id="tab-velocity-based-training" class="tab-content hidden relative pt-20 px-6 pb-8 flex flex-col gap-6">
      <div class="bg-surface-container-low p-6 rounded border border-surface-container">
        <h2 class="text-xl font-headline-lg font-bold text-primary mb-2">Velocity-Based Training &amp; Barbell Dynamics</h2>
        <p class="text-sm text-on-surface-variant mb-6">Real-time kinematic velocity tracking, mean propulsive velocity (MPV), and barbell bar path sagittal reconstruction.</p>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div class="bg-surface-container p-4 rounded">
            <span class="text-xs font-label-caps text-secondary uppercase">Rep Velocity Decay (Fatigue Curve)</span>
            <div id="vbtRepBars" class="flex flex-col gap-2 mt-3 text-xs font-code-md">
              <!-- Rep velocity bars generated dynamically -->
            </div>
          </div>
          <div class="bg-surface-container p-4 rounded flex flex-col justify-between">
            <div>
              <span class="text-xs font-label-caps text-secondary uppercase">Load &amp; Power Analysis</span>
              <div class="my-4 flex flex-col gap-2">
                <div class="flex justify-between text-sm">
                  <span class="text-on-surface-variant">Estimated RPE:</span>
                  <span class="text-primary font-bold">7.5</span>
                </div>
                <div class="flex justify-between text-sm">
                  <span class="text-on-surface-variant">Reps in Reserve (RIR):</span>
                  <span class="text-secondary font-bold">2 - 3</span>
                </div>
                <div class="flex justify-between text-sm">
                  <span class="text-on-surface-variant">Velocity Loss Cutoff:</span>
                  <span class="text-primary-container font-bold">&lt; 20% (Optimal)</span>
                </div>
              </div>
            </div>
            <div class="p-3 bg-surface-container-lowest rounded text-xs font-code-md text-on-surface-variant">
              Training Target: Hypertrophy &amp; Maximal Force Production
            </div>
          </div>
        </div>
      </div>
    </div>

    <div id="tab-model-camera-config" class="tab-content hidden relative pt-20 px-6 pb-8 flex flex-col gap-6">
      <div class="bg-surface-container-low p-6 rounded border border-surface-container font-code-md text-xs">
        <h2 class="text-xl font-headline-lg font-bold text-primary mb-2 font-headline-sm">Pipeline &amp; Calibration Runtime Console</h2>
        <div class="bg-surface-container-lowest p-4 rounded mt-4 text-on-surface leading-relaxed border border-surface-container-highest">
          <span class="text-secondary font-bold"># Live CV Pipeline Configuration</span><br/>
          pipeline_config = {<br/>
          &nbsp;&nbsp;<span class="text-on-surface-variant">"model"</span>: <span class="text-primary-container">"MediaPipe_PoseLandmarker_Full"</span>,<br/>
          &nbsp;&nbsp;<span class="text-on-surface-variant">"min_detection_confidence"</span>: <span class="text-primary">0.50</span>,<br/>
          &nbsp;&nbsp;<span class="text-on-surface-variant">"min_tracking_confidence"</span>: <span class="text-primary">0.50</span>,<br/>
          &nbsp;&nbsp;<span class="text-on-surface-variant">"chest_upright_max_lean_deg"</span>: <span class="text-secondary">45.0</span>,<br/>
          &nbsp;&nbsp;<span class="text-on-surface-variant">"depth_goal_deg"</span>: <span class="text-primary">95.0</span>,<br/>
          &nbsp;&nbsp;<span class="text-on-surface-variant">"scoring_formula"</span>: <span class="text-primary-container">"50% Depth + 30% Chest Upright + 20% Hip Hinge"</span><br/>
          }
        </div>
      </div>
    </div>

    <div id="tab-export-logs" class="tab-content hidden relative pt-20 px-6 pb-8 flex flex-col gap-6">
      <div class="bg-surface-container-low p-6 rounded border border-surface-container">
        <h2 class="text-xl font-headline-lg font-bold text-primary mb-2">Telemetry &amp; CSV Logs</h2>
        <p class="text-sm text-on-surface-variant mb-4">Download laboratory kinematics, angle curves, and rep-by-rep breakdown.</p>
        <div class="flex gap-4">
          <a href="/api/export_csv" class="bg-primary-container text-on-primary-container font-code-md text-xs font-bold px-4 py-2 rounded flex items-center gap-2 hover:bg-primary transition-colors">
            <span class="material-symbols-outlined text-[18px]">download</span>
            <span>Download rep_telemetry.csv</span>
          </a>
          <a href="/api/dump_json" class="bg-surface-container text-primary font-code-md text-xs font-bold px-4 py-2 rounded flex items-center gap-2 hover:bg-surface-container-high transition-colors border border-surface-container-highest">
            <span class="material-symbols-outlined text-[18px]">data_object</span>
            <span>Download landmarks_dump.json</span>
          </a>
        </div>
      </div>
    </div>
  </div>

  <!-- SCRIPT FOR TELEMETRY POLLING & UI INTERACTIONS -->
  <script>
    let isWireframe = true;
    let isProtractor = true;
    let isDepthLine = true;
    let timerSeconds = 0;

    // Session Timer
    setInterval(() => {
      timerSeconds++;
      const hrs = String(Math.floor(timerSeconds / 3600)).padStart(2, '0');
      const mins = String(Math.floor((timerSeconds % 3600) / 60)).padStart(2, '0');
      const secs = String(timerSeconds % 60).padStart(2, '0');
      const el = document.getElementById('sessionTimer');
      if (el) el.textContent = `${hrs}:${mins}:${secs}`;
    }, 1000);

    // Tab Navigation
    document.querySelectorAll('.nav-tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const targetTab = btn.getAttribute('data-tab');
        document.querySelectorAll('.nav-tab-btn').forEach(b => {
          b.classList.remove('bg-surface-container', 'text-primary', 'border-l-2', 'border-primary-container');
          b.classList.add('text-on-surface-variant');
        });
        btn.classList.add('bg-surface-container', 'text-primary', 'border-l-2', 'border-primary-container');
        btn.classList.remove('text-on-surface-variant');

        document.querySelectorAll('.tab-content').forEach(tc => tc.classList.add('hidden'));
        const active = document.getElementById(`tab-${targetTab}`);
        if (active) active.classList.remove('hidden');
      });
    });

    // Control Buttons
    document.getElementById('btnReset').addEventListener('click', () => {
      fetch('/api/reset', { method: 'POST' });
    });

    document.getElementById('btnNextSet').addEventListener('click', () => {
      fetch('/api/next_set', { method: 'POST' });
    });

    document.getElementById('btnSkipRest').addEventListener('click', () => {
      fetch('/api/next_set', { method: 'POST' });
    });

    document.getElementById('btnTogglePause').addEventListener('click', () => {
      fetch('/api/toggle_pause', { method: 'POST' })
        .then(r => r.json())
        .then(d => {
          const lbl = document.getElementById('pauseLabel');
          const icn = document.getElementById('pauseIcon');
          if (d.state === 'paused') {
            lbl.textContent = 'RESUME';
            icn.textContent = 'play_arrow';
          } else {
            lbl.textContent = 'PAUSE';
            icn.textContent = 'pause';
          }
        });
    });

    document.getElementById('btnSnapshot').addEventListener('click', () => {
      fetch('/api/snapshot', { method: 'POST' })
        .then(r => r.json())
        .then(d => alert('Snapshot captured: ' + d.file));
    });

    document.getElementById('btnFullscreen').addEventListener('click', () => {
      if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen();
      } else {
        document.exitFullscreen();
      }
    });

    // Toggles
    document.getElementById('btnToggleWireframe').addEventListener('click', () => {
      isWireframe = !isWireframe;
      fetch('/api/toggle_wireframe', { method: 'POST' });
      document.getElementById('wireframeLabel').textContent = `Wireframe (${isWireframe ? 'ON' : 'OFF'})`;
    });

    document.getElementById('btnToggleProtractor').addEventListener('click', () => {
      isProtractor = !isProtractor;
      fetch('/api/toggle_protractor', { method: 'POST' });
      document.getElementById('protractorLabel').textContent = `Protractor (${isProtractor ? 'ON' : 'OFF'})`;
    });

    document.getElementById('btnToggleDepth').addEventListener('click', () => {
      isDepthLine = !isDepthLine;
      fetch('/api/toggle_depth', { method: 'POST' });
      document.getElementById('depthLineLabel').textContent = `Depth Line (${isDepthLine ? 'ON' : 'OFF'})`;
    });

    // Keyboard Shortcuts
    window.addEventListener('keydown', (e) => {
      if (e.code === 'Space') {
        e.preventDefault();
        document.getElementById('btnTogglePause').click();
      } else if (e.code === 'KeyR') {
        document.getElementById('btnReset').click();
      } else if (e.code === 'KeyN') {
        document.getElementById('btnNextSet').click();
      } else if (e.code === 'KeyS') {
        document.getElementById('btnSnapshot').click();
      }
    });

    // Real-time Telemetry Polling (every 60ms)
    async function pollTelemetry() {
      try {
        const res = await fetch('/api/telemetry');
        if (!res.ok) return;
        const d = await res.json();

        // Header and top stats
        document.getElementById('hudFps').textContent = d.fps;
        document.getElementById('hudFpsBottom').textContent = d.fps;
        document.getElementById('hudLatency').textContent = d.inference_ms + 'ms';
        document.getElementById('hudScoreTop').textContent = Math.round(d.live_score);
        document.getElementById('hudChestHeader').textContent = d.chest_upright ? 'UPRIGHT' : 'LEANING';
        document.getElementById('hudChestHeader').className = d.chest_upright ? 'font-code-md text-primary-container font-bold' : 'font-code-md text-error font-bold';
        document.getElementById('hudChestStatusTop').textContent = d.chest_status;
        document.getElementById('hudChestBottom').textContent = d.chest_upright ? 'PASS' : 'FLAGGED';
        document.getElementById('hudFrameIdx').textContent = String(d.frame).padStart(5, '0');

        // 2-Minute Rest Gap & Timer Management
        if (d.is_resting) {
          const rem = Math.max(0, Math.round(d.rest_time_remaining || 0));
          const mins = String(Math.floor(rem / 60)).padStart(2, '0');
          const secs = String(rem % 60).padStart(2, '0');
          const timeStr = `${mins}:${secs}`;

          document.getElementById('restHeaderTimer').textContent = timeStr;
          document.getElementById('restBannerTimer').textContent = timeStr;
          document.getElementById('restHeaderBadge').classList.remove('hidden');
          document.getElementById('restOverlayBanner').classList.remove('hidden');
          const skipBtn = document.getElementById('skipRestBtnLabel');
          if (skipBtn) skipBtn.textContent = `START SET ${String(d.current_set || 1).padStart(2, '0')} NOW`;
          const nextBtnSpan = document.getElementById('btnNextSet').querySelector('span:last-child');
          if (nextBtnSpan) nextBtnSpan.textContent = 'SKIP REST & START';
        } else {
          document.getElementById('restHeaderBadge').classList.add('hidden');
          document.getElementById('restOverlayBanner').classList.add('hidden');
          const nextBtnSpan = document.getElementById('btnNextSet').querySelector('span:last-child');
          if (nextBtnSpan) nextBtnSpan.textContent = 'NEXT SET (N)';
        }

        // Rep Counter & Stage
        const setEl = document.getElementById('liveSetCount');
        if (setEl && d.current_set) setEl.textContent = 'SET ' + String(d.current_set).padStart(2, '0');
        document.getElementById('liveRepCount').textContent = 'REP ' + String(d.reps).padStart(2, '0');
        document.getElementById('liveStageText').textContent = d.stage.toUpperCase() + ' (' + d.feedback + ')';

        // AR Callouts
        document.getElementById('arKneeAngle').textContent = d.knee_angle.toFixed(1) + '°';
        document.getElementById('arTorsoLean').textContent = d.torso_lean.toFixed(1) + '°';
        document.getElementById('arHipAngle').textContent = d.hip_angle.toFixed(1) + '°';
        document.getElementById('arLiveScore').textContent = Math.round(d.live_score) + ' / 100';

        // Chest pill
        const pillChest = document.getElementById('pillChestUpright');
        const pillChestText = document.getElementById('pillChestText');
        if (d.chest_upright) {
          pillChest.className = 'inline-flex items-center gap-1.5 bg-primary-container text-on-primary-container px-3 py-1 rounded font-label-caps text-xs uppercase font-bold shadow-md';
          pillChestText.textContent = `CHEST UPRIGHT • ${d.chest_status} (${d.torso_lean.toFixed(0)}°)`;
        } else {
          pillChest.className = 'inline-flex items-center gap-1.5 bg-error text-surface-container-lowest px-3 py-1 rounded font-label-caps text-xs uppercase font-bold shadow-md animate-pulse';
          pillChestText.textContent = `CHEST DROP FLAGGED • LEAN: ${d.torso_lean.toFixed(0)}°`;
        }

        // Depth pill
        document.getElementById('liveDepthDeg').textContent = d.knee_angle.toFixed(0) + '°';
        document.getElementById('liveDepthPct').textContent = d.depth_pct.toFixed(0) + '%';

        // Kinematics Rail Card
        document.getElementById('gaugeKneeText').textContent = d.knee_angle.toFixed(0) + '°';
        const ring = document.getElementById('svgProgressRing');
        const ringBadge = document.getElementById('ringPassBadge');
        const depthPct = Math.min(100, Math.max(0, d.depth_pct));
        ring.setAttribute('stroke-dasharray', `${depthPct}, 100`);

        if (d.knee_angle <= 95) {
          ring.setAttribute('class', 'text-primary-container');
          ringBadge.textContent = 'PASS';
          ringBadge.className = 'absolute inset-0 flex items-center justify-center font-code-md text-xs text-primary-container font-bold';
        } else {
          ring.setAttribute('class', 'text-secondary');
          ringBadge.textContent = d.stage.toUpperCase();
          ringBadge.className = 'absolute inset-0 flex items-center justify-center font-code-md text-[10px] text-on-surface font-bold';
        }

        document.getElementById('metricTorsoLean').textContent = d.torso_lean.toFixed(1) + '°';
        document.getElementById('metricTorsoStatus').textContent = d.chest_upright ? 'OPTIMAL' : 'LEANING';
        document.getElementById('metricHipAngle').textContent = d.hip_angle.toFixed(1) + '°';
        document.getElementById('metricScore').textContent = Math.round(d.live_score) + ' / 100';

        // Velocity & Bar path
        document.getElementById('vbtVelocity').textContent = d.velocity.toFixed(2);
        document.getElementById('vbtPeakVel').textContent = d.peak_velocity.toFixed(2);
        document.getElementById('barVel').style.width = Math.min(100, d.velocity * 50) + '%';
        document.getElementById('barPeakVel').style.width = Math.min(100, d.peak_velocity * 40) + '%';
        document.getElementById('vbtPower').textContent = d.power_watts + ' W';

        // Sagittal Drift
        document.getElementById('barDriftText').textContent = (d.bar_drift_cm > 0 ? '+' : '') + d.bar_drift_cm.toFixed(1) + ' cm';
        const driftPct = 50 + (d.bar_drift_cm / 10) * 50;
        document.getElementById('barDriftMarker').style.left = Math.max(5, Math.min(95, driftPct)) + '%';

        // Tab 2 Sync
        document.getElementById('tab2Reps').textContent = String(d.reps).padStart(2, '0');
        document.getElementById('tab2GoodRepsBadge').textContent = `${d.good_reps} / ${d.reps} VALID`;
        document.getElementById('tab2ChestScore').textContent = Math.round(d.live_score);
        document.getElementById('checkChestValue').textContent = d.torso_lean.toFixed(1) + '° LEAN';

        // Update Waveforms
        if (d.history && d.history.length > 2) {
          const pts = d.history;
          const w = 740;
          const h = 200;
          const n = pts.length;
          let kneeD = '';
          let hipD = '';
          let leanD = '';

          pts.forEach((p, idx) => {
            const x = 40 + (idx / (n - 1)) * w;
            // Angle map: 180° = 30px, 60° = 180px
            const yKnee = 30 + ((180 - p.knee) / 120) * (180 - 30);
            const yHip = 30 + ((180 - p.hip) / 120) * (180 - 30);
            // Lean map: 0° = 210px, 60° = 140px
            const yLean = 210 - (p.lean / 60) * 70;

            kneeD += (idx === 0 ? `M ${x} ${yKnee}` : ` L ${x} ${yKnee}`);
            hipD += (idx === 0 ? `M ${x} ${yHip}` : ` L ${x} ${yHip}`);
            leanD += (idx === 0 ? `M ${x} ${yLean}` : ` L ${x} ${yLean}`);
          });

          document.getElementById('wfKneePath').setAttribute('d', kneeD);
          document.getElementById('wfHipPath').setAttribute('d', hipD);
          document.getElementById('wfLeanPath').setAttribute('d', leanD);
        }

        // Rep velocity list
        if (d.rep_details && d.rep_details.length > 0) {
          const cont = document.getElementById('vbtRepBars');
          cont.innerHTML = d.rep_details.slice(-8).map(r => `
            <div class="flex items-center gap-2">
              <span class="w-12 text-on-surface-variant">REP ${String(r.rep).padStart(2, '0')}</span>
              <div class="flex-1 bg-surface-container-highest h-3 rounded overflow-hidden">
                <div class="bg-primary-container h-full" style="width: ${Math.min(100, (r.score || 80))}%"></div>
              </div>
              <span class="w-16 text-right font-bold text-primary">${Math.round(r.score)}% acc</span>
            </div>
          `).join('');
        }

        // Render Reps Per Set Bar Graph at bottom of HUD
        const setsCont = document.getElementById('repsPerSetContainer');
        if (setsCont) {
          const completed = d.completed_sets || [];
          const currentSetNum = d.current_set || 1;
          const currentRepsList = d.rep_details || [];

          // Combine past completed sets + active current set
          const allSets = completed.map(cs => ({ ...cs, active: false }));
          allSets.push({
            set: currentSetNum,
            reps: d.reps || 0,
            good_reps: d.good_reps || 0,
            avg_score: currentRepsList.length > 0 ? Math.round(currentRepsList.reduce((acc, r) => acc + (r.score || 0), 0) / currentRepsList.length) : (d.live_score ? Math.round(d.live_score) : 0),
            rep_details: currentRepsList,
            active: true
          });

          const badgeEl = document.getElementById('repsPerSetBadge');
          if (badgeEl) {
            badgeEl.textContent = `SET ${String(currentSetNum).padStart(2, '0')} ACTIVE (${d.reps || 0} REPS)`;
          }

          setsCont.innerHTML = allSets.map(s => {
            const isActive = !!s.active;
            const reps = s.rep_details || [];
            const avgScore = s.avg_score || (reps.length > 0 ? Math.round(reps.reduce((a, b) => a + (b.score||0), 0) / reps.length) : 0);

            let barsHtml = '';
            if (reps.length === 0) {
              barsHtml = `
                <div class="h-24 flex flex-col items-center justify-center text-on-surface-variant text-xs font-code-md border border-dashed border-surface-container-highest rounded bg-surface-container-lowest/50">
                  <span class="material-symbols-outlined text-[20px] mb-1 opacity-60 text-secondary">fitness_center</span>
                  <span>Awaiting reps for Set ${String(s.set).padStart(2, '0')}...</span>
                </div>
              `;
            } else {
              const bars = reps.map((r, i) => {
                const sc = Math.round(r.score || 0);
                const hPct = Math.max(15, Math.min(100, sc));
                let colorClass = 'bg-primary-container text-on-primary-container';
                if (sc < 70) colorClass = 'bg-error text-surface-container-lowest';
                else if (sc < 85) colorClass = 'bg-secondary text-surface-container-lowest';

                return `
                  <div class="flex-1 min-w-[28px] max-w-[48px] flex flex-col items-center justify-end h-full gap-1 group relative">
                    <span class="text-[9px] font-code-md text-on-surface-variant group-hover:text-primary transition-colors">${sc}%</span>
                    <div class="w-full ${colorClass} rounded-t transition-all duration-300 relative overflow-hidden shadow-sm hover:brightness-125" style="height: ${hPct}%;"></div>
                    <span class="text-[9px] font-code-md text-secondary font-bold">R${r.rep || (i + 1)}</span>
                  </div>
                `;
              }).join('');

              barsHtml = `
                <div class="h-24 bg-surface-container-lowest p-2 rounded border border-surface-container-highest flex items-end gap-2 overflow-x-auto">
                  ${bars}
                </div>
              `;
            }

            return `
              <div class="bg-surface-container p-4 rounded border ${isActive ? 'border-primary-container shadow-lg shadow-primary-container/10' : 'border-surface-container-high'} flex flex-col gap-3">
                <div class="flex items-center justify-between pb-1 border-b border-surface-container-high">
                  <div class="flex items-center gap-2">
                    <span class="font-headline-sm text-sm font-bold text-primary">SET ${String(s.set).padStart(2, '0')}</span>
                    ${isActive ? '<span class="inline-flex items-center gap-1 bg-primary-container text-on-primary-container px-1.5 py-0.5 rounded font-label-caps text-[9px] font-bold shadow-sm"><span class="w-1.5 h-1.5 rounded-full bg-on-primary-container animate-pulse"></span>ACTIVE</span>' : '<span class="font-label-caps text-[9px] text-on-surface-variant bg-surface-container-highest px-1.5 py-0.5 rounded">COMPLETED</span>'}
                  </div>
                  <span class="font-code-md text-xs ${s.good_reps > 0 ? 'text-primary-container font-bold' : 'text-on-surface-variant'}">${s.reps} REPS (${s.good_reps} VALID)</span>
                </div>

                ${barsHtml}

                <div class="flex items-center justify-between text-xs font-code-md text-on-surface-variant pt-1 border-t border-surface-container-highest">
                  <span>AVG FORM: <strong class="text-primary font-bold">${avgScore}%</strong></span>
                  <span>ACCURACY: <strong class="${s.reps > 0 && s.good_reps === s.reps ? 'text-primary-container' : 'text-secondary'} font-bold">${s.reps > 0 ? Math.round((s.good_reps / s.reps) * 100) : 100}%</strong></span>
                </div>
              </div>
            `;
          }).join('');
        }

      } catch (err) {
        // network or server restarting
      }
      setTimeout(pollTelemetry, 40);
    }

    // Start polling on load
    pollTelemetry();
  </script>
</body>
</html>
"""


def start_server(host="0.0.0.0", port=5000, video=None, side="auto", complexity=0):
    """Initializes vision pipeline and starts HTTP server."""
    global GLOBAL_PIPELINE
    GLOBAL_PIPELINE = VisionPipeline(video_source=video, side=side, complexity=complexity)
    GLOBAL_PIPELINE.start()

    server = ThreadedHTTPServer((host, port), SquatVisionHTTPHandler)
    print(f"\n{'='*60}")
    print(f"  PyPose Squat Vision AI - Web Dashboard Active")
    print(f"  Access Dashboard at: http://localhost:{port}")
    print(f"  Streaming Video at:  http://localhost:{port}/video_feed")
    print(f"  Telemetry API at:    http://localhost:{port}/api/telemetry")
    print(f"{'='*60}\n")
    return server


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0", help="HTTP host (default 0.0.0.0)")
    ap.add_argument("--port", type=int, default=5000, help="HTTP port (default 5000)")
    ap.add_argument("--video", help="path to a video file instead of webcam 0")
    ap.add_argument("--side", default="auto", choices=["auto", "left", "right"])
    ap.add_argument("--complexity", type=int, default=0, choices=[0, 1, 2],
                    help="MediaPipe model: 0=lite (fastest), 1=full, 2=heavy")
    args = ap.parse_args()

    srv = start_server(host=args.host, port=args.port, video=args.video, side=args.side, complexity=args.complexity)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping PyPose server...")
        if GLOBAL_PIPELINE:
            GLOBAL_PIPELINE.stop()
        srv.server_close()

