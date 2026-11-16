"""Wraps MediaPipe PoseLandmarker (Tasks API). Swap this file out to use YOLO/MoveNet later.

The pose model file (~6 MB for "lite", ~9 MB "full") is downloaded once on first run.
"""
import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from landmarks import *  # noqa: F401,F403  (index constants + POSE_CONNECTIONS)

MODEL_URLS = {
    0: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    1: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    2: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def ensure_model(complexity):
    url = MODEL_URLS[complexity]
    path = os.path.join(MODEL_DIR, os.path.basename(url))
    if not os.path.exists(path):
        os.makedirs(MODEL_DIR, exist_ok=True)
        print(f"Downloading pose model to {path} ...")
        urllib.request.urlretrieve(url, path)
    return path


class PoseDetector:
    def __init__(self, complexity=1, min_det=0.5, min_track=0.5, model_path=None):
        model_path = model_path or ensure_model(complexity)
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_det,
            min_tracking_confidence=min_track,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self._t0 = time.monotonic()
        self._last_ts = -1
        self.result = None

    def process(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int((time.monotonic() - self._t0) * 1000)
        ts = max(ts, self._last_ts + 1)        # timestamps must strictly increase
        self._last_ts = ts
        self.result = self.landmarker.detect_for_video(image, ts)
        return bool(self.result.pose_landmarks)

    def landmarks(self, w, h):
        """Return {index: (x_px, y_px, visibility)} or None."""
        if not self.result or not self.result.pose_landmarks:
            return None
        return {
            i: (lm.x * w, lm.y * h, lm.visibility)
            for i, lm in enumerate(self.result.pose_landmarks[0])
        }

    def draw(self, frame_bgr, color=(0, 255, 0)):
        h, w = frame_bgr.shape[:2]
        lm = self.landmarks(w, h)
        if not lm:
            return
        for a, b in POSE_CONNECTIONS:
            if lm[a][2] > 0.5 and lm[b][2] > 0.5:
                cv2.line(frame_bgr, (int(lm[a][0]), int(lm[a][1])),
                         (int(lm[b][0]), int(lm[b][1])), color, 3)
        for x, y, v in lm.values():
            if v > 0.5:
                cv2.circle(frame_bgr, (int(x), int(y)), 4, (255, 255, 255), -1)

    def close(self):
        self.landmarker.close()
