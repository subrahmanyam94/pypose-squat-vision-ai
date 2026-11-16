"""Pure math helpers: angles, distances, smoothing. No camera or model code here."""
import math


def torso_lean_angle(shoulder_xy, hip_xy):
    """Angle of torso lean from vertical in degrees (0 = upright, 90 = horizontal).

    Uses absolute offsets so it works regardless of coordinate system
    (image coords where y↓ or math coords where y↑).
    """
    dx = abs(shoulder_xy[0] - hip_xy[0])
    dy = abs(shoulder_xy[1] - hip_xy[1])
    if dx < 1e-6 and dy < 1e-6:
        return 0.0
    return math.degrees(math.atan2(dx, dy))


def angle(a, b, c):
    """Angle at point b (degrees) formed by a-b-c. Points are (x, y) tuples."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    norm = math.hypot(*ba) * math.hypot(*bc)
    if norm == 0:
        return 0.0
    cos = max(-1.0, min(1.0, dot / norm))
    return math.degrees(math.acos(cos))


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class EMA:
    """Exponential moving average to remove keypoint jitter."""

    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.value = None

    def update(self, x):
        self.value = x if self.value is None else self.alpha * x + (1 - self.alpha) * self.value
        return self.value

    def reset(self):
        self.value = None
