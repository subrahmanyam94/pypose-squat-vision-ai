"""Rep counting (state machine with hysteresis), squat form rules, and chest uprightness analysis.

Scoring is *continuous* 0-100 at every angle, not binary pass/fail.
Evaluates depth, chest/torso uprightness, and back posture.
"""
from dataclasses import dataclass, field
import time
from geometry import angle, torso_lean_angle, EMA, distance
import landmarks as D

MIN_VIS = 0.5


def _clamp01(x):
    return max(0.0, min(1.0, x))


def depth_score(knee_angle: float) -> float:
    """Continuous 0-100 squat depth score based on knee flexion angle.

    Perfect squat (parallel or below, ≤90°) → 100
    Good depth   (90-95°)                    → 90-100  (linear blend)
    Acceptable   (95-110°)                   → 50-90   (linear blend)
    Shallow      (110-140°)                  → 10-50   (linear blend)
    Standing     (≥160°)                     → 0
    """
    if knee_angle <= 90:
        return 100.0
    if knee_angle <= 95:
        return 90 + (95 - knee_angle) / 5 * 10          # 90-100
    if knee_angle <= 110:
        return 50 + (110 - knee_angle) / 15 * 40        # 50-90
    if knee_angle <= 140:
        return 10 + (140 - knee_angle) / 30 * 40        # 10-50
    if knee_angle <= 160:
        return (160 - knee_angle) / 20 * 10             # 0-10
    return 0.0


def chest_score(torso_lean: float) -> float:
    """Continuous 0-100 chest uprightness score based on torso lean from vertical.
    0° - 35° lean   → 100    (optimal upright chest)
    35° - 50° lean  → 75-100 (acceptable athletic squat incline)
    50° - 60° lean  → 35-75  (moderate lean, borderline chest drop)
    60° - 75° lean  → 0-35   (excessive forward lean / chest collapsed)
    > 75° lean      → 0
    """
    if torso_lean <= 35.0:
        return 100.0
    if torso_lean <= 50.0:
        return 75.0 + (50.0 - torso_lean) / 15.0 * 25.0
    if torso_lean <= 60.0:
        return 35.0 + (60.0 - torso_lean) / 10.0 * 40.0
    if torso_lean <= 75.0:
        return max(0.0, (75.0 - torso_lean) / 15.0 * 35.0)
    return 0.0


def back_score(back_angle: float) -> float:
    """Continuous 0-100 back posture score (shoulder-hip-knee angle).

    Upright  (≥120°) → 100
    Slightly forward (80-120°) → 50-100
    Too far forward  (50-80°)  → 10-50
    Extreme lean (<50°) → 0-10
    """
    if back_angle >= 120:
        return 100.0
    if back_angle >= 80:
        return 50 + (back_angle - 80) / 40 * 50         # 50-100
    if back_angle >= 50:
        return 10 + (back_angle - 50) / 30 * 40         # 10-50
    if back_angle >= 30:
        return (back_angle - 30) / 20 * 10              # 0-10
    return 0.0


def overall_score(knee_angle: float, back_angle: float, torso_lean: float = 25.0) -> float:
    """Weighted composite accuracy score: 50% depth + 30% chest uprightness + 20% back posture.
    Returns 0-100.
    """
    d_sc = depth_score(knee_angle)
    c_sc = chest_score(torso_lean)
    b_sc = back_score(back_angle)
    total = 0.50 * d_sc + 0.30 * c_sc + 0.20 * b_sc
    return round(max(0.0, min(100.0, total)), 1)


@dataclass
class SquatState:
    stage: str = "up"            # "up", "descending", "down", "ascent"
    reps: int = 0
    feedback: str = "Get in frame"
    good_reps: int = 0
    rep_scores: list = field(default_factory=list)
    rep_details: list = field(default_factory=list)
    
    # --- Set & Rest tracking ---
    current_set: int = 1
    total_reps: int = 0
    completed_sets: list = field(default_factory=list)
    is_resting: bool = False
    rest_time_total: float = 120.0       # 2 minutes rest between sets
    rest_time_remaining: float = 0.0     # countdown in seconds
    rest_start_time: float = 0.0
    
    # --- live continuous metrics (exposed for UI) ---
    live_score: float = 0.0
    live_depth_score: float = 0.0
    live_chest_score: float = 0.0
    live_back_score: float = 0.0
    knee_angle: float = 180.0
    hip_angle: float = 180.0
    back_angle: float = 180.0
    torso_lean: float = 0.0
    ankle_angle: float = 85.0
    chest_upright: bool = True
    chest_status: str = "UPRIGHT"
    depth_pct: float = 0.0       # 0 = standing, 1 = deepest squat
    
    # --- Biomechanics telemetry ---
    velocity: float = 0.0        # m/s estimated
    peak_velocity: float = 0.0
    bar_drift_cm: float = 0.0
    power_watts: float = 0.0
    
    # --- per-rep tracking ---
    _min_knee: float = 180.0
    _max_torso_lean: float = 0.0
    _min_back: float = 180.0
    _min_thigh_level: float = 1.0  # (knee_y - hip_y) / femur_len: 1.0 = standing, <= 0.30 = parallel
    _best_score: float = 0.0
    _max_hip_drop: float = 0.0
    _start_time: float = field(default_factory=time.time)
    _rep_start_time: float = 0.0
    _last_rep_finish_time: float = 0.0


class SquatAnalyzer:
    DOWN_THRESH = 100       # knee angle below this = bottom of squat (strict depth)
    UP_THRESH = 160         # knee angle above this = standing upright
    DEPTH_GOAL = 90         # knee angle needed for a "deep enough" parallel rep
    BACK_MIN = 45           # shoulder-hip-knee angle below this = folding forward
    MAX_HIP_ANGLE = 102.0   # max acceptable hip angle at bottom (must sit hips back)
    MAX_THIGH_LEVEL = 0.30  # max vertical knee-hip distance / femur length (thigh near horizontal)
    MIN_HIP_DROP = 0.38     # min vertical hip descent / femur length
    CHEST_LEAN_MIN = 12.0   # min torso lean at bottom (prevents standing knee bends)
    CHEST_LEAN_MAX = 50.0   # max acceptable torso lean from vertical (degrees)
    REST_TIME_FOR_NEW_SET = 120.0  # 2 minutes rest between sets
    INACTIVITY_BEFORE_REST = 4.0   # seconds standing upright after reps before 2m rest gap begins

    def __init__(self, side="auto"):
        self.side = side
        self.knee_ema = EMA(0.3)
        self.back_ema = EMA(0.3)
        self.lean_ema = EMA(0.3)
        self.hip_ema = EMA(0.3)
        self.state = SquatState()
        self._last_knee_y = None
        self._last_time = time.time()
        self._standing_hip_y = None

    def _pick_side(self, lm):
        if self.side in ("left", "right"):
            return self.side
        left_vis = sum(lm[i][2] for i in (D.L_SHOULDER, D.L_HIP, D.L_KNEE, D.L_ANKLE))
        right_vis = sum(lm[i][2] for i in (D.R_SHOULDER, D.R_HIP, D.R_KNEE, D.R_ANKLE))
        return "left" if left_vis >= right_vis else "right"

    def update(self, lm):
        """lm: {index: (x, y, vis)} or None. Returns (state, colour_ok)."""
        s = self.state
        now = time.time()
        dt = max(now - self._last_time, 1e-4)
        self._last_time = now

        if lm is None:
            s.feedback = "Step back so your whole body is visible"
            s.live_score = 0
            return s, False

        side = self._pick_side(lm)
        idx = (
            (D.L_SHOULDER, D.L_HIP, D.L_KNEE, D.L_ANKLE)
            if side == "left"
            else (D.R_SHOULDER, D.R_HIP, D.R_KNEE, D.R_ANKLE)
        )
        sh, hip, knee, ank = (lm[i] for i in idx)
        if min(p[2] for p in (sh, hip, knee, ank)) < MIN_VIS:
            s.feedback = "Step back so your whole body is visible"
            s.live_score = 0
            return s, False

        raw_knee = angle(hip[:2], knee[:2], ank[:2])
        raw_back = angle(sh[:2], hip[:2], knee[:2])
        raw_lean = torso_lean_angle(sh[:2], hip[:2])

        knee_ang = self.knee_ema.update(raw_knee)
        back_ang = self.back_ema.update(raw_back)
        lean_ang = self.lean_ema.update(raw_lean)
        hip_ang = self.hip_ema.update(raw_back)

        # Approximate velocity from hip/knee movement
        if self._last_knee_y is not None:
            dy_px = abs(hip[1] - self._last_knee_y)
            torso_px = max(abs(hip[1] - sh[1]), 50.0)
            scale_m = 0.50 / torso_px
            v = (dy_px * scale_m) / dt
            s.velocity = round(min(2.5, v), 2)
            s.peak_velocity = max(s.peak_velocity, s.velocity)
            s.power_watts = round(s.velocity * 850.0, 1)
        self._last_knee_y = hip[1]

        # Barbell / shoulder sagittal drift approximation in cm
        drift = (sh[0] - ank[0]) * 0.15
        s.bar_drift_cm = round(max(-6.0, min(6.0, drift)), 1)

        # --- Continuous scoring ---
        s.knee_angle = round(knee_ang, 1)
        s.back_angle = round(back_ang, 1)
        s.torso_lean = round(lean_ang, 1)
        s.hip_angle = round(hip_ang, 1)
        
        s.live_depth_score = depth_score(knee_ang)
        s.live_chest_score = chest_score(lean_ang)
        s.live_back_score = back_score(back_ang)
        s.live_score = overall_score(knee_ang, back_ang, lean_ang)
        s.depth_pct = _clamp01((180.0 - knee_ang) / 95.0)

        # Chest upright validation
        s.chest_upright = (lean_ang <= self.CHEST_LEAN_MAX) and (back_ang >= self.BACK_MIN)
        if lean_ang <= 32.0:
            s.chest_status = "UPRIGHT (OPTIMAL)"
        elif lean_ang <= self.CHEST_LEAN_MAX:
            s.chest_status = "NEUTRAL INCLINE"
        else:
            s.chest_status = "LEANING FORWARD"

        ok = True
        s.feedback = "Good"

        # Calibrate standing hip height when upright
        if knee_ang > 145:
            if self._standing_hip_y is None:
                self._standing_hip_y = hip[1]
            elif s.stage == "up":
                self._standing_hip_y = 0.95 * self._standing_hip_y + 0.05 * hip[1]
        elif self._standing_hip_y is None:
            self._standing_hip_y = hip[1]

        femur_len = distance(hip[:2], knee[:2])
        if femur_len < 10.0:
            femur_len = max(abs(hip[1] - knee[1]), 50.0)

        hip_drop = (hip[1] - self._standing_hip_y) if self._standing_hip_y is not None else 0.0
        hip_drop_ratio = (hip_drop / femur_len) if femur_len > 0 else 0.0
        # Ratio of vertical distance between knee and hip to femur length
        # When standing: (knee[1] - hip[1]) / femur_len ≈ 0.95-1.0
        # At parallel squat depth: (knee[1] - hip[1]) / femur_len ≈ 0.0 to 0.25
        thigh_level = (knee[1] - hip[1]) / femur_len if femur_len > 0 else 1.0

        # Check if currently in 2-minute rest gap window
        if s.is_resting:
            elapsed = now - s.rest_start_time
            s.rest_time_remaining = max(0.0, s.rest_time_total - elapsed)
            # If 2 minutes of rest expire without squats -> automatically finalize set & advance
            if s.rest_time_remaining <= 0.0:
                self.finish_set_and_advance()
            # If user begins squatting down within 2 minutes (< DOWN_THRESH) -> CONTINUE PREVIOUS SET!
            elif knee_ang < self.DOWN_THRESH:
                s.is_resting = False
                s.rest_time_remaining = 0.0
                s.stage = "down"
                s._rep_start_time = now
                s.feedback = f"Continuing Set {s.current_set:02d}!"
            else:
                mins = int(s.rest_time_remaining // 60)
                secs = int(s.rest_time_remaining % 60)
                s.feedback = f"REST GAP: {mins:02d}:{secs:02d} (Set {s.current_set:02d} - squat within 2m to continue)"
                if knee_ang > 155:
                    if self._standing_hip_y is None:
                        self._standing_hip_y = hip[1]
                    else:
                        self._standing_hip_y = 0.95 * self._standing_hip_y + 0.05 * hip[1]
                return s, True

        # Track per-rep extremes while descending
        if s.stage in ("down", "ascent") or knee_ang < self.UP_THRESH:
            s._min_knee = min(s._min_knee, knee_ang)
            s._min_back = min(s._min_back, back_ang)
            s._max_torso_lean = max(s._max_torso_lean, lean_ang)
            s._best_score = max(s._best_score, s.live_score)
            s._max_hip_drop = max(s._max_hip_drop, hip_drop_ratio)
            s._min_thigh_level = min(s._min_thigh_level, thigh_level)

        # Live chest upright check
        if not s.chest_upright and knee_ang < self.UP_THRESH:
            s.feedback = "Chest up! Keep torso upright"
            ok = False

        # State machine with hysteresis (strict depth: only full squats count)
        if s.stage == "up" and knee_ang < self.DOWN_THRESH:
            s.stage = "down"
            s._rep_start_time = now
        elif s.stage == "down" and knee_ang > self.DOWN_THRESH + 10:
            s.stage = "ascent"
        elif s.stage in ("down", "ascent") and knee_ang > self.UP_THRESH:
            s.stage = "up"
            self._finish_rep()

        # Auto-trigger 2-min rest gap window if user completed reps and stayed standing for INACTIVITY_BEFORE_REST seconds
        # Keep current set active so if user squats within 2 minutes, previous set continues!
        if s.stage == "up" and s.reps > 0 and s._last_rep_finish_time > 0 and not s.is_resting:
            if (now - s._last_rep_finish_time) >= self.INACTIVITY_BEFORE_REST:
                s.is_resting = True
                s.rest_start_time = s._last_rep_finish_time
                s.rest_time_remaining = max(0.0, self.REST_TIME_FOR_NEW_SET - (now - s.rest_start_time))

        if s.stage == "down" and knee_ang > self.DEPTH_GOAL and ok:
            s.feedback = "Go deeper"
            ok = False

        s.feedback = s.feedback if s.reps or s.stage in ("down", "ascent") or not ok else "Ready"
        return s, ok

    def _finish_rep(self):
        s = self.state
        # Strict validation for a genuine, perfect squat:
        # 1. Depth: knee angle must reach parallel depth (<= DOWN_THRESH, 98 deg)
        # 2. Hip Hinge: hips must sit back and hinge (min_back <= MAX_HIP_ANGLE, 102 deg)
        # 3. Thigh Parallel: hip must drop level with or near knees (min_thigh_level <= 0.30)
        # 4. Vertical Hip Descent: hips must physically descend >= 38% of femur length
        # 5. Torso Incline: natural athletic lean (12 - 50 deg), not standing straight (<12) or collapsed (>50)
        
        depth_achieved = s._min_knee <= self.DOWN_THRESH
        hip_hinged = s._min_back <= self.MAX_HIP_ANGLE
        thigh_parallel = s._min_thigh_level <= self.MAX_THIGH_LEVEL
        hip_dropped = s._max_hip_drop >= self.MIN_HIP_DROP

        if not depth_achieved:
            s.feedback = f"Squat deeper: thighs to parallel (knee {s._min_knee:.0f}° > {self.DOWN_THRESH}°)"
            self._reset_rep_tracking()
            return

        if not (hip_hinged and thigh_parallel and hip_dropped and s._max_torso_lean >= self.CHEST_LEAN_MIN):
            if not hip_hinged or s._max_torso_lean < self.CHEST_LEAN_MIN:
                s.feedback = "Sit hips back into squat — don't just bend knees!"
            elif not thigh_parallel:
                s.feedback = "Drop hips lower until thighs are parallel to ground"
            else:
                s.feedback = "Lower your hips down into the squat"
            self._reset_rep_tracking()
            return

        s.reps += 1
        s.total_reps += 1
        s._last_rep_finish_time = time.time()
        score = overall_score(s._min_knee, s._min_back, s._max_torso_lean)
        s.rep_scores.append(score)
        
        rep_info = {
            "rep": s.reps,
            "set": s.current_set,
            "score": score,
            "min_knee": round(s._min_knee, 1),
            "max_torso_lean": round(s._max_torso_lean, 1),
            "min_hip": round(s._min_back, 1),
            "chest_passed": s._max_torso_lean <= self.CHEST_LEAN_MAX,
            "depth_passed": s._min_knee <= self.DEPTH_GOAL,
            "mpv": round(s.velocity, 2)
        }
        s.rep_details.append(rep_info)

        if score >= 75 and rep_info["chest_passed"] and rep_info["depth_passed"]:
            s.good_reps += 1
            s.feedback = f"Perfect Squat! (Score {score:.0f})"
        elif not rep_info["chest_passed"]:
            s.feedback = f"Chest drop flagged (Score {score:.0f})"
        else:
            s.feedback = f"Good Rep {s.reps} (Score {score:.0f})"
            
        self._reset_rep_tracking()

    def _reset_rep_tracking(self):
        s = self.state
        s._min_knee = 180.0
        s._min_back = 180.0
        s._max_torso_lean = 0.0
        s._min_thigh_level = 1.0
        s._best_score = 0.0
        s._max_hip_drop = 0.0

    def finish_set_and_advance(self):
        """Archives current set and advances to the next set."""
        s = self.state
        if s.reps > 0:
            set_summary = {
                "set": s.current_set,
                "reps": s.reps,
                "good_reps": s.good_reps,
                "avg_score": round(sum(s.rep_scores) / len(s.rep_scores)) if s.rep_scores else 0,
                "rep_details": list(s.rep_details)
            }
            existing_idx = next((i for i, cs in enumerate(s.completed_sets) if cs["set"] == s.current_set), None)
            if existing_idx is not None:
                s.completed_sets[existing_idx] = set_summary
            else:
                s.completed_sets.append(set_summary)
            s.completed_sets.sort(key=lambda x: x["set"])

            max_set = max([cs["set"] for cs in s.completed_sets], default=s.current_set)
            s.current_set = max(s.current_set + 1, max_set + 1)
            s.reps = 0
            s.good_reps = 0
            s.rep_scores = []
            s.rep_details = []
            s._last_rep_finish_time = 0.0
            s.feedback = f"Set {set_summary['set']} completed! Next: Set {s.current_set:02d}"
        else:
            s.current_set += 1
            s.feedback = f"Advanced to Set {s.current_set:02d}"

        s.is_resting = False
        s.rest_time_remaining = 0.0
        self._recalc_total_reps()
        return s

    def select_set(self, target_set: int):
        """Selects a specific set to continue. Can revive a completed set or jump to a set."""
        s = self.state
        target_set = max(1, int(target_set))
        if target_set == s.current_set:
            s.is_resting = False
            s.rest_time_remaining = 0.0
            s.feedback = f"Continuing Set {s.current_set:02d} ({s.reps} reps)"
            return s

        # If current set has reps, archive it first
        if s.reps > 0:
            cur_summary = {
                "set": s.current_set,
                "reps": s.reps,
                "good_reps": s.good_reps,
                "avg_score": round(sum(s.rep_scores) / len(s.rep_scores)) if s.rep_scores else 0,
                "rep_details": list(s.rep_details)
            }
            existing_idx = next((i for i, cs in enumerate(s.completed_sets) if cs["set"] == s.current_set), None)
            if existing_idx is not None:
                s.completed_sets[existing_idx] = cur_summary
            else:
                s.completed_sets.append(cur_summary)

        # Check if target_set is in completed_sets
        matched_idx = next((i for i, cs in enumerate(s.completed_sets) if cs["set"] == target_set), None)
        if matched_idx is not None:
            target = s.completed_sets.pop(matched_idx)
            s.current_set = target["set"]
            s.reps = target["reps"]
            s.good_reps = target["good_reps"]
            s.rep_scores = [d["score"] for d in target.get("rep_details", [])]
            s.rep_details = list(target.get("rep_details", []))
        else:
            s.current_set = target_set
            s.reps = 0
            s.good_reps = 0
            s.rep_scores = []
            s.rep_details = []

        s.completed_sets.sort(key=lambda x: x["set"])
        s.is_resting = False
        s.rest_time_remaining = 0.0
        s._last_rep_finish_time = time.time()
        s.feedback = f"Continuing Set {s.current_set:02d} ({s.reps} reps)"
        self._recalc_total_reps()
        return s

    def resume_set(self):
        """Cancels rest and resumes current active set immediately."""
        s = self.state
        s.is_resting = False
        s.rest_time_remaining = 0.0
        s.feedback = f"Resumed Set {s.current_set:02d} ({s.reps} reps)"
        return s

    def stop_set_and_begin_rest(self):
        """Manually trigger rest gap."""
        s = self.state
        s.is_resting = True
        s.rest_start_time = time.time()
        s.rest_time_remaining = self.REST_TIME_FOR_NEW_SET
        s.feedback = f"Set {s.current_set:02d} resting (2:00 gap)..."
        return s

    def end_rest_and_start_set(self):
        """Ends rest period."""
        return self.resume_set()

    def next_set(self):
        """Manual next set: finishes current set and advances to the next set."""
        return self.finish_set_and_advance()

    def _recalc_total_reps(self):
        s = self.state
        s.total_reps = sum(cs.get("reps", 0) for cs in s.completed_sets) + s.reps

    def summary(self):
        s = self.state
        self._recalc_total_reps()
        all_rep_details = [d for cs in s.completed_sets for d in cs.get("rep_details", [])] + list(s.rep_details)
        all_scores = [d["score"] for d in all_rep_details if "score" in d]
        avg = sum(all_scores) / len(all_scores) if all_scores else 0
        total_good = sum(cs.get("good_reps", 0) for cs in s.completed_sets) + s.good_reps
        total_sets = max(s.current_set, len(s.completed_sets) + (1 if s.reps > 0 else 0))
        return {
            "sets": total_sets,
            "current_set": s.current_set,
            "reps": s.reps if not s.completed_sets else s.total_reps,
            "current_set_reps": s.reps,
            "total_reps": s.total_reps,
            "good_reps": total_good,
            "avg_form_score": round(avg),
            "completed_sets": s.completed_sets,
            "rep_details": s.rep_details,
            "is_resting": s.is_resting,
            "rest_time_remaining": round(s.rest_time_remaining, 1)
        }
