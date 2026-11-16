"""Tests angle math, squat rep counter, and chest uprightness on synthetic landmarks.
Run:  python test_logic.py
"""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import landmarks as L
from geometry import angle, torso_lean_angle, EMA
from exercises import SquatAnalyzer, depth_score, back_score, chest_score, overall_score


def test_angle():
    assert abs(angle((1, 0), (0, 0), (0, 1)) - 90) < 1e-6
    assert abs(angle((1, 0), (0, 0), (-1, 0)) - 180) < 1e-6
    assert angle((0, 0), (0, 0), (1, 1)) == 0.0


def test_torso_lean_and_chest_score():
    """Torso lean angle and continuous chest uprightness scoring."""
    # Vertical torso: shoulder directly above hip
    assert torso_lean_angle((100, 50), (100, 150)) == 0.0
    assert chest_score(0.0) == 100.0
    assert chest_score(20.0) == 100.0
    
    # Moderate athletic incline (40 deg)
    assert 75.0 <= chest_score(40.0) <= 100.0
    
    # Severe chest forward collapse (68 deg)
    assert chest_score(68.0) < 30.0
    
    # Horizontal torso (90 deg)
    assert chest_score(90.0) == 0.0


def test_continuous_scores():
    """Score functions return sensible values at key angles."""
    # depth_score: perfect at 90, decent at 100, bad at 150, zero at 170
    assert depth_score(90) == 100
    assert 60 < depth_score(100) < 90
    assert 10 <= depth_score(140) < 50
    assert depth_score(170) < 5

    # back_score: perfect when upright, bad when bent far forward
    assert back_score(150) == 100
    assert 60 < back_score(100) < 100
    assert back_score(40) < 15

    # overall blends depth + chest uprightness + back posture
    perfect = overall_score(85, 140, torso_lean=20.0)
    assert perfect >= 90
    bad = overall_score(140, 40, torso_lean=65.0)
    assert bad < 25


def pose(knee_deg, torso_lean_deg=25.0, vis=0.95):
    """Side-view left-leg landmarks with the given knee angle and torso lean."""
    ankle = (0.0, 0.0)
    knee = (0.0, -100.0)                       # shin length 100, pointing "up"
    a = math.radians(knee_deg)
    v = (0.0, 1.0)                             # knee->ankle direction (down)
    hip = (knee[0] + 100 * (v[0] * math.cos(a) - v[1] * math.sin(a)),
           knee[1] + 100 * (v[0] * math.sin(a) + v[1] * math.cos(a)))
    
    # Shoulder positioned above hip with torso lean from vertical
    lean_rad = math.radians(torso_lean_deg)
    sh = (hip[0] + 100 * math.sin(lean_rad), hip[1] - 100 * math.cos(lean_rad))
    
    lm = {i: (0.0, 0.0, vis) for i in range(33)}
    lm[L.L_ANKLE], lm[L.L_KNEE], lm[L.L_HIP], lm[L.L_SHOULDER] = (
        (*ankle, vis), (*knee, vis), (*hip, vis), (*sh, vis))
    for r in (L.R_ANKLE, L.R_KNEE, L.R_HIP, L.R_SHOULDER):
        lm[r] = (0.0, 0.0, 0.1)  # right side hidden -> auto picks left
    return lm


def test_pose_builder():
    lm = pose(120, torso_lean_deg=25.0)
    k = angle(lm[L.L_HIP][:2], lm[L.L_KNEE][:2], lm[L.L_ANKLE][:2])
    lean = torso_lean_angle(lm[L.L_SHOULDER][:2], lm[L.L_HIP][:2])
    assert abs(k - 120) < 0.5 and abs(lean - 25.0) < 0.5, (k, lean)


def squat_cycle(min_knee, torso_lean=25.0, steps=30):
    down = [170 - (170 - min_knee) * i / steps for i in range(steps + 1)]
    return [(k, torso_lean) for k in down + [min_knee] * 8 + down[::-1]]


def run(cycles, noise=0.0):
    an = SquatAnalyzer()
    import random
    random.seed(0)
    for k, lean in cycles:
        an.update(pose(k + random.uniform(-noise, noise), torso_lean_deg=lean))
    return an


def test_counts_five_reps():
    seq = []
    for _ in range(5):
        seq += squat_cycle(80, torso_lean=22.0)
    an = run(seq)
    summ = an.summary()
    assert summ["reps"] == 5, summ
    assert summ["good_reps"] == 5, summ


def test_noise_does_not_double_count():
    seq = []
    for _ in range(4):
        seq += squat_cycle(85, torso_lean=25.0)
    an = run(seq, noise=6)
    assert an.summary()["reps"] == 4, an.summary()


def test_shallow_reps_not_counted_or_low_score():
    # Never goes below 100 deg -> not a rep at all
    an = run(squat_cycle(115) * 3)
    assert an.summary()["reps"] == 0
    # Reaches 98: counts, decent continuous score
    an = run(squat_cycle(98) * 2)
    s = an.summary()
    assert s["reps"] == 2, s
    assert 70 < s["avg_form_score"] < 98, s


def test_forward_lean_and_chest_upright_flagged():
    # Heavy forward lean (65 deg) -> chest drop flagged
    an = run(squat_cycle(80, torso_lean=65.0) * 2)
    s = an.summary()
    assert s["reps"] == 2 and s["good_reps"] == 0, s
    assert s["avg_form_score"] < 70, s


def test_perfect_form_high_score():
    """Perfect depth + upright chest (20 deg) should yield score >= 85."""
    an = run(squat_cycle(85, torso_lean=20.0) * 3)
    s = an.summary()
    assert s["reps"] == 3 and s["avg_form_score"] >= 85, s


def test_live_score_and_chest_updates():
    """The live_score, chest_upright, and torso_lean fields should be populated."""
    an = SquatAnalyzer()
    st, _ = an.update(pose(90, torso_lean_deg=22.0))
    assert st.live_score > 0
    assert st.knee_angle < 100
    assert st.chest_upright is True
    assert abs(st.torso_lean - 22.0) < 1.0


def test_low_visibility_pauses():
    an = SquatAnalyzer()
    st, ok = an.update(pose(90, vis=0.2))
    assert not ok and "visible" in st.feedback
    st, ok = an.update(None)
    assert not ok


def test_set_counting():
    """Sets count up when advancing sets and preserve completed set summaries."""
    seq = []
    for _ in range(3):
        seq += squat_cycle(85, torso_lean=22.0)
    an = run(seq)
    assert an.state.reps == 3
    assert an.state.current_set == 1

    # Advance to next set
    an.next_set()
    assert an.state.current_set == 2
    assert an.state.reps == 0
    assert len(an.state.completed_sets) == 1
    assert an.state.completed_sets[0]["reps"] == 3

    # Do 2 more reps in set 2
    for _ in range(2):
        for k, lean in squat_cycle(85, torso_lean=22.0):
            an.update(pose(k, torso_lean_deg=lean))
    assert an.state.reps == 2
    summ = an.summary()
    assert summ["sets"] == 2
    assert summ["total_reps"] == 5


def test_standing_knee_bend_not_counted():
    """Standing knee bends or leg kicks without hip drop should NOT be counted as squats."""
    an = SquatAnalyzer()
    # Baseline standing upright
    an.update(pose(170))
    # Fake knee bend: knee angle drops to 85, but hip Y stays completely stationary (no descent)
    fake_frames = []
    for k in [170, 150, 120, 85, 85, 120, 150, 170]:
        lm = pose(k)
        # Force hip to stay at standing height (-198.5) and torso vertical
        lm[L.L_HIP] = (lm[L.L_HIP][0], -198.48, 0.95)
        fake_frames.append(lm)

    for lm in fake_frames:
        an.update(lm)

    assert an.summary()["reps"] == 0, f"Standing knee bend should not count, got {an.summary()['reps']}"


def test_two_minute_rest_between_sets():
    """Verify 2-minute rest gap between sets:
    1. If user squats within 2 minutes, previous set is continued (not next set).
    2. If 2 minutes elapse without squatting, it auto-advances to next set.
    3. Manual next_set() advances immediately.
    4. select_set(target) allows reviving / continuing any set.
    """
    import time
    an = SquatAnalyzer()
    for _ in range(3):
        for k, lean in squat_cycle(85, torso_lean=22.0):
            an.update(pose(k, torso_lean_deg=lean))

    assert an.state.reps == 3
    assert an.state.current_set == 1

    # 1. User rests for 10 seconds (within 2-min gap window)
    an.stop_set_and_begin_rest()
    assert an.state.is_resting is True
    assert an.state.current_set == 1
    assert an.state.reps == 3

    # User squats within 2 minutes -> CONTINUES PREVIOUS SET (Set 1)!
    for k, lean in squat_cycle(85, torso_lean=22.0):
        an.update(pose(k, torso_lean_deg=lean))

    assert an.state.is_resting is False
    assert an.state.current_set == 1, "Must stay on Set 1 within 2 min gap"
    assert an.state.reps == 4, f"Must continue previous set reps to 4, got {an.state.reps}"

    # 2. 2-minute gap expires: simulate 121 seconds passing
    an.stop_set_and_begin_rest()
    an.state.rest_start_time = time.time() - 121.0
    an.update(pose(170))  # standing upright frame
    assert an.state.current_set == 2, "Must auto-advance to Set 2 after 2 mins"
    assert an.state.reps == 0
    assert len(an.state.completed_sets) == 1
    assert an.state.completed_sets[0]["set"] == 1
    assert an.state.completed_sets[0]["reps"] == 4

    # 3. User does 2 reps in Set 2
    for _ in range(2):
        for k, lean in squat_cycle(85, torso_lean=22.0):
            an.update(pose(k, torso_lean_deg=lean))
    assert an.state.current_set == 2
    assert an.state.reps == 2

    # 4. User selects Set 1 to continue Set 1!
    an.select_set(1)
    assert an.state.current_set == 1
    assert an.state.reps == 4, "Must revive Set 1 with its 4 reps"
    # Set 2 is saved to completed_sets
    assert len(an.state.completed_sets) == 1
    assert an.state.completed_sets[0]["set"] == 2
    assert an.state.completed_sets[0]["reps"] == 2

    # User does another rep in Set 1 -> rep 5 of Set 1!
    for k, lean in squat_cycle(85, torso_lean=22.0):
        an.update(pose(k, torso_lean_deg=lean))
    assert an.state.reps == 5
    assert an.state.current_set == 1
    summ = an.summary()
    assert summ["total_reps"] == 7  # 5 in Set 1 + 2 in Set 2


def test_simple_knee_bend_rejected():
    """Verify that forward knee bending without hip hinge/sitting back is strictly rejected."""
    an = SquatAnalyzer()
    # Baseline standing upright
    an.update(pose(170))
    # Fake knee bend: knees push forward to 90 deg, but torso stays vertical and hips stay high (no hinge)
    frames = []
    for k in [170, 150, 130, 90, 90, 130, 150, 170]:
        lm = {i: (0.0, 0.0, 0.95) for i in range(33)}
        lm[L.L_ANKLE] = (0.0, 0.0, 0.95)
        lm[L.L_KNEE] = (70.7, -70.7, 0.95)
        lm[L.L_HIP] = (0.0, -141.4, 0.95)
        lm[L.L_SHOULDER] = (0.0, -241.4, 0.95)
        for r in (L.R_ANKLE, L.R_KNEE, L.R_HIP, L.R_SHOULDER):
            lm[r] = (0.0, 0.0, 0.1)
        frames.append(lm)

    for lm in frames:
        an.update(lm)

    assert an.summary()["reps"] == 0, f"Simple knee bend must be rejected, got {an.summary()['reps']}"


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for f in fns:
        f()
        print("PASS", f.__name__)
