"""Premium HUD overlay rendering for the squat tracker.

All drawing lives here so main.py stays clean.
Uses only OpenCV + numpy (no extra deps).
"""
import math
import cv2
import numpy as np


# ─── colour palette ──────────────────────────────────────────────────────────
BG_DARK   = (20, 20, 25)
BG_PANEL  = (28, 31, 41)        # matches web theme #1c1f29
ACCENT    = (20, 255, 57)       # neon lime #39ff14 (BGR: 20, 255, 57)
SECONDARY = (246, 215, 76)      # cyan #4cd7f6 (BGR)
WHITE     = (240, 240, 245)
GREY      = (140, 140, 150)
RED       = (60, 60, 255)
GREEN     = (20, 255, 57)
YELLOW    = (50, 220, 240)
SHADOW    = (10, 14, 23)

# Panel constants
PANEL_W   = 340
PANEL_ALPHA = 0.82


# ─── helpers ──────────────────────────────────────────────────────────────────
def _lerp_color(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def score_color(score: float):
    """Red→Yellow→Green gradient based on 0-100 score."""
    if score < 50:
        return _lerp_color(RED, YELLOW, score / 50)
    return _lerp_color(YELLOW, GREEN, (score - 50) / 50)


def _put_text(img, text, pos, scale=0.55, color=WHITE, thick=1, shadow=False,
              font=cv2.FONT_HERSHEY_SIMPLEX):
    """Draw text. Shadow disabled by default for performance."""
    if shadow:
        cv2.putText(img, text, (pos[0]+1, pos[1]+1), font, scale, SHADOW, thick+2, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, scale, color, thick, cv2.LINE_AA)


# ─── arc gauge ────────────────────────────────────────────────────────────────
def draw_arc_gauge(img, cx, cy, radius, score, label="SCORE", thickness=10):
    """Draw a 270° arc gauge with the score value in the centre."""
    start_angle = 135
    sweep = 270
    filled = sweep * (max(0.0, min(100.0, score)) / 100.0)

    color = score_color(score)

    # background arc
    cv2.ellipse(img, (cx, cy), (radius, radius), 0,
                start_angle, start_angle + sweep, (40, 45, 55), thickness, cv2.LINE_AA)
    # filled arc (glow removed for performance)
    if filled > 0:
        cv2.ellipse(img, (cx, cy), (radius, radius), 0,
                    start_angle, start_angle + filled, color, thickness, cv2.LINE_AA)
    
    # score text
    txt = f"{int(score)}"
    sz = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)[0]
    _put_text(img, txt, (cx - sz[0]//2, cy + sz[1]//2 - 2), 1.3, color, 3)
    # label
    lsz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
    _put_text(img, label, (cx - lsz[0]//2, cy + radius - 6), 0.4, GREY, 1)


# ─── progress bar ────────────────────────────────────────────────────────────
def draw_progress_bar(img, x, y, w, h, pct, label="", color=ACCENT, bg=(40,45,55)):
    """Horizontal progress bar with rounded look."""
    pct = max(0.0, min(1.0, pct))
    # bg
    cv2.rectangle(img, (x, y), (x+w, y+h), bg, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x, y), (x+w, y+h), (60,65,75), 1, cv2.LINE_AA)
    # fill
    fw = int(w * pct)
    if fw > 0:
        cv2.rectangle(img, (x, y), (x+fw, y+h), color, -1, cv2.LINE_AA)
    if label:
        _put_text(img, label, (x + w + 8, y + h - 2), 0.4, GREY, 1)


# ─── vertical depth indicator ────────────────────────────────────────────────
def draw_depth_meter(img, x, y, w, h, depth_pct, score):
    """Vertical bar on the right showing squat depth."""
    color = score_color(score)
    cv2.rectangle(img, (x, y), (x+w, y+h), (30,35,45), -1)
    cv2.rectangle(img, (x, y), (x+w, y+h), (60,65,75), 1, cv2.LINE_AA)
    # fill from bottom
    fill_h = int(h * depth_pct)
    if fill_h > 0:
        cv2.rectangle(img, (x, y+h-fill_h), (x+w, y+h), color, -1)
    # target depth notch at 85 deg (~90% depth)
    notch_y = y + int(h * 0.15)
    cv2.line(img, (x - 4, notch_y), (x + w + 4, notch_y), SECONDARY, 2, cv2.LINE_AA)
    _put_text(img, "DEPTH", (x - 4, y - 8), 0.38, GREY, 1)
    _put_text(img, f"{int(depth_pct*100)}%", (x - 4, y + h + 16), 0.38, color, 1)


# ─── stats panel ──────────────────────────────────────────────────────────────
def draw_stats_panel(img, x, y, state, fps):
    """Draw the translucent left-side stats panel."""
    pw, ph = PANEL_W, 360
    h_img, w_img = img.shape[:2]
    if x + pw > w_img:
        pw = w_img - x
    if y + ph > h_img:
        ph = h_img - y
    if pw <= 0 or ph <= 0:
        return

    # translucent background – ROI-only blend to avoid expensive full-frame copy
    roi = img[y:y+ph, x:x+pw].copy()
    cv2.rectangle(roi, (0, 0), (pw, ph), BG_PANEL, -1)
    cv2.addWeighted(roi, PANEL_ALPHA, img[y:y+ph, x:x+pw], 1 - PANEL_ALPHA, 0, img[y:y+ph, x:x+pw])
    cv2.rectangle(img, (x, y), (x+pw, y+ph), (50,55,68), 1, cv2.LINE_AA)

    # neon accent stripe at top
    cv2.rectangle(img, (x, y), (x+pw, y+3), ACCENT, -1)

    cx = x + 12
    cy = y + 26

    # ── Title & Badge
    _put_text(img, "PYPOSE SQUAT AI", (cx, cy), 0.62, ACCENT, 2)
    _put_text(img, f"{fps:.0f} FPS", (x + pw - 68, cy - 2), 0.4, SECONDARY, 1)
    cy += 30

    # ── Set & Reps
    set_num = getattr(state, "current_set", 1)
    _put_text(img, f"SET {set_num:02d}", (cx, cy), 0.42, SECONDARY, 1)
    _put_text(img, f"REPS {state.reps:02d}", (cx + 65, cy), 0.60, WHITE, 2)
    good_txt = f"({state.good_reps} valid)"
    _put_text(img, good_txt, (cx + 155, cy), 0.4, ACCENT if state.good_reps else GREY, 1)
    cy += 28

    # ── Stage indicator / Rest Timer
    _put_text(img, "PHASE", (cx, cy), 0.42, GREY, 1)
    if getattr(state, "is_resting", False):
        rem = max(0, int(getattr(state, "rest_time_remaining", 0)))
        rest_str = f"REST {rem//60:02d}:{rem%60:02d}"
        _put_text(img, rest_str, (cx + 65, cy), 0.55, YELLOW, 2)
    else:
        stage_col = ACCENT if state.stage == "down" else SECONDARY
        _put_text(img, state.stage.upper(), (cx + 65, cy), 0.55, stage_col, 2)
    cy += 28

    # ── CHEST UPRIGHT STATUS (New)
    chest_col = ACCENT if state.chest_upright else RED
    _put_text(img, f"CHEST LEAN: {state.torso_lean:.1f}\u00b0", (cx, cy), 0.42, GREY, 1)
    _put_text(img, state.chest_status, (cx + 140, cy), 0.38, chest_col, 1)
    cy += 6
    draw_progress_bar(img, cx, cy, pw - 40, 11, state.live_chest_score / 100,
                      color=score_color(state.live_chest_score))
    cy += 24

    # ── Knee angle bar
    _put_text(img, f"KNEE FLEX: {state.knee_angle:.1f}\u00b0", (cx, cy), 0.42, GREY, 1)
    cy += 6
    draw_progress_bar(img, cx, cy, pw - 40, 11, state.live_depth_score / 100,
                      color=score_color(state.live_depth_score))
    cy += 24

    # ── Back angle bar
    _put_text(img, f"HIP HINGE: {state.back_angle:.1f}\u00b0", (cx, cy), 0.42, GREY, 1)
    cy += 6
    draw_progress_bar(img, cx, cy, pw - 40, 11, state.live_back_score / 100,
                      color=score_color(state.live_back_score))
    cy += 28

    # ── Feedback
    fb_col = ACCENT if "Good" in state.feedback or "Ready" in state.feedback else RED
    _put_text(img, state.feedback, (cx, cy), 0.55, fb_col, 2)
    cy += 30

    # ── Rep history (last 5)
    if state.rep_scores:
        _put_text(img, "RECENT REPS ACCURACY:", (cx, cy), 0.38, GREY, 1)
        cy += 16
        last5 = state.rep_scores[-5:]
        bar_w_max = pw - 90
        for i, sc in enumerate(last5):
            lbl = f"R{len(state.rep_scores) - len(last5) + i + 1}"
            _put_text(img, lbl, (cx, cy + 10), 0.35, GREY, 1)
            bw = int(bar_w_max * sc / 100)
            cv2.rectangle(img, (cx + 30, cy + 2), (cx + 30 + bw, cy + 12),
                          score_color(sc), -1)
            _put_text(img, f"{sc:.0f}%", (cx + 34 + bw, cy + 11), 0.33, score_color(sc), 1)
            cy += 16


# ─── gradient skeleton ───────────────────────────────────────────────────────
def draw_skeleton(img, lm, connections, score, glow=False):
    """Draw skeleton with colour based on the live score. Glow disabled by default for perf."""
    if not lm:
        return
    color = score_color(score)

    for a, b in connections:
        if a not in lm or b not in lm:
            continue
        if lm[a][2] > 0.5 and lm[b][2] > 0.5:
            p1 = (int(lm[a][0]), int(lm[a][1]))
            p2 = (int(lm[b][0]), int(lm[b][1]))
            if glow:
                glow_color = _lerp_color(color, (255, 255, 255), 0.4)
                cv2.line(img, p1, p2, glow_color, 6, cv2.LINE_AA)
            cv2.line(img, p1, p2, color, 3, cv2.LINE_AA)

    # joints
    for idx in lm:
        x, y, v = lm[idx]
        if v > 0.5:
            pt = (int(x), int(y))
            cv2.circle(img, pt, 5, color, -1, cv2.LINE_AA)
            cv2.circle(img, pt, 5, WHITE, 1, cv2.LINE_AA)


# ─── angle annotation arcs ───────────────────────────────────────────────────
def draw_angle_arc(img, vertex, p1, p2, angle_val, score_val, radius=35, label=""):
    """Draw a small arc + angle text at a joint."""
    color = score_color(score_val)
    vx, vy = int(vertex[0]), int(vertex[1])

    a1 = math.degrees(math.atan2(p1[1]-vertex[1], p1[0]-vertex[0]))
    a2 = math.degrees(math.atan2(p2[1]-vertex[1], p2[0]-vertex[0]))

    cv2.ellipse(img, (vx, vy), (radius, radius), 0,
                int(min(a1,a2)), int(max(a1,a2)), color, 2, cv2.LINE_AA)
    mid_a = math.radians((a1+a2)/2)
    tx = int(vx + (radius + 14) * math.cos(mid_a))
    ty = int(vy + (radius + 14) * math.sin(mid_a))
    txt = f"{angle_val:.0f}\u00b0" if not label else f"{label} {angle_val:.0f}\u00b0"
    _put_text(img, txt, (tx-14, ty+4), 0.38, color, 1)


# ─── full HUD composite ──────────────────────────────────────────────────────
def render_hud(frame, state, lm, connections, fps):
    """Main entry point: draws the full premium overlay on the frame."""
    h, w = frame.shape[:2]

    # 1. Gradient skeleton
    draw_skeleton(frame, lm, connections, state.live_score)

    # 2. Angle arcs at knee, hip, and chest
    if lm:
        import landmarks as D
        left_vis = sum(lm.get(i, (0,0,0))[2] for i in (D.L_SHOULDER, D.L_HIP, D.L_KNEE, D.L_ANKLE))
        right_vis = sum(lm.get(i, (0,0,0))[2] for i in (D.R_SHOULDER, D.R_HIP, D.R_KNEE, D.R_ANKLE))
        if left_vis >= right_vis:
            sh_i, hip_i, knee_i, ank_i = D.L_SHOULDER, D.L_HIP, D.L_KNEE, D.L_ANKLE
        else:
            sh_i, hip_i, knee_i, ank_i = D.R_SHOULDER, D.R_HIP, D.R_KNEE, D.R_ANKLE

        sh, hip, knee, ank = lm[sh_i], lm[hip_i], lm[knee_i], lm[ank_i]
        if min(sh[2], hip[2], knee[2], ank[2]) > 0.5:
            draw_angle_arc(frame, knee[:2], hip[:2], ank[:2],
                           state.knee_angle, state.live_depth_score, radius=38)
            draw_angle_arc(frame, hip[:2], sh[:2], knee[:2],
                           state.back_angle, state.live_back_score, radius=38)
            # Torso lean line
            vertical_pt = (hip[0], hip[1] - 80)
            cv2.line(frame, (int(hip[0]), int(hip[1])), (int(vertical_pt[0]), int(vertical_pt[1])),
                     (90, 95, 110), 1, cv2.LINE_AA)

    # 3. Score gauge (top-right)
    gauge_r = 55
    gauge_cx = w - gauge_r - 24
    gauge_cy = gauge_r + 24
    draw_arc_gauge(frame, gauge_cx, gauge_cy, gauge_r, state.live_score, "ACCURACY", 8)

    # 4. Depth meter (right edge)
    meter_x = w - 30
    meter_y = gauge_cy + gauge_r + 30
    meter_h = min(200, h - meter_y - 40)
    if meter_h > 40:
        draw_depth_meter(frame, meter_x, meter_y, 16, meter_h,
                         state.depth_pct, state.live_score)

    # 5. Stats panel (left side)
    draw_stats_panel(frame, 10, 10, state, fps)
