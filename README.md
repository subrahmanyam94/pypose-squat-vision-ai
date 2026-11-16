# 🏋️ PyPose Squat Vision AI - Biomechanics & Telemetry HUD

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-PoseLandmarker-brightgreen.svg)](https://developers.google.com/mediapipe)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red.svg)](https://opencv.org/)
[![Test Suite](https://img.shields.io/badge/Tests-15%20Passing-success.svg)](test_logic.py)

A computer vision web application and desktop AI squat analysis tool. Uses Google MediaPipe BlazePose and OpenCV to track human kinematics in real-time, compute continuous 0–100 form accuracy, verify chest uprightness, track sets with 2-minute rest intervals, and analyze barbell velocity and sagittal drift.

---

## 🌟 Core Capabilities

### 1. Continuous 0–100 Scoring Formula (At Any Angle)
Instead of binary pass/fail counting, every degree of motion contributes to an aggregate accuracy score:
$$\text{Score} = 0.50 \times \text{Depth} + 0.30 \times \text{Chest Upright} + 0.20 \times \text{Hip Hinge}$$
- **Depth Score**: Sub-parallel ($\le 90^\circ \rightarrow 100$), Parallel ($90^\circ$–$95^\circ \rightarrow 90$–$100$), Acceptable ($95^\circ$–$110^\circ \rightarrow 50$–$90$).
- **Chest Score**: Torso lean from vertical ($\le 35^\circ \rightarrow 100$, $35^\circ$–$50^\circ \rightarrow 75$–$100$, $>75^\circ \rightarrow 0$).
- **Hip Hinge Score**: Shoulder-hip-knee spine angle ($140^\circ$–$165^\circ \rightarrow 100$).

### 2. Chest Uprightness & Anti-Collapse Validation
- Accurately computes absolute torso lean angle from vertical.
- Live alerts: **"CHEST UPRIGHT • OPTIMAL"** vs. **"CHEST UP! KEEP TORSO UPRIGHT"**.
- Prevents lumbar strain and forward torso collapse during descent and ascent.

### 3. Set Tracking & 2-Minute Rest Gap Window
- **Continue Current Set Within 2 Minutes**: Resting or pausing after reps triggers a 2-minute gap countdown (`02:00` down to `00:00`). If you start squatting again within 2 minutes, your **previous set continues seamlessly** (reps are added to the active set, not counted as a new set).
- **Automatic Set Advance After 2 Minutes**: If the full 2 minutes elapse with no squats, the set automatically finalizes and archives into completed sets, and the next set begins automatically.
- **Manual Next Set**: Finish a set immediately at any point using the **"NEXT SET (N)"** button or hotkey <kbd>N</kbd>.
- **Select & Continue Any Set**: Each completed set card in the Reps Per Set section has a **"Continue Set X"** button (or `/api/select_set`), allowing you to reactivate any previous set and append more reps to it at any time.

### 4. Interactive Reps Per Set Bar Graph
- Renders individual rep accuracy bars ($R_1, R_2, R_3\dots$) grouped by set at the bottom of the HUD.
- Color-coded:
  - 🟢 **Neon Green (#39ff14)**: Optimal form ($\ge 85\%$)
  - 🔵 **Cyan (#4cd7f6)**: Acceptable form ($\ge 70\%$)
  - 🔴 **Red (#ffb4ab)**: Sub-optimal / flagged form ($< 70\%$)
- Preserves past sets with average score, total reps, valid reps, and accuracy percentage.

### 5. Strict Perfect Squat Validation & Anti-Knee-Bend Protection
Requires all 5 biomechanical criteria of a genuine squat:
- **Strict Depth**: Knee angle must reach parallel depth ($\le 98^\circ$, target $\le 90^\circ$).
- **Hip Hinge (Sitting Back)**: Hip joint (`shoulder-hip-knee`) must close to $\le 102^\circ$ (simple knee bends stay $> 125^\circ$).
- **Thigh Parallel to Floor**: Relative vertical knee-hip distance $(\text{knee}_y - \text{hip}_y) / \text{femur\_len} \le 0.30$.
- **Physical Hip Descent**: Hips must vertically drop downwards by $\ge 38\%$ of femur length.
- **Athletic Torso Incline**: Torso inclination must be between $12^\circ$ and $50^\circ$ (rejects standing upright knee bobs $< 12^\circ$).
- Completely rejects simple knee bends, standing knee dips, shallow bounces, and leg curls.

### 6. Velocity-Based Training (VBT) & Barbell Path
- Approximates barbell/shoulder vertical velocity ($m/s$) and peak concentric velocity.
- Tracks horizontal sagittal drift ($\pm \text{cm}$) relative to midfoot.

---

## 📂 Project Architecture

```
files/
├── client/                      # 🎨 Local Frontend Dashboard Assets
│   ├── index.html               # Stitch / PyPose Dark Mode Dashboard
│   ├── app.js                   # Client JS (Telemetry polling, SVG waveforms, bar charts)
│   └── styles.css               # Dark theme, neon accents, sleek scrollbars
│
├── server/                      # ⚙️ Python Backend Engine
│   ├── web_app.py               # Built-in threaded HTTP server & REST endpoints
│   ├── exercises.py             # Squat state machine, 2m rest, sets, depth & chest
│   ├── detector.py              # MediaPipe PoseLandmarker wrapper
│   ├── geometry.py              # Joint angle & torso lean trigonometry
│   ├── landmarks.py             # 33 pose landmark indices & wireframe connections
│   ├── ui_overlay.py            # OpenCV HUD overlay renderer
│   ├── test_logic.py            # Comprehensive 15-test unit suite
│   └── requirements.txt         # Backend Python dependencies
│
├── main.py                      # 💻 Unified Local App Launcher (Web + Native OpenCV)
├── test_logic.py                # Root test suite runner
├── requirements.txt             # Project root dependencies
└── README.md                    # Project documentation
```

---

## ⚡ Quick Start (Local Machine)

### Requirements
- Python 3.10+
- Webcam or recorded video file

```bash
# 1. Install dependencies
pip install -r requirements.txt
```

### 1. Launch Full-Screen Web Dashboard (Default)
```bash
python main.py
```
Opens `http://localhost:5000` automatically in your browser with the full PyPose dark dashboard, live video feed, telemetry charts, and controls.

### 2. Launch Native OpenCV Desktop Mode
```bash
python main.py --mode cv
```
Runs directly in an OpenCV window with high-performance overlay graphics. Press <kbd>Q</kbd> to quit, <kbd>R</kbd> to reset.

### 3. Analyze a Recorded Video File
```bash
python main.py --video my_squats.mp4
```

---

## 🔌 REST & Streaming API Reference

The backend provides a lightweight, zero-dependency HTTP server:

| Endpoint | Method | Description |
| --- | --- | --- |
| `/` | `GET` | Serves the full PyPose web dashboard UI |
| `/video_feed` | `GET` | Multipart MJPEG camera stream with live HUD overlay |
| `/api/telemetry` | `GET` | Real-time JSON: angles, scores, stage, sets, 2m rest timer, waveforms |
| `/api/next_set` | `POST` | Manually finishes current set and advances to the next set |
| `/api/resume_set` | `POST` | Closes rest gap timer and immediately continues the active set |
| `/api/select_set` | `POST` | Selects / revives a specific set to continue (`{"set": 1}`) |
| `/api/reset` | `POST` | Resets reps, sets, and session statistics |
| `/api/toggle_pause` | `POST` | Pauses / resumes vision pipeline |
| `/api/snapshot` | `POST` | Captures high-res JPEG snapshot to `/snapshots` |
| `/api/export_csv` | `GET` | Downloads session angle curves as `rep_telemetry.csv` |
| `/api/dump_json` | `GET` | Downloads complete landmark and set breakdown as `landmarks_dump.json` |

---

## 🧪 Automated Testing

Run the test suite to verify all trigonometry math, chest scoring, anti-cheat validation, and 2-minute rest intervals:

```bash
python test_logic.py
```

```
PASS test_angle
PASS test_torso_lean_and_chest_score
PASS test_continuous_scores
PASS test_pose_builder
PASS test_counts_five_reps
PASS test_noise_does_not_double_count
PASS test_shallow_reps_not_counted_or_low_score
PASS test_forward_lean_and_chest_upright_flagged
PASS test_perfect_form_high_score
PASS test_live_score_and_chest_updates
PASS test_low_visibility_pauses
PASS test_set_counting
PASS test_standing_knee_bend_not_counted
PASS test_two_minute_rest_between_sets
PASS test_simple_knee_bend_rejected
```
**15 / 15 Tests Passing (100%)**.

---

## ⌨️ Dashboard Hotkeys

| Key | Action |
| --- | --- |
| <kbd>SPACE</kbd> | Pause / Resume Camera Processing |
| <kbd>R</kbd> | Reset Session Counters & Timer |
| <kbd>N</kbd> | Next Set / Stop Set & Begin 2m Rest / Skip Rest |
| <kbd>S</kbd> | Save Camera Snapshot to disk |

---

## 📄 License
MIT License. Open-source for athletes, coaches, and developers.
