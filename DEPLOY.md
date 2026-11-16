# 🚀 Deployment Guide: PyPose Squat Vision AI

This guide provides step-by-step instructions for deploying the **Frontend to Vercel** and the **Backend to Render**.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        DEPLOYMENT TOPOLOGY                             │
├──────────────────────────────────┬─────────────────────────────────────┤
│         CLIENT (Vercel)          │          SERVER (Render)            │
│       https://your-app.vercel.app│      https://your-api.onrender.com  │
├──────────────────────────────────┼─────────────────────────────────────┤
│ • Static CDN Hosting             │ • Python 3 Runtime / Container      │
│ • Dark Mode UI Dashboard         │ • MediaPipe BlazePose AI Engine     │
│ • SVG Waveforms & Gauges         │ • Real-time Telemetry REST API      │
│ • Reps Per Set Bar Graphs        │ • MJPEG Video Stream (/video_feed)  │
│ • 2-Min Rest Countdown Modal     │ • CORS Enabled (* All Origins)      │
└──────────────────────────────────┴─────────────────────────────────────┘
```

---

## Part 1: Deploying the Backend on Render

Render hosts Python web services with automated SSL and background processes.

### Method A: Blueprint (Recommended - 1 Click)
1. Push this repository to **GitHub**.
2. Go to your [Render Dashboard](https://dashboard.render.com/) and click **New +** $\rightarrow$ **Blueprint**.
3. Select your GitHub repository.
4. Render automatically reads `server/render.yaml` and provisions:
   - **Root Directory**: `server`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
   - **Environment Variables**: `COMPLEXITY=0`, `PORT=10000`
5. Click **Apply**. When deployment finishes, copy your Render service URL (e.g., `https://pypose-vision-ai.onrender.com`).

### Method B: Manual Web Service Setup
1. Go to [Render Dashboard](https://dashboard.render.com/) $\rightarrow$ **New +** $\rightarrow$ **Web Service**.
2. Connect your GitHub repository.
3. Configure the following settings:
   - **Name**: `pypose-squat-vision-api`
   - **Root Directory**: `server`
   - **Environment**: `Python 3`
   - **Region**: `Oregon (US West)` or closest to your users
   - **Branch**: `main`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
   - **Plan**: `Free`
4. Under **Advanced** $\rightarrow$ **Environment Variables**, add:
   | Key | Value | Description |
   | --- | --- | --- |
   | `COMPLEXITY` | `0` | MediaPipe BlazePose Lite (high FPS) |
   | `HOST` | `0.0.0.0` | Bind to all interfaces |
5. Click **Create Web Service**.
6. Note your Render URL once active (e.g. `https://pypose-squat-vision-api.onrender.com`).

---

## Part 2: Deploying the Frontend on Vercel

Vercel provides edge delivery for HTML/CSS/JavaScript applications.

### Step-by-Step Vercel Setup:
1. Go to your [Vercel Dashboard](https://vercel.com/dashboard) and click **Add New...** $\rightarrow$ **Project**.
2. Import your GitHub repository.
3. In the project configuration screen:
   - **Framework Preset**: `Other`
   - **Root Directory**: Click **Edit** and select **`client`** (Important!).
   - **Build and Output Settings**: Leave as default (`vercel.json` handles routing).
4. Click **Deploy**.
5. Your frontend is live at `https://<your-project>.vercel.app`!

---

## Part 3: Connecting Client & Server

The client includes an interactive **Server URL Switcher** in the top navigation bar:

1. Open your deployed Vercel URL (e.g., `https://your-app.vercel.app`).
2. In the top header bar, locate the **SERVER** input field:
   ```
   [●] SERVER: [ https://your-api.onrender.com ]
   ```
3. Enter your Render backend URL (e.g., `https://your-api.onrender.com`) and hit Enter or click outside.
4. The client saves this URL to `localStorage`, so you only need to enter it once.
5. The connection dot turns **Neon Green [●]**, and the video stream and live joint telemetry begin streaming immediately!

> **Direct URL Parameter**: You can also bookmark or share direct links with the server pre-configured:
> `https://your-app.vercel.app/?server=https://your-api.onrender.com`

---

## Part 4: Local Development & Testing

You can run both client and server locally on your development machine:

### 1. Start the Backend Server:
```bash
cd server
pip install -r requirements.txt
python app.py
# Server starts at http://localhost:5000
```

### 2. Start the Frontend Client:
In a separate terminal:
```bash
cd client
# Run using any static server:
npx serve .
# Or Python stdlib:
python -m http.server 3000
```
Visit `http://localhost:3000` — it connects to `http://localhost:5000` automatically!

### 3. All-in-One Local Mode:
If you prefer running everything from a single command without separate frontend servers:
```bash
python main.py
# Serves both dashboard and vision pipeline at http://localhost:5000
```

---

## Troubleshooting & FAQs

### Q: Why does Render take 30–50 seconds on the first request?
**A**: Render's Free Tier spins down idle containers after 15 minutes of inactivity. When a new visitor connects, Render spins the container back up. Once awake, performance remains fast (25–30 FPS). Paid plans ($7/mo) keep containers permanently warm.

### Q: Does CORS block requests between Vercel and Render?
**A**: No. The server code in `server/web_app.py` has full pre-flight CORS headers (`OPTIONS`, `GET`, `POST`) and sends `Access-Control-Allow-Origin: *` across all REST and streaming endpoints.

### Q: How does camera access work when deployed on the cloud?
**A**: Browsers require **HTTPS** to allow webcam access. Because Vercel automatically provisions SSL (`https://`), webcam permissions are granted. For cloud servers without hardware cameras, you can supply video files via `--video path/to/video.mp4` or the client stream bridge.

