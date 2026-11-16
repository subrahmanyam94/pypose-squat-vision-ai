"""PyPose Squat Vision AI - Biomechanics & Live Telemetry HUD.

Launches the full-screen web application dashboard or native OpenCV window.

Usage:
    python main.py                 # Launches the Full-Screen Web Application (default)
    python main.py --mode cv       # Native OpenCV Desktop Window mode
    python main.py --video x.mp4   # Analyze recorded video file
    python main.py --port 5000     # Custom port
"""
import argparse
import sys
import time
import webbrowser
import cv2

from detector import PoseDetector
from exercises import SquatAnalyzer
from landmarks import POSE_CONNECTIONS
from ui_overlay import render_hud
import web_app


def run_cv_mode(args):
    """Native OpenCV desktop window execution."""
    cap = cv2.VideoCapture(args.video if args.video else 0)
    if not cap.isOpened():
        raise SystemExit("Could not open camera/video")

    det = PoseDetector(complexity=args.complexity)
    analyzer = SquatAnalyzer(side=args.side)
    t0 = time.time()
    prev = time.time()
    fps = 30.0

    print("\nStarting Native OpenCV Window. Press 'q' to quit, 'r' to reset...")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if not args.video:
            frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        det.process(frame)
        lm = det.landmarks(w, h)
        st, form_ok = analyzer.update(lm)

        now = time.time()
        dt = max(now - prev, 1e-6)
        fps = 0.9 * fps + 0.1 * (1.0 / dt)
        prev = now

        render_hud(frame, st, lm, POSE_CONNECTIONS, fps)

        cv2.imshow("PyPose Squat Vision AI", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            analyzer = SquatAnalyzer(side=args.side)

    cap.release()
    cv2.destroyAllWindows()
    det.close()

    summ = analyzer.summary()
    elapsed = time.time() - t0
    print(f"\n{'='*55}")
    print(f"  SESSION SUMMARY")
    print(f"{'='*55}")
    print(f"  Reps:               {summ['reps']}")
    print(f"  Good reps:          {summ['good_reps']}")
    print(f"  Avg form score:     {summ['avg_form_score']}/100")
    print(f"  Duration:           {elapsed:.0f}s")
    print(f"{'='*55}")
    if summ['reps'] > 0:
        print(f"  Per-rep scores:     {', '.join(f'{s:.0f}%' for s in analyzer.state.rep_scores)}")


def run_web_mode(args):
    """Full-screen web application mode matching Stitch/PyPose UI."""
    server = web_app.start_server(
        host=args.host,
        port=args.port,
        video=args.video,
        side=args.side,
        complexity=args.complexity
    )

    url = f"http://localhost:{args.port}"
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    print(f"Web Dashboard ready at: {url}")
    print("Press Ctrl+C to stop server.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down PyPose Web Server...")
        if web_app.GLOBAL_PIPELINE:
            web_app.GLOBAL_PIPELINE.stop()
        server.server_close()


def main():
    ap = argparse.ArgumentParser(description="PyPose Squat Vision AI")
    ap.add_argument("--mode", default="web", choices=["web", "cv"],
                    help="UI mode: 'web' for full-screen web dashboard (default), 'cv' for OpenCV window")
    ap.add_argument("--video", help="Path to video file instead of live camera")
    ap.add_argument("--side", default="auto", choices=["auto", "left", "right"])
    ap.add_argument("--complexity", type=int, default=0, choices=[0, 1, 2],
                    help="MediaPipe model: 0=lite (fastest), 1=full, 2=heavy")
    ap.add_argument("--port", type=int, default=5000, help="Web server port (default 5000)")
    ap.add_argument("--host", default="0.0.0.0", help="Web server host (default 0.0.0.0)")
    ap.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = ap.parse_args()

    if args.mode == "web":
        run_web_mode(args)
    else:
        run_cv_mode(args)


if __name__ == "__main__":
    main()
