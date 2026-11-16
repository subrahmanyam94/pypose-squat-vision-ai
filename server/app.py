"""Production Entry Point for Render / Cloud Deployments.
Reads $PORT environment variable assigned by Render.
"""
import os
import signal
import sys
import web_app

def main():
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5000"))
    video = os.environ.get("VIDEO_SOURCE", None)
    complexity = int(os.environ.get("COMPLEXITY", "0"))

    print(f"Starting PyPose Vision AI Server on {host}:{port} ...")
    server = web_app.start_server(
        host=host,
        port=port,
        video=video,
        complexity=complexity
    )

    def shutdown(signum, frame):
        print("\nShutting down server...")
        if web_app.GLOBAL_PIPELINE:
            web_app.GLOBAL_PIPELINE.stop()
        server.server_close()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        shutdown(None, None)

if __name__ == "__main__":
    main()

