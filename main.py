"""PyPose Squat Vision AI - Root Launcher.
Delegates to the server/ package to prevent file duplication.
"""
import os
import sys

# Add server directory to module search path and change working directory
SERVER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server")
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)
os.chdir(SERVER_DIR)

from main import main

if __name__ == "__main__":
    main()
