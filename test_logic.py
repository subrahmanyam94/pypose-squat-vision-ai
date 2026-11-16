"""PyPose Squat Vision AI - Test Suite Runner.
Delegates to server/test_logic.py.
"""
import os
import sys
import subprocess

if __name__ == "__main__":
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server", "test_logic.py")
    sys.exit(subprocess.call([sys.executable, script]))
