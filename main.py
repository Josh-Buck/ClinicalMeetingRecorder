"""
Entry point for the Clinical Meeting Recorder Streamlit app.

Run with:
    streamlit run main.py

This is just a wrapper that defers to app.py.
"""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    app_path = Path(__file__).parent / "app.py"
    if not app_path.exists():
        print("Could not find app.py")
        sys.exit(1)
    # Re-launch via streamlit if run directly
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])
