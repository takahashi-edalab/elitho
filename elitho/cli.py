"""Command-line interface for elitho."""

import os
import sys
from pathlib import Path


def run_gui():
    """Launch the Streamlit GUI for elitho."""
    # Get the path to gui.py
    gui_path = Path(__file__).parent / "gui.py"

    if not gui_path.exists():
        print(f"Error: GUI file not found at {gui_path}", file=sys.stderr)
        sys.exit(1)

    # Run streamlit with the gui.py file
    os.execvp("streamlit", ["streamlit", "run", str(gui_path)])


if __name__ == "__main__":
    run_gui()
