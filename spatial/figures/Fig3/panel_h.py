"""Fig 3 h: drawn by panel_f.py, which holds the shared layout of f, g and h.

Inputs and outputs: see panel_f.py. Further arguments are passed on to panel_f.py.

Usage: python figures/Fig3/panel_h.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Run panel_f.py for letter h and exit with its status."""
    panel_f = str(Path(__file__).with_name("panel_f.py"))
    sys.exit(subprocess.call([sys.executable, panel_f, "--letters", "h", *sys.argv[1:]]))


if __name__ == "__main__":
    main()
