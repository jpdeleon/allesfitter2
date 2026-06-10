import runpy
import sys
from pathlib import Path

_scripts_dir = Path(__file__).parent.parent.parent / "scripts"
script_path = _scripts_dir / "prepare_allesfit.py"

def prepare_allesfit():
    sys.path.insert(0, str(script_path.parent))
    runpy.run_path(str(script_path), run_name="__main__")


def run_allesfitter_grid():
    grid_path = _scripts_dir / "run_allesfitter_grid.py"
    sys.path.insert(0, str(grid_path.parent))
    runpy.run_path(str(grid_path), run_name="__main__")
