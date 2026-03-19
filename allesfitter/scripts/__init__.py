import runpy
import sys
from pathlib import Path

script_path = Path(__file__).parent.parent.parent / "scripts" / "prepare_allesfit.py"

def prepare_allesfit():
    sys.path.insert(0, str(script_path.parent))
    runpy.run_path(str(script_path), run_name="__main__")
