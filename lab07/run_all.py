from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).parent / "src" / "run_all.py"), run_name="__main__")
