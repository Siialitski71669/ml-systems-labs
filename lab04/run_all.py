import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

scripts = [
    "src/pipeline.py",
    "src/modeling.py",
    "src/label_audit.py"
]

for script in scripts:
    print(f"\n{'=' * 50}")
    print(f"Running {script}")
    print("=" * 50)

    subprocess.run(
        [sys.executable, str(ROOT / script)],
        cwd=ROOT,
        check=True
    )

print("\nAll Lab 4 steps completed successfully.")
