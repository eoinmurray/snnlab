"""Execute the standalone documentation examples in a disposable directory."""

import os
import runpy
import tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]


def main():
    with tempfile.TemporaryDirectory(prefix="snnlab-examples-") as directory:
        for path in sorted((ROOT / "examples").glob("*.py")):
            print(f"Running {path.name}", flush=True)
            namespace = runpy.run_path(str(path))
            namespace["main"](Path(directory) / path.stem)
    print("All documentation examples passed")


if __name__ == "__main__":
    main()
