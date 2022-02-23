import sys
from pathlib import Path


root = Path(__file__).parent.parent
for folder in ["hi-ml-azure", "hi-ml"]:
    full_folder = str(root / folder / "src")
    if full_folder not in sys.path:
        print(f"Adding to sys.path for running hi-ml: {full_folder}")
        sys.path.insert(0, full_folder)
