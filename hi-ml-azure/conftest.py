import sys
from pathlib import Path


full_folder = str(Path(__file__).parent / "src")
if full_folder not in sys.path:
    print(f"Adding to sys.path for running hi-ml-azure: {full_folder}")
    sys.path.insert(0, str(full_folder))
