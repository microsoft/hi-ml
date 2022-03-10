import logging
import sys
from pathlib import Path


root = Path(__file__).parent.parent.parent
for folder in [Path("hi-ml-azure") / "src", Path("hi-ml") / "src", Path("hi-ml-azure") / "testazure"]:
    full_folder = str(root / folder)
    if full_folder not in sys.path:
        print(f"Adding to sys.path for running hi-ml: {full_folder}")
        sys.path.insert(0, full_folder)

# Matplotlib is very talkative in DEBUG mode
logging.getLogger('matplotlib').setLevel(logging.INFO)
