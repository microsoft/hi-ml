import sys
from typing import Generator

from pathlib import Path
import pytest


root = Path(__file__).parent.parent.parent
paths_to_add = [
    Path("hi-ml-azure") / "src",
    Path("hi-ml-azure") / "testazure",
    Path("hi-ml") / "src",
]
for folder in paths_to_add:
    full_folder = str(root / folder)
    if full_folder not in sys.path:
        print(f"Adding to sys.path for running hi-ml: {full_folder}")
        sys.path.insert(0, full_folder)

from health_ml.utils import package_setup  # noqa: E402

# Reduce logging noise in DEBUG mode
package_setup()
