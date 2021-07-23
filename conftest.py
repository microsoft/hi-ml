import sys
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture(autouse=True, scope='session')
def test_suite_setup() -> Generator:
    try:
        # Try a first import. If hi-ml was installed as a package, this should work.
        from health import azure
    except ImportError:
        src_folder = Path(__file__) / "src"
        assert src_folder.is_dir()
        sys.path.insert(0, str(src_folder))
        try:
            # Try importing again. If that fails, give up.
            from health import azure
        except ImportError:
            raise ValueError("Unable to access the hi-ml package, even after messing with sys.path")
    yield
