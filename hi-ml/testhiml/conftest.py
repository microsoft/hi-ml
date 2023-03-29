from pathlib import Path
import pytest

from health_ml.runner import Runner
from health_ml.utils import health_ml_package_setup

# Reduce logging noise in DEBUG mode
health_ml_package_setup()


@pytest.fixture
def mock_runner(tmp_path: Path) -> Runner:
    """A test fixture that creates a Runner object in a temporary folder."""

    return Runner(project_root=tmp_path)
