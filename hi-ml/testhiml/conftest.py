from pathlib import Path
import uuid
import pytest

from health_ml.runner import Runner
from health_ml.utils import health_ml_package_setup
from health_ml.utils.fixed_paths import repository_root_directory

# Reduce logging noise in DEBUG mode
health_ml_package_setup()

root = repository_root_directory()

print("Creating test outputs folder.")
new_dir = root / ("test_outputs_" + uuid.uuid4().hex)
new_dir.mkdir()
assert new_dir.is_dir(), "New directory should have been created."
new_file = new_dir / "test_file.txt"
new_file.write_text("Hello world!")
assert new_file.is_file(), "New file should have been created."


@pytest.fixture
def mock_runner(tmp_path: Path) -> Runner:
    """A test fixture that creates a Runner object in a temporary folder."""

    return Runner(project_root=tmp_path)
