import uuid

import pytest

from health_ml.utils.fixed_paths import repository_root_directory


@pytest.mark.fast
def test_folders() -> None:
    """
    Test that the test outputs folder is created and that it contains a file.
    """
    root = repository_root_directory()

    print("Creating test outputs folder.")
    new_dir = root / ("test_outputs_" + uuid.uuid4().hex)
    new_dir.mkdir()
    assert new_dir.is_dir(), "New directory should have been created."
    new_file = new_dir / "test_file.txt"
    new_file.write_text("Hello world!")
    assert new_file.is_file(), "New file should have been created."
