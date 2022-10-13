import pytest
from pathlib import Path
from unittest.mock import patch

from health_azure.paths import ENVIRONMENT_YAML_FILE_NAME, git_repo_root_folder, shared_himl_conda_env_file


@pytest.mark.fast
def test_git_repo_root_folder() -> None:
    with patch("health_azure.paths.is_himl_used_from_git_repo", return_value=True):
        assert git_repo_root_folder() is not None
    with patch("health_azure.paths.is_himl_used_from_git_repo", return_value=False):
        with pytest.raises(ValueError) as e:
            git_repo_root_folder()
            assert "This function should not be used if hi-ml is used as an installed package." in str(e)


@pytest.mark.fast
def test_shared_himl_conda_env_file(tmp_path: Path) -> None:
    with patch("health_azure.paths.git_repo_root_folder", return_value=tmp_path):
        assert shared_himl_conda_env_file() == tmp_path / "hi-ml" / ENVIRONMENT_YAML_FILE_NAME
    with patch("health_azure.paths.is_himl_used_from_git_repo", return_value=False):
        with pytest.raises(ValueError) as e:
            shared_himl_conda_env_file()
            assert "This function should not be used if hi-ml is used as an installed package" in str(e)
