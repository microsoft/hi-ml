#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Generator
import pytest
from pathlib import Path
from torch.nn import Module
from unittest.mock import patch

from health_azure.paths import ENVIRONMENT_YAML_FILE_NAME
from health_ml.utils import set_model_to_eval_mode
from health_ml.utils.common_utils import change_working_directory, check_conda_environment, choose_conda_env_file


@pytest.mark.fast
def test_set_to_eval_mode() -> None:
    model = Module()
    model.train(True)
    assert model.training
    with set_model_to_eval_mode(model):
        assert not model.training
    assert model.training

    model.eval()
    assert not model.training
    with set_model_to_eval_mode(model):
        assert not model.training
    assert not model.training


@pytest.fixture(scope="session")
def temp_project_root(tmp_path_factory: pytest.TempPathFactory,) -> Generator:
    temp_project_root = tmp_path_factory.mktemp("test_folder")
    yield temp_project_root


@pytest.fixture(scope="session")
def temp_env_path(temp_project_root: Path) -> Generator:
    tmp_env_path = temp_project_root / ENVIRONMENT_YAML_FILE_NAME
    print(f"TEMP ENV PATH: {tmp_env_path}")
    tmp_env_path.write_text("name: DummyEnv")
    yield tmp_env_path
    tmp_env_path.unlink()


@pytest.fixture(scope="session")
def empty_temp_dir(temp_project_root: Path) -> Generator:
    # Create a directory that will not contain an environment definition file
    empty_temp_dir = temp_project_root / "empty"
    empty_temp_dir.mkdir()
    yield empty_temp_dir


@pytest.mark.fast
def test_choose_conda_env_file1(tmp_path: Path) -> None:
    """Test if a fixed conda file is correctly handled in choose_conda_env_file"""
    env_file = tmp_path / "some_file.txt"
    assert not env_file.is_file()
    with pytest.raises(FileNotFoundError) as ex:
        choose_conda_env_file(env_file)
    assert "The Conda file specified on the commandline could not be found" in str(ex)
    env_file.touch()
    assert choose_conda_env_file(env_file) == env_file


@pytest.mark.fast
def test_choose_conda_env_file2(tmp_path: Path) -> None:
    folder1 = tmp_path / "folder1"
    folder2 = folder1 / "folder2"
    folder3 = folder2 / "folder3"
    folder3.mkdir(parents=True)
    # A mock environment file that lives in folder 1. When later setting the current working directory to a folder
    # below this file, this file should be picked up
    env_file = folder1 / ENVIRONMENT_YAML_FILE_NAME
    env_file.touch()
    # Firstly check the cases when is_himl_used_from_git_repo returns True
    with patch("health_azure.paths.is_himl_used_from_git_repo", return_value=True):
        # Work directory is folder3, but the repo root is set to folder2 above it: The environment file that lives
        # in folder1 should not be found.
        with patch("health_azure.paths.git_repo_root_folder", return_value=folder2):
            with change_working_directory(folder3):
                with pytest.raises(FileNotFoundError) as ex:
                    choose_conda_env_file(env_file=None)
                assert f"No Conda environment file '{ENVIRONMENT_YAML_FILE_NAME}' was found" in str(ex)
        # Work directory is folder3, the repo root is set to folder1 where environment file lives
        with patch("health_azure.paths.git_repo_root_folder", return_value=folder1):
            with change_working_directory(folder3):
                assert choose_conda_env_file(env_file=None) == env_file
    # Check use when hi-ml is not used from git repo
    with patch("health_azure.paths.is_himl_used_from_git_repo", return_value=False):
        # Work directory is folder3, environment file 2 levels up should be found
        with change_working_directory(folder3):
            assert choose_conda_env_file(env_file=None) == env_file


@pytest.mark.fast
def test_check_conda_environments(tmp_path: Path) -> None:
    """Test if the check_conda_environments function correctly identifies environment files that contain
    a `-r` part in the pip section"""
    invalid_env = tmp_path / "invalid_env.yaml"
    invalid_env.write_text("""name: DummyEnv
dependencies:
  - pip:
      - foo==1.0
      - -r requirements.txt
""")
    with pytest.raises(ValueError) as e:
        check_conda_environment(invalid_env)
    assert "uses '-r' to reference pip requirements" in str(e.value)

    valid_env = tmp_path / "valid_env.yaml"

    valid_env.write_text("""name: DummyEnv
dependencies:
  - pip:
      - foo==1.0
""")
    check_conda_environment(valid_env)
