#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Generator
import pytest
from pathlib import Path
from torch.nn import Module
from unittest.mock import patch

from health_azure.paths import ENVIRONMENT_YAML_FILE_NAME, git_repo_root_folder, is_himl_used_from_git_repo
from health_ml.utils import set_model_to_eval_mode
from health_ml.utils.common_utils import check_conda_environments, get_all_environment_files


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
def test_get_all_environment_files(temp_project_root: Path, temp_env_path: Path, empty_temp_dir: Path) -> None:
    # Firstly check the cases when is_himl_used_from_git_repo returns True
    with patch("health_azure.paths.is_himl_used_from_git_repo", return_value=True):
        assert is_himl_used_from_git_repo()
        # If we don't patch git_repo_root_folder it will return top-level hi-ml folder
        himl_root = git_repo_root_folder()
        # Separate workspaces means the environment.yml file is located in 'hi-ml/hi-ml'
        # As an initial sanity check, ensure this file exists
        env_file_root = himl_root / "hi-ml"
        expected_env_path = env_file_root / ENVIRONMENT_YAML_FILE_NAME
        assert expected_env_path.is_file()
        # Now check that get_all_environment_files returns this path
        assert get_all_environment_files(project_root=himl_root) == [expected_env_path]

        # If we patch the git_repo_root_folder to return a different folder, we should be able
        # to pick up an env file there
        with patch("health_azure.paths.git_repo_root_folder", return_value=temp_project_root):
            assert get_all_environment_files(project_root=temp_project_root) == [temp_env_path]

        # If we patch git_repo_root_folder to return a different folder that does not contain an env
        # file, an error should be raised
        with patch("health_azure.paths.git_repo_root_folder", return_value=empty_temp_dir):
            with pytest.raises(AssertionError) as e1:
                get_all_environment_files(project_root=empty_temp_dir)
            assert "Didn't find an environment file at" in str(e1)

    # Now check the case where is_himl_used_from_git_repo returns False. In this case, the project root repo
    # won't be hi-ml, so we replace it here with temp_path (which has no concept of the hi-ml directory)
    with patch("health_azure.paths.is_himl_used_from_git_repo", return_value=False):
        with patch("health_azure.paths.git_repo_root_folder", return_value=temp_project_root):
            assert get_all_environment_files(project_root=temp_project_root) == [temp_env_path]

        # Now pass a directory that doesn't have an env file and check that an error is raised
        with patch("health_azure.paths.git_repo_root_folder", return_value=empty_temp_dir):
            with pytest.raises(ValueError) as e2:
                get_all_environment_files(project_root=empty_temp_dir)
            assert "No Conda environment files were found in the repository" in str(e2)


@pytest.mark.fast
def test_check_conda_environments(temp_env_path: Path) -> None:
    with patch("health_azure.paths.is_himl_used_from_git_repo", return_value=True):
        with patch("health_azure.paths.shared_himl_conda_env_file", return_value=temp_env_path):
            # Calling check_conda_environments on an empty list should do nothing
            check_conda_environments([])

            # Pass a non-empty list and mock the rturn value of is_conda_file_with_pip_include
            with patch("health_ml.utils.common_utils.is_conda_file_with_pip_include", return_value=(True, None)):
                with pytest.raises(ValueError) as e:
                    check_conda_environments([Path('some_path')])
                assert "uses '-r' to reference pip requirements" in str(e)

        # If the file that we pass is the same as the return value of shared_himl_conda_env_file
        # an error will not be raised
        with patch("health_azure.paths.shared_himl_conda_env_file", return_value=Path('some_path')):
            with patch("health_ml.utils.common_utils.is_conda_file_with_pip_include", return_value=(True, None)):
                check_conda_environments([Path('some_path')])

        # If not is_conda_is_conda_file_with_pip_include, excpect nothing to happen
        with patch("health_azure.paths.shared_himl_conda_env_file", return_value=temp_env_path):
            with patch("health_ml.utils.common_utils.is_conda_file_with_pip_include", return_value=(False, None)):
                check_conda_environments([Path('some_path')])

    # Now check cases where is_himl_used_from_git_repo is False:
    with patch("health_azure.paths.is_himl_used_from_git_repo", return_value=False):
        # If not is_conda_is_conda_file_with_pip_include, excpect nothing to happen
        with patch("health_ml.utils.common_utils.is_conda_file_with_pip_include", return_value=(False, None)):
            check_conda_environments([])
            check_conda_environments([Path('some_path')])

        # If is_conda_is_conda_file_with_pip_include=True, expect a ValueError to be raised
        with patch("health_ml.utils.common_utils.is_conda_file_with_pip_include", return_value=(True, None)):
            with pytest.raises(ValueError) as e:
                check_conda_environments([Path('some_path')])
            assert "uses '-r' to reference pip requirements" in str(e)
