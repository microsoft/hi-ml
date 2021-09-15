#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
Tests for the functions in health.azure.azure_util
"""
from argparse import ArgumentParser
import os
import logging
import time
from pathlib import Path
from typing import Optional, List
from unittest import mock
from unittest.mock import MagicMock, patch
from uuid import uuid4

import conda_merge
import health.azure.azure_util as util
import pytest
from _pytest.capture import CaptureFixture
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.conda_dependencies import CondaDependencies
from health.azure.himl import AML_IGNORE_FILE, append_to_amlignore
from testhiml.health.azure.util import repository_root

RUN_ID = uuid4().hex
RUN_NUMBER = 42
EXPERIMENT_NAME = "fancy-experiment"


def oh_no() -> None:
    """
    Raise a simple exception. To be used as a side_effect for mocks.
    """
    raise ValueError("Throwing an exception")


@patch("health.azure.azure_util.Run")
def test_create_run_recovery_id(mock_run: MagicMock) -> None:
    """
    The recovery id created for a run
    """
    mock_run.id = RUN_ID
    mock_run.experiment.name = EXPERIMENT_NAME
    recovery_id = util.create_run_recovery_id(mock_run)
    assert recovery_id == EXPERIMENT_NAME + util.EXPERIMENT_RUN_SEPARATOR + RUN_ID


@pytest.mark.parametrize("on_azure", [True, False])
def test_is_running_on_azure_agent(on_azure: bool) -> None:
    with mock.patch.dict(os.environ, {"AGENT_OS": "LINUX" if on_azure else ""}):
        assert on_azure == util.is_running_on_azure_agent()


@patch("health.azure.azure_util.Workspace")
@patch("health.azure.azure_util.Experiment")
@patch("health.azure.azure_util.Run")
def test_fetch_run(mock_run: MagicMock, mock_experiment: MagicMock, mock_workspace: MagicMock) -> None:
    mock_run.id = RUN_ID
    mock_run.experiment = mock_experiment
    mock_experiment.name = EXPERIMENT_NAME
    recovery_id = EXPERIMENT_NAME + util.EXPERIMENT_RUN_SEPARATOR + RUN_ID
    mock_run.number = RUN_NUMBER
    with mock.patch("health.azure.azure_util.get_run", return_value=mock_run):
        run_to_recover = util.fetch_run(mock_workspace, recovery_id)
        # TODO: should we assert that the correct string is logged?
        assert run_to_recover.number == RUN_NUMBER
    mock_experiment.side_effect = oh_no
    with pytest.raises(Exception) as e:
        util.fetch_run(mock_workspace, recovery_id)
    assert str(e.value).startswith(f"Unable to retrieve run {RUN_ID}")


@patch("health.azure.azure_util.Run")
@patch("health.azure.azure_util.Experiment")
@patch("health.azure.azure_util.get_run")
def test_fetch_run_for_experiment(get_run: MagicMock, mock_experiment: MagicMock, mock_run: MagicMock) -> None:
    get_run.side_effect = oh_no
    mock_run.id = RUN_ID
    mock_experiment.get_runs = lambda: [mock_run, mock_run, mock_run]
    mock_experiment.name = EXPERIMENT_NAME
    with pytest.raises(Exception) as e:
        util.fetch_run_for_experiment(mock_experiment, RUN_ID)
    exp = f"Run {RUN_ID} not found for experiment: {EXPERIMENT_NAME}. Available runs are: {RUN_ID}, {RUN_ID}, {RUN_ID}"
    assert str(e.value) == exp


@patch("health.azure.azure_util.InteractiveLoginAuthentication")
def test_get_authentication(mock_interactive_authentication: MagicMock) -> None:
    with mock.patch.dict(os.environ, {}, clear=True):
        util.get_authentication()
        assert mock_interactive_authentication.called
    service_principal_id = "1"
    tenant_id = "2"
    service_principal_password = "3"
    with mock.patch.dict(
            os.environ,
            {
                util.ENV_SERVICE_PRINCIPAL_ID: service_principal_id,
                util.ENV_TENANT_ID: tenant_id,
                util.ENV_SERVICE_PRINCIPAL_PASSWORD: service_principal_password
            },
            clear=True):
        spa = util.get_authentication()
        assert isinstance(spa, ServicePrincipalAuthentication)
        assert spa._service_principal_id == service_principal_id
        assert spa._tenant_id == tenant_id
        assert spa._service_principal_password == service_principal_password


def test_get_secret_from_environment() -> None:
    env_variable_name = uuid4().hex.upper()
    env_variable_value = "42"
    with pytest.raises(ValueError) as e:
        util.get_secret_from_environment(env_variable_name)
    assert str(e.value) == f"There is no value stored for the secret named '{env_variable_name}'"
    assert util.get_secret_from_environment(env_variable_name, allow_missing=True) is None
    with mock.patch.dict(os.environ, {env_variable_name: env_variable_value}):
        assert util.get_secret_from_environment(env_variable_name) == env_variable_value


def test_to_azure_friendly_string() -> None:
    """
    Tests the to_azure_friendly_string function which should replace everything apart from a-zA-Z0-9_ with _, and
    replace multiple _ with a single _
    """
    bad_string = "full__0f_r*bb%sh"
    good_version = util.to_azure_friendly_string(bad_string)
    assert good_version == "full_0f_r_bb_sh"
    good_string = "Not_Full_0f_Rubbish"
    good_version = util.to_azure_friendly_string(good_string)
    assert good_version == good_string
    optional_string = None
    assert optional_string == util.to_azure_friendly_string(optional_string)


def test_split_recovery_id_fails() -> None:
    """
    Other tests test the main branch of split_recovery_id, but they do not test the exceptions
    """
    with pytest.raises(ValueError) as e:
        id = util.EXPERIMENT_RUN_SEPARATOR.join([str(i) for i in range(3)])
        util.split_recovery_id(id)
    assert str(e.value) == f"recovery_id must be in the format: 'experiment_name:run_id', but got: {id}"
    with pytest.raises(ValueError) as e:
        id = "foo_bar"
        util.split_recovery_id(id)
    assert str(e.value) == f"The recovery ID was not in the expected format: {id}"


@pytest.mark.parametrize(["id", "expected1", "expected2"],
                         [("foo:bar", "foo", "bar"),
                          ("foo:bar_ab_cd", "foo", "bar_ab_cd"),
                          ("a_b_c_00_123", "a_b_c", "a_b_c_00_123"),
                          ("baz_00_123", "baz", "baz_00_123"),
                          ("foo_bar_abc_123_456", "foo_bar_abc", "foo_bar_abc_123_456"),
                          # This is the run ID of a hyperdrive parent run. It only has one numeric part at the end
                          ("foo_bar_123", "foo_bar", "foo_bar_123"),
                          # This is a hyperdrive child run
                          ("foo_bar_123_3", "foo_bar", "foo_bar_123_3"),
                          ])
def test_split_recovery_id(id: str, expected1: str, expected2: str) -> None:
    """
    Check that run recovery ids are correctly parsed into experiment and run id.
    """
    assert util.split_recovery_id(id) == (expected1, expected2)


def test_merge_conda(
        random_folder: Path,
        caplog: CaptureFixture,
        ) -> None:
    """
    Tests the logic for merging Conda environment files.
    """
    env1 = """
channels:
  - defaults
  - pytorch
dependencies:
  - conda1=1.0
  - conda2=2.0
  - conda_both=3.0
  - pip:
      - azureml-sdk==1.7.0
      - foo==1.0
"""
    env2 = """
channels:
  - defaults
dependencies:
  - conda1=1.1
  - conda_both=3.0
  - pip:
      - azureml-sdk==1.6.0
      - bar==2.0
"""
    # Spurious test failures on Linux build agents, saying that they can't write the file. Wait a bit.
    time.sleep(0.5)
    file1 = random_folder / "env1.yml"
    file1.write_text(env1)
    file2 = random_folder / "env2.yml"
    file2.write_text(env2)
    # Spurious test failures on Linux build agents, saying that they can't read the file. Wait a bit.
    time.sleep(0.5)
    files = [file1, file2]
    merged_file = random_folder / "merged.yml"
    util.merge_conda_files(files, merged_file)
    merged_file_text = merged_file.read_text()
    assert merged_file_text.splitlines() == """channels:
- defaults
- pytorch
dependencies:
- conda1=1.0
- conda1=1.1
- conda2=2.0
- conda_both=3.0
- pip:
  - azureml-sdk==1.6.0
  - azureml-sdk==1.7.0
  - bar==2.0
  - foo==1.0
""".splitlines()
    conda_dep = CondaDependencies(merged_file)

    # We expect to see the union of channels.
    assert list(conda_dep.conda_channels) == ["defaults", "pytorch"]

    # Package version conflicts are not resolved, both versions are retained.
    assert list(conda_dep.conda_packages) == ["conda1=1.0", "conda1=1.1", "conda2=2.0", "conda_both=3.0"]
    assert list(conda_dep.pip_packages) == ["azureml-sdk==1.6.0", "azureml-sdk==1.7.0", "bar==2.0", "foo==1.0"]

    # Are names merged correctly?
    assert "name:" not in merged_file_text
    env1 = "name: env1\n" + env1
    file1.write_text(env1)
    env2 = "name: env2\n" + env2
    file2.write_text(env2)
    util.merge_conda_files(files, merged_file)
    assert "name: env2" in merged_file.read_text()

    def raise_a_merge_error() -> None:
        raise conda_merge.MergeError("raising an exception")

    with mock.patch("health.azure.azure_util.conda_merge.merge_channels") as mock_merge_channels:
        mock_merge_channels.side_effect = lambda _: raise_a_merge_error()
        with pytest.raises(conda_merge.MergeError):
            util.merge_conda_files(files, merged_file)
    assert "Failed to merge channel priorities" in caplog.text  # type: ignore

    # If there are no channels do not produce any merge of them
    with mock.patch("health.azure.azure_util.conda_merge.merge_channels") as mock_merge_channels:
        mock_merge_channels.return_value = []
        util.merge_conda_files(files, merged_file)
        assert "channels:" not in merged_file.read_text()

    with mock.patch("health.azure.azure_util.conda_merge.merge_dependencies") as mock_merge_dependencies:
        mock_merge_dependencies.side_effect = lambda _: raise_a_merge_error()
        with pytest.raises(conda_merge.MergeError):
            util.merge_conda_files(files, merged_file)
    assert "Failed to merge dependencies" in caplog.text  # type: ignore

    # If there are no dependencies then something is wrong with the conda files or our parsing of them
    with mock.patch("health.azure.azure_util.conda_merge.merge_dependencies") as mock_merge_dependencies:
        mock_merge_dependencies.return_value = []
        with pytest.raises(ValueError) as e:
            util.merge_conda_files(files, merged_file)
        assert "No dependencies found in any of the conda files" in str(e.value)


@pytest.mark.parametrize(["s", "expected"],
                         [
                             ("1s", 1),
                             ("0.5m", 30),
                             ("1.5h", 90 * 60),
                             ("1.0d", 24 * 3600),
                             ("", None),
                         ])
def test_run_duration(s: str, expected: Optional[float]) -> None:
    actual = util.run_duration_string_to_seconds(s)
    assert actual == expected
    if expected:
        assert isinstance(actual, int)


def test_run_duration_fails() -> None:
    with pytest.raises(Exception):
        util.run_duration_string_to_seconds("17b")


def test_repository_root() -> None:
    root = repository_root()
    assert (root / "SECURITY.md").is_file()


def test_nonexisting_amlignore(random_folder: Path) -> None:
    """
    Test that we can create an .AMLignore file, and it gets deleted after use.
    """
    folder1 = "Added1"
    added_folders = [folder1]
    cwd = Path.cwd()
    amlignore = random_folder / AML_IGNORE_FILE
    assert not amlignore.is_file()
    os.chdir(random_folder)
    with append_to_amlignore(added_folders):
        new_contents = amlignore.read_text()
        for f in added_folders:
            assert f in new_contents
    assert not amlignore.is_file()
    os.chdir(cwd)


@patch("health.azure.azure_util.Workspace")
def test_create_python_environment(
        mock_workspace: mock.MagicMock,
        random_folder: Path,
        ) -> None:
    just_conda_str_env_name = "HealthML-a26b35a434dd27a44a8224c709fb4760"
    conda_str = """name: simple-env
dependencies:
  - pip=20.1.1
  - python=3.7.3
  - pip:
    - azureml-sdk==1.23.0
    - conda-merge==0.1.5
  - pip:
    - --index-url https://test.pypi.org/simple/
    - --extra-index-url https://pypi.org/simple
    - hi-ml-azure
"""
    conda_environment_file = random_folder / "environment.yml"
    conda_environment_file.write_text(conda_str)
    conda_dependencies = CondaDependencies(conda_dependencies_file_path=conda_environment_file)
    env = util.create_python_environment(conda_environment_file=conda_environment_file)
    assert list(env.python.conda_dependencies.conda_channels) == list(conda_dependencies.conda_channels)
    assert list(env.python.conda_dependencies.conda_packages) == list(conda_dependencies.conda_packages)
    assert list(env.python.conda_dependencies.pip_options) == list(conda_dependencies.pip_options)
    assert list(env.python.conda_dependencies.pip_packages) == list(conda_dependencies.pip_packages)
    assert "AZUREML_OUTPUT_UPLOAD_TIMEOUT_SEC" in env.environment_variables
    assert "AZUREML_RUN_KILL_SIGNAL_TIMEOUT_SEC" in env.environment_variables
    assert "RSLEX_DIRECT_VOLUME_MOUNT" in env.environment_variables
    assert "RSLEX_DIRECT_VOLUME_MOUNT_MAX_CACHE_SIZE" in env.environment_variables
    assert env.name == just_conda_str_env_name

    pip_extra_index_url = "https://where.great.packages.live/"
    docker_base_image = "viennaglobal.azurecr.io/azureml/azureml_a187a87cc7c31ac4d9f67496bc9c8239"
    env = util.create_python_environment(
        conda_environment_file=conda_environment_file,
        pip_extra_index_url=pip_extra_index_url,
        docker_base_image=docker_base_image,
        environment_variables={"HELLO": "world"})
    assert "HELLO" in env.environment_variables
    assert env.name != just_conda_str_env_name
    assert env.docker.base_image == docker_base_image

    private_pip_wheel_url = "https://some.blob/private/wheel"
    with mock.patch("health.azure.azure_util.Environment") as mock_environment:
        mock_environment.add_private_pip_wheel.return_value = private_pip_wheel_url
        env = util.create_python_environment(
            conda_environment_file=conda_environment_file,
            workspace=mock_workspace,
            private_pip_wheel_path=Path(__file__))
    envs_pip_packages = list(env.python.conda_dependencies.pip_packages)
    assert "hi-ml-azure" in envs_pip_packages
    assert private_pip_wheel_url in envs_pip_packages

    private_pip_wheel_path = Path("a_file_that_does_not.exist")
    with pytest.raises(FileNotFoundError) as e:
        _ = util.create_python_environment(
            conda_environment_file=conda_environment_file,
            workspace=mock_workspace,
            private_pip_wheel_path=private_pip_wheel_path)
    assert f"Cannot add add_private_pip_wheel: {private_pip_wheel_path}" in str(e.value)


@patch("health.azure.azure_util.Environment")
@patch("health.azure.azure_util.Workspace")
def test_register_environment(
        mock_workspace: mock.MagicMock,
        mock_environment: mock.MagicMock,
        caplog: CaptureFixture,
        ) -> None:
    env_name = "an environment"
    env_version = "an environment"
    mock_environment.get.return_value = mock_environment
    mock_environment.name = env_name
    mock_environment.version = env_version
    with caplog.at_level(logging.INFO):  # type: ignore
        _ = util.register_environment(mock_workspace, mock_environment)
        assert f"Using existing Python environment '{env_name}'" in caplog.text  # type: ignore
        mock_environment.get.side_effect = oh_no
        _ = util.register_environment(mock_workspace, mock_environment)
        assert f"environment '{env_name}' does not yet exist, creating and registering" in caplog.text  # type: ignore


def test_set_environment_variables_for_multi_node(
        caplog: CaptureFixture,
        capsys: CaptureFixture,
        ) -> None:
    with caplog.at_level(logging.INFO):  # type: ignore
        util.set_environment_variables_for_multi_node()
        assert "No settings for the MPI central node found" in caplog.text  # type: ignore
        assert "Assuming that this is a single node training job" in caplog.text  # type: ignore

    with mock.patch.dict(
            os.environ,
            {
                util.ENV_AZ_BATCHAI_MPI_MASTER_NODE: "here",
                util.ENV_MASTER_PORT: "there",
                util.ENV_OMPI_COMM_WORLD_RANK: "everywhere",
                util.ENV_MASTER_ADDR: "else",
            },
            clear=True):
        util.set_environment_variables_for_multi_node()
    out, _ = capsys.readouterr()
    assert "Distributed training: MASTER_ADDR = here, MASTER_PORT = there, NODE_RANK = everywhere" in out

    with mock.patch.dict(
            os.environ,
            {
                util.ENV_MASTER_IP: "here",
                util.ENV_NODE_RANK: "everywhere",
                util.ENV_MASTER_ADDR: "else",
            },
            clear=True):
        util.set_environment_variables_for_multi_node()
    out, _ = capsys.readouterr()
    assert "Distributed training: MASTER_ADDR = here, MASTER_PORT = 6105, NODE_RANK = everywhere" in out


class MockRun:
    def __init__(self, run_id: str = 'run1234') -> None:
        self.id = run_id


def test_determine_run_id_source(tmp_path: Path) -> None:
    parser = ArgumentParser()
    parser.add_argument("--latest_run_file", type=str)
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--run_recovery_ids", type=str)
    parser.add_argument("--run_ids", type=str)

    # If latest run path provided, expect source to be latest run file
    mock_latest_run_path = tmp_path / "most_recent_run.txt"
    mock_args = parser.parse_args(["--latest_run_file", str(mock_latest_run_path)])
    assert util.determine_run_id_source(mock_args) == util.AzureRunIdSource.LATEST_RUN_FILE

    # If experiment name is provided, expect source to be experiment
    mock_args = parser.parse_args(["--experiment_name", "fake_experiment"])
    assert util.determine_run_id_source(mock_args) == util.AzureRunIdSource.EXPERIMENT_LATEST

    # If run recovery id is provided, expect source to be that
    mock_args = parser.parse_args(["--run_recovery_ids", "experiment:run1234"])
    assert util.determine_run_id_source(mock_args) == util.AzureRunIdSource.RUN_RECOVERY_ID

    # If run ids provided, expect source to be that
    mock_args = parser.parse_args(["--run_ids", "run1234"])
    assert util.determine_run_id_source(mock_args) == util.AzureRunIdSource.RUN_ID

    # if none are provided, raise ValueError
    mock_args = parser.parse_args([])
    with pytest.raises(Exception):
        util.determine_run_id_source(mock_args)


def test_get_aml_runs_from_latest_run_file(tmp_path: Path) -> None:
    mock_run_id = 'mockrunid123'
    mock_latest_run_path = tmp_path / "most_recent_run.txt"
    with open(mock_latest_run_path, 'w+') as f_path:
        f_path.write(mock_run_id)
    parser = ArgumentParser()
    parser.add_argument("--latest_run_file", type=str)
    mock_args = parser.parse_args(["--latest_run_file", str(mock_latest_run_path)])
    with mock.patch("health.azure.azure_util.Workspace") as mock_workspace:
        with mock.patch("health.azure.azure_util.fetch_run") as mock_fetch_run:
            mock_fetch_run.return_value = MockRun(mock_run_id)
            aml_runs = util.get_aml_runs_from_latest_run_file(mock_args, mock_workspace)
            mock_fetch_run.assert_called_once_with(workspace=mock_workspace, run_recovery_id=mock_run_id)
            assert len(aml_runs) == 1
            assert aml_runs[0].id == mock_run_id

    # if path doesn't exist, expect error
    with pytest.raises(Exception):
        mock_args = parser.parse_args(["--latest_run_file", "idontexist"])
        with mock.patch("health.azure.azure_util.Workspace") as mock_workspace:
            util.get_aml_runs_from_latest_run_file(mock_args, mock_workspace)

    # if arg not provided, expect error
    with pytest.raises(Exception):
        mock_args = parser.parse_args(["--latest_run_file", None])  # type: ignore
        with mock.patch("health.azure.azure_util.Workspace") as mock_workspace:
            util.get_aml_runs_from_latest_run_file(mock_args, mock_workspace)


def test_get_latest_aml_runs_from_experiment() -> None:

    def _get_experiment_runs() -> List[MockRun]:

        return [MockRun(), MockRun(), MockRun(), MockRun()]

    mock_experiment_name = "MockExperiment"
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--tags", action="append", default=[])
    parser.add_argument("--num_runs", type=int, default=1)
    mock_args = parser.parse_args(["--experiment_name", mock_experiment_name])
    with mock.patch("health.azure.azure_util.Experiment") as mock_experiment:
        with mock.patch("health.azure.azure_util.Workspace",
                        experiments={mock_experiment_name: mock_experiment}
                        ) as mock_workspace:
            mock_experiment.get_runs.return_value = _get_experiment_runs()
            aml_runs = util.get_latest_aml_runs_from_experiment(mock_args, mock_workspace)
            assert len(aml_runs) == 1  # if num_runs not provided, returns 1 by default
            assert aml_runs[0].id == "run1234"

    # Test that correct number of runs are returned if both experiment_name and num_runs are provided
    mock_args = parser.parse_args(["--experiment_name", mock_experiment_name, "--num_runs", "3"])
    with mock.patch("health.azure.azure_util.Experiment") as mock_experiment:
        mock_experiment.get_runs.return_value = _get_experiment_runs()
        with mock.patch("health.azure.azure_util.Workspace",
                        experiments={mock_experiment_name: mock_experiment}
                        ) as mock_workspace:
            runs = util.get_latest_aml_runs_from_experiment(mock_args, mock_workspace)  # type: ignore
    assert len(runs) == 3
    assert runs[0].id == "run1234"

    # Test that correct number of returns if both experiment_name and tags are provided
    mock_args = parser.parse_args(["--experiment_name", mock_experiment_name, "--tags", "3"])
    with mock.patch("health.azure.azure_util.Experiment") as mock_experiment:
        mock_experiment.get_runs.return_value = _get_experiment_runs()
        with mock.patch("health.azure.azure_util.Workspace",
                        experiments={mock_experiment_name: mock_experiment}
                        ) as mock_workspace:
            runs = util.get_latest_aml_runs_from_experiment(mock_args, mock_workspace)  # type: ignore
    assert len(runs) == 1
    assert runs[0].id == "run1234"

    # Test that value error is raised if experiment name is not in workspace
    mock_args = parser.parse_args(["--experiment_name", "idontexist"])
    with pytest.raises(Exception):
        with mock.patch("health.azure.azure_util.Workspace",
                        experiments={mock_experiment_name: mock_experiment}
                        ) as mock_workspace:
            util.get_latest_aml_runs_from_experiment(mock_args, mock_workspace)  # type: ignore


def test_get_aml_runs_from_recovery_ids() -> None:
    def _mock_get_most_recent_run(path: Path, workspace: Workspace) -> MockRun:
        return MockRun()

    parser = ArgumentParser()
    parser.add_argument("--run_recovery_ids", type=str, action="append", default=None)

    # Test that the correct number of runs are returned if run_recovery_id(s) is(are) provided
    mock_args = parser.parse_args(["--run_recovery_id", "expt:run123"])
    with mock.patch("health.azure.azure_util.Workspace") as mock_workspace:
        with mock.patch("health.azure.azure_util.fetch_run", _mock_get_most_recent_run):
            runs = util.get_aml_runs_from_recovery_ids(mock_args, mock_workspace)  # type: ignore
    assert len(runs) == 1
    assert runs[0].id == "run1234"


def test_get_aml_runs_from_runids() -> None:
    parser = ArgumentParser()
    parser.add_argument("--run_ids", action="append", default=[])

    # assert single run returned if single run id provided
    mock_run_id = "run123"
    mock_args = parser.parse_args(["--run_ids", mock_run_id])
    with mock.patch("health.azure.azure_util.Workspace") as mock_workspace:
        mock_workspace.get_run.return_value = MockRun(mock_run_id)
        aml_runs = util.get_aml_runs_from_runids(mock_args, mock_workspace)
        mock_workspace.get_run.assert_called_with(mock_run_id)
        assert len(aml_runs) == 1
        assert aml_runs[0].id == mock_run_id

    # assert multiple runs returned if multiple run ids provided
    mock_run_id_2 = "run456"
    mock_args = parser.parse_args(["--run_ids", mock_run_id, "--run_ids", mock_run_id_2])
    with mock.patch("health.azure.azure_util.Workspace") as mock_workspace:
        mock_workspace.get_run.return_value = MockRun(mock_run_id_2)
        aml_runs = util.get_aml_runs_from_runids(mock_args, mock_workspace)

        assert len(aml_runs) == 2
        assert aml_runs[1].id == mock_run_id_2


def test_get_aml_runs(tmp_path: Path) -> None:
    parser = ArgumentParser()
    mock_latest_run_path = tmp_path / "most_recent_run.txt"
    parser.add_argument("--latest_run_file", type=str)

    # if latest run path has been provided:
    mock_args = parser.parse_args(["--latest_run_file", str(mock_latest_run_path)])
    run_id_source = util.AzureRunIdSource.LATEST_RUN_FILE
    with mock.patch("health.azure.azure_util.get_aml_runs_from_latest_run_file") as mock_get_from_run_path:
        with mock.patch("health.azure.azure_util.Workspace") as mock_workspace:
            aml_runs = util.get_aml_runs(mock_args, mock_workspace, run_id_source)
            mock_get_from_run_path.assert_called_once()

    # if experiment name has been provided:
    parser.add_argument("--experiment_name", type=str)
    mock_args = parser.parse_args(["--experiment_name", "mockExperiment"])
    run_id_source = util.AzureRunIdSource.EXPERIMENT_LATEST
    with mock.patch("health.azure.azure_util.get_latest_aml_runs_from_experiment") as mock_get_from_experiment:
        with mock.patch("health.azure.azure_util.Workspace") as mock_workspace:
            aml_runs = util.get_aml_runs(mock_args, mock_workspace, run_id_source)
            mock_get_from_experiment.assert_called_once()

    # if run_recovery_id has been provided:
    parser.add_argument("--run_recovery_ids", action="append")
    mock_args = parser.parse_args(["--run_recovery_ids", "experiment:run1234"])
    run_id_source = util.AzureRunIdSource.RUN_RECOVERY_ID
    with mock.patch("health.azure.azure_util.get_aml_runs_from_recovery_ids") as mock_get_from_recovery_ids:
        with mock.patch("health.azure.azure_util.Workspace") as mock_workspace:
            aml_runs = util.get_aml_runs(mock_args, mock_workspace, run_id_source)
            mock_get_from_recovery_ids.assert_called_once()

    # if run_ids has been provided:
    parser.add_argument("--run_ids", action="append")
    mock_args = parser.parse_args(["--run_ids", "run1234"])
    run_id_source = util.AzureRunIdSource.RUN_ID
    with mock.patch("health.azure.azure_util.get_aml_runs_from_runids",
                    return_value=[MockRun()]) as mock_get_from_run_id:
        with mock.patch("health.azure.azure_util.Workspace") as mock_workspace:
            aml_runs = util.get_aml_runs(mock_args, mock_workspace, run_id_source)
            assert len(aml_runs) == 1
            mock_get_from_run_id.assert_called_once()

    # otherwise assert Exception raised
    mock_args = parser.parse_args([])
    run_id_source = None
    with pytest.raises(Exception):
        with mock.patch("health.azure.azure_util.Workspace") as mock_workspace:
            util.get_aml_runs(mock_args, mock_workspace, run_id_source)  # type: ignore
