#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
Tests for the functions in health_azure.azure_util
"""
import logging
import os
import sys
import time
from argparse import ArgumentParser
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from unittest import mock
from unittest.mock import MagicMock, patch
from uuid import uuid4

import conda_merge
import param
import pytest
from _pytest.capture import CaptureFixture
from _pytest.logging import LogCaptureFixture
from azureml._vendor.azure_storage.blob import Blob
from azureml.core import Experiment, ScriptRunConfig
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.environment import CondaDependencies
from azureml.data.azure_storage_datastore import AzureBlobDatastore

import health_azure.utils as util
from health_azure import himl
from health_azure.himl import AML_IGNORE_FILE, append_to_amlignore
from testazure.test_himl import RunTarget, render_and_run_test_script
from testazure.util import DEFAULT_WORKSPACE, change_working_directory, repository_root, MockRun

RUN_ID = uuid4().hex
RUN_NUMBER = 42
EXPERIMENT_NAME = "fancy-experiment"
AML_TESTS_EXPERIMENT = "test_experiment"


def oh_no() -> None:
    """
    Raise a simple exception. To be used as a side_effect for mocks.
    """
    raise ValueError("Throwing an exception")


@pytest.mark.fast
def test_find_file(tmp_path: Path) -> None:
    file_name = "some_file.json"
    file = tmp_path / file_name
    file.touch()
    python_root = tmp_path / "python_root"
    python_root.mkdir(exist_ok=False)
    start_path = python_root / "starting_directory"
    start_path.mkdir(exist_ok=False)
    where_are_we_now = Path.cwd()
    os.chdir(start_path)
    found_file = util._find_file(file_name, False)
    assert found_file
    with mock.patch.dict(os.environ, {"PYTHONPATH": str(python_root.absolute())}):
        found_file = util._find_file(file_name)
        assert not found_file
    os.chdir(where_are_we_now)


def test_is_running_in_azureml() -> None:
    """
    Test if the code correctly recognizes that it is executed in AzureML
    """
    # These tests would always run outside of AzureML, on local box or Azure build agents. Function should return
    # False in all those cases
    assert not util.is_running_in_azure_ml()
    assert not util.is_running_in_azure_ml(util.RUN_CONTEXT)
    # When in AzureML, the Run has a field "experiment"
    mock_workspace = "foo"
    with patch("health_azure.utils.RUN_CONTEXT") as mock_run_context:
        mock_run_context.experiment = MagicMock(workspace=mock_workspace)
        # We can't try that with the default argument because of Python's handling of mutable default arguments
        # (default argument value has been assigned already before mocking)
        assert util.is_running_in_azure_ml(util.RUN_CONTEXT)


@pytest.mark.fast
@patch("health_azure.utils.Workspace.from_config")
@patch("health_azure.utils.get_authentication")
@patch("health_azure.utils.Workspace")
def test_get_workspace(
        mock_workspace: mock.MagicMock,
        mock_get_authentication: mock.MagicMock,
        mock_from_config: mock.MagicMock,
        tmp_path: Path) -> None:
    # Test the case when running on AML
    with patch("health_azure.utils.RUN_CONTEXT") as mock_run_context:
        mock_run_context.experiment = MagicMock(workspace=mock_workspace)
        workspace = util.get_workspace(None, None)
        assert workspace == mock_workspace

    # Test the case when a workspace object is provided. The test always runs outside AzureML, and should return the
    # workspace object unchanged
    mock_workspace2 = "foo"
    workspace = util.get_workspace(mock_workspace2, None)  # type: ignore
    assert workspace == mock_workspace2

    # Test the case when a workspace config path is provided
    mock_get_authentication.return_value = "auth"
    _ = util.get_workspace(None, Path(__file__))
    mock_from_config.assert_called_once_with(path=__file__, auth="auth")

    # Work off a temporary directory: No config file is present
    with change_working_directory(tmp_path):
        with pytest.raises(ValueError) as ex:
            util.get_workspace(None, None)
        assert "No workspace config file given" in str(ex)
    # Workspace config file is set to a file that does not exist
    with pytest.raises(ValueError) as ex:
        util.get_workspace(None, workspace_config_path=tmp_path / "does_not_exist")
    assert "Workspace config file does not exist" in str(ex)


@patch("health_azure.utils.Run")
def test_create_run_recovery_id(mock_run: MagicMock) -> None:
    """
    The recovery id created for a run
    """
    mock_run.id = RUN_ID
    mock_run.experiment.name = EXPERIMENT_NAME
    recovery_id = util.create_run_recovery_id(mock_run)
    assert recovery_id == EXPERIMENT_NAME + util.EXPERIMENT_RUN_SEPARATOR + RUN_ID


@patch("health_azure.utils.Workspace")
@patch("health_azure.utils.Experiment")
@patch("health_azure.utils.Run")
def test_fetch_run(mock_run: MagicMock, mock_experiment: MagicMock, mock_workspace: MagicMock) -> None:
    mock_run.id = RUN_ID
    mock_run.experiment = mock_experiment
    mock_experiment.name = EXPERIMENT_NAME
    recovery_id = EXPERIMENT_NAME + util.EXPERIMENT_RUN_SEPARATOR + RUN_ID
    mock_run.number = RUN_NUMBER
    with mock.patch("health_azure.utils.get_run", return_value=mock_run):
        run_to_recover = util.fetch_run(mock_workspace, recovery_id)
        assert run_to_recover.number == RUN_NUMBER
    mock_experiment.side_effect = oh_no
    with pytest.raises(Exception) as e:
        util.fetch_run(mock_workspace, recovery_id)
    assert str(e.value).startswith(f"Unable to retrieve run {RUN_ID}")


@patch("health_azure.utils.Run")
@patch("health_azure.utils.Experiment")
@patch("health_azure.utils.get_run")
def test_fetch_run_for_experiment(get_run: MagicMock, mock_experiment: MagicMock, mock_run: MagicMock) -> None:
    get_run.side_effect = oh_no
    mock_run.id = RUN_ID
    mock_experiment.get_runs = lambda: [mock_run, mock_run, mock_run]
    mock_experiment.name = EXPERIMENT_NAME
    with pytest.raises(Exception) as e:
        util.fetch_run_for_experiment(mock_experiment, RUN_ID)
    exp = f"Run {RUN_ID} not found for experiment: {EXPERIMENT_NAME}. Available runs are: {RUN_ID}, {RUN_ID}, {RUN_ID}"
    assert str(e.value) == exp


@patch("health_azure.utils.InteractiveLoginAuthentication")
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

    with mock.patch("health_azure.utils.conda_merge.merge_channels") as mock_merge_channels:
        mock_merge_channels.side_effect = lambda _: raise_a_merge_error()
        with pytest.raises(conda_merge.MergeError):
            util.merge_conda_files(files, merged_file)
    assert "Failed to merge channel priorities" in caplog.text  # type: ignore

    # If there are no channels do not produce any merge of them
    with mock.patch("health_azure.utils.conda_merge.merge_channels") as mock_merge_channels:
        mock_merge_channels.return_value = []
        util.merge_conda_files(files, merged_file)
        assert "channels:" not in merged_file.read_text()

    with mock.patch("health_azure.utils.conda_merge.merge_dependencies") as mock_merge_dependencies:
        mock_merge_dependencies.side_effect = lambda _: raise_a_merge_error()
        with pytest.raises(conda_merge.MergeError):
            util.merge_conda_files(files, merged_file)
    assert "Failed to merge dependencies" in caplog.text  # type: ignore

    # If there are no dependencies then something is wrong with the conda files or our parsing of them
    with mock.patch("health_azure.utils.conda_merge.merge_dependencies") as mock_merge_dependencies:
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
                         ])  # NOQA
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


@patch("health_azure.utils.Workspace")
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
    with mock.patch("health_azure.utils.Environment") as mock_environment:
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


class MockEnvironment:
    def __init__(self, name: str, version: str = "autosave") -> None:
        self.name = name
        self.version = version


@patch("health_azure.utils.Environment")
@patch("health_azure.utils.Workspace")
def test_register_environment(
        mock_workspace: mock.MagicMock,
        mock_environment: mock.MagicMock,
        caplog: LogCaptureFixture,
) -> None:
    def _mock_env_get(workspace: Workspace, name: str = "", version: Optional[str] = None) -> MockEnvironment:
        if version is None:
            raise Exception("not found")
        return MockEnvironment(name, version=version)

    env_name = "an environment"
    env_version = "environment version"
    mock_environment.get.return_value = mock_environment
    mock_environment.name = env_name
    mock_environment.version = env_version
    with caplog.at_level(logging.INFO):  # type: ignore
        _ = util.register_environment(mock_workspace, mock_environment)
        caplog_text: str = caplog.text  # for mypy
        assert f"Using existing Python environment '{env_name}' with version '{env_version}'" in caplog_text

        # test that log is correct when exception is triggered
        mock_environment.get.side_effect = oh_no
        _ = util.register_environment(mock_workspace, mock_environment)
        caplog_text = caplog.text  # for mypy
        assert f"environment '{env_name}' does not yet exist, creating and registering it with version" \
               f" '{env_version}'" in caplog_text

        # test that environment version equals ENVIRONMENT_VERSION when exception is triggered
        # rather than default value of "autosave"
        mock_environment.version = None
        with patch.object(mock_environment, "get", _mock_env_get):
            with patch.object(mock_environment, "register") as mock_register:
                mock_register.return_value = mock_environment
                env = util.register_environment(mock_workspace, mock_environment)
                assert env.version == util.ENVIRONMENT_VERSION


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


@pytest.mark.fast
@patch("health_azure.utils.fetch_run")
@patch("azureml.core.Workspace")
def test_get_most_recent_run(mock_workspace: MagicMock, mock_fetch_run: MagicMock, tmp_path: Path) -> None:
    mock_run_id = "run_abc_123"
    mock_run = MockRun(mock_run_id)
    mock_workspace.get_run.return_value = mock_run
    mock_fetch_run.return_value = mock_run

    latest_path = tmp_path / "most_recent_run.txt"
    latest_path.touch()
    latest_path.write_text(mock_run_id)

    run = util.get_most_recent_run(latest_path, mock_workspace)
    assert run.id == mock_run_id 


def test_get_aml_runs_from_latest_run_file(tmp_path: Path) -> None:
    mock_run_id = 'mockrunid123'
    mock_latest_run_path = tmp_path / "most_recent_run.txt"
    with open(mock_latest_run_path, 'w+') as f_path:
        f_path.write(mock_run_id)
    parser = ArgumentParser()
    parser.add_argument("--latest_run_file", type=str)
    mock_args = parser.parse_args(["--latest_run_file", str(mock_latest_run_path)])
    with mock.patch("health_azure.utils.Workspace") as mock_workspace:
        with mock.patch("health_azure.utils.get_aml_run_from_run_id") as mock_fetch_run:
            mock_fetch_run.return_value = MockRun(mock_run_id)
            aml_run = util.get_aml_run_from_latest_run_file(mock_args, mock_workspace)
            mock_fetch_run.assert_called_once_with(mock_run_id, aml_workspace=mock_workspace)
            assert aml_run.id == mock_run_id

    # if path doesn't exist, expect error
    with pytest.raises(AssertionError) as ex:
        mock_args = parser.parse_args(["--latest_run_file", "idontexist"])
        with mock.patch("health_azure.utils.Workspace") as mock_workspace:
            util.get_aml_run_from_latest_run_file(mock_args, mock_workspace)
    expected_str = "When running in cloud builds, this should pick up the ID of a previous training run"
    assert str(ex.value) == expected_str

    # if arg not provided, expect error
    with pytest.raises(AssertionError) as assertionException:
        mock_args = parser.parse_args(["--latest_run_file", None])  # type: ignore
        with mock.patch("health_azure.utils.Workspace") as mock_workspace:
            util.get_aml_run_from_latest_run_file(mock_args, mock_workspace)
    assert str(assertionException.value) == expected_str


def _get_experiment_runs(tags: Dict[str, str]) -> List[MockRun]:
    mock_run_no_tags = MockRun()
    mock_run_tags = MockRun(tags={"completed": "True"})
    all_runs = [mock_run_no_tags for _ in range(5)] + [mock_run_tags for _ in range(5)]
    return [r for r in all_runs if r.tags == tags] if tags else all_runs


@pytest.mark.fast
@pytest.mark.parametrize("num_runs, tags, expected_num_returned", [
    (1, {"completed": "True"}, 1),
    (3, {}, 3),
    (2, {"Completed: False"}, 0)
])
def test_get_latest_aml_run_from_experiment(num_runs: int, tags: Dict[str, str], expected_num_returned: int) -> None:
    mock_experiment_name = "MockExperiment"

    with mock.patch("health_azure.utils.Experiment") as mock_experiment:
        with mock.patch("health_azure.utils.Workspace",
                        experiments={mock_experiment_name: mock_experiment}
                        ) as mock_workspace:
            mock_experiment.get_runs.return_value = _get_experiment_runs(tags)
            aml_runs = util.get_latest_aml_runs_from_experiment(mock_experiment_name, num_runs=num_runs,
                                                                tags=tags, aml_workspace=mock_workspace)
            assert len(aml_runs) == expected_num_returned


def test_get_latest_aml_run_from_experiment_remote(tmp_path: Path) -> None:
    """
    Test that a remote run with particular tags can be correctly retrieved, ignoring any more recent
    experiments which do not have the correct tags. Note: this test will instantiate 2 new Runs in the
    workspace described in your config.json file, under an experiment defined by AML_TESTS_EXPERIMENT
    """
    ws = DEFAULT_WORKSPACE.workspace
    assert True

    experiment = Experiment(ws, AML_TESTS_EXPERIMENT)
    config = ScriptRunConfig(
        source_directory=".",
        command=["cd ."],  # command that does nothing
        compute_target="local"
    )
    # Create first run and tag
    first_run = experiment.submit(config)
    tags = {"experiment_type": "great_experiment"}
    first_run.set_tags(tags)
    first_run.wait_for_completion()

    # Create second run and ensure no tags
    second_run = experiment.submit(config)
    if any(second_run.get_tags()):
        second_run.remove_tags(tags)

    # Retrieve latest run with given tags (expect first_run to be returned)
    retrieved_runs = util.get_latest_aml_runs_from_experiment(AML_TESTS_EXPERIMENT, tags=tags, aml_workspace=ws)
    assert len(retrieved_runs) == 1
    assert retrieved_runs[0].id == first_run.id
    assert retrieved_runs[0].get_tags() == tags


@pytest.mark.fast
@patch("health_azure.utils.fetch_run")
@patch("health_azure.utils.Workspace")
@pytest.mark.parametrize("mock_run_id, run_id_type", [
    ("run_abc_123", util.RunId),
    ("experiment1:run_bcd_456", util.RunRecoveryId)
])
def test_get_aml_run_from_run_id(mock_workspace: MagicMock, mock_fetch_run: MagicMock, mock_run_id: str,
                                 run_id_type: util.RunSource) -> None:
    mock_run = MockRun(mock_run_id)
    mock_workspace.get_run.return_value = mock_run
    mock_fetch_run.return_value = mock_run

    aml_run = util.get_aml_run_from_run_id(mock_run_id, aml_workspace=mock_workspace)
    if run_id_type == util.RunId:
        mock_workspace.get_run.assert_called_with(mock_run_id)
    else:
        mock_fetch_run.assert_called_with(mock_workspace, mock_run_id)
    assert aml_run.id == mock_run_id


def _get_file_names(pref: str = "") -> List[str]:
    file_names = ["somepath.txt", "abc/someotherpath.txt", "abc/def/anotherpath.txt"]
    if len(pref) > 0:
        return [u for u in file_names if u.startswith(pref)]
    else:
        return file_names


def test_get_run_file_names() -> None:
    with patch("azureml.core.Run") as mock_run:
        expected_file_names = _get_file_names()
        mock_run.get_file_names.return_value = expected_file_names
        # check that we get the expected run paths if no filter is applied
        run_paths = util.get_run_file_names(mock_run)  # type: ignore
        assert len(run_paths) == len(expected_file_names)
        assert sorted(run_paths) == sorted(expected_file_names)

        # Now check we get the expected run paths if a filter is applied
        prefix = "abc"
        run_paths = util.get_run_file_names(mock_run, prefix=prefix)
        assert all([f.startswith(prefix) for f in run_paths])


def _mock_download_file(filename: str, output_file_path: Optional[Path] = None,
                        _validate_checksum: bool = False) -> None:
    """
    Creates an empty file at the given output_file_path
    """
    output_file_path = Path('test_output') if output_file_path is None else output_file_path
    output_file_path = Path(output_file_path) if not isinstance(output_file_path, Path) else output_file_path  # mypy
    output_file_path.parent.mkdir(exist_ok=True, parents=True)
    output_file_path.touch(exist_ok=True)


@pytest.mark.parametrize("dummy_env_vars", [{}, {util.ENV_LOCAL_RANK: "1"}])
@pytest.mark.parametrize("prefix", ["", "abc"])
def test_download_run_files(tmp_path: Path, dummy_env_vars: Dict[Optional[str], Optional[str]], prefix: str) -> None:
    # Assert that 'downloaded' paths don't exist to begin with
    dummy_paths = [Path(x) for x in _get_file_names(pref=prefix)]
    expected_paths = [tmp_path / dummy_path for dummy_path in dummy_paths]
    # Ensure that paths don't already exist
    [p.unlink() for p in expected_paths if p.exists()]  # type: ignore
    assert not any([p.exists() for p in expected_paths])

    mock_run = MockRun(run_id="id123")
    with mock.patch.dict(os.environ, dummy_env_vars):
        with patch("health_azure.utils.get_run_file_names") as mock_get_run_paths:
            mock_get_run_paths.return_value = dummy_paths  # type: ignore
            mock_run.download_file = MagicMock()  # type: ignore
            mock_run.download_file.side_effect = _mock_download_file

            util._download_files_from_run(mock_run, output_dir=tmp_path)
            # First test the case where is_local_rank_zero returns True
            if not any(dummy_env_vars):
                # Check that our mocked _download_file_from_run has been called once for each file
                assert sum([p.exists() for p in expected_paths]) == len(expected_paths)
            # Now test the case where is_local_rank_zero returns False - in this case nothing should be created
            else:
                assert not any([p.exists() for p in expected_paths])


@patch("health_azure.utils.get_workspace")
@patch("health_azure.utils.get_aml_run_from_run_id")
@patch("health_azure.utils._download_files_from_run")
def test_download_files_from_run_id(mock_download_run_files: MagicMock,
                                    mock_get_aml_run_from_run_id: MagicMock,
                                    mock_workspace: MagicMock) -> None:
    mock_run = {"id": "run123"}
    mock_get_aml_run_from_run_id.return_value = mock_run
    util.download_files_from_run_id("run123", Path(__file__))
    mock_download_run_files.assert_called_with(mock_run, Path(__file__), prefix="", validate_checksum=False)


@pytest.mark.parametrize("dummy_env_vars, expect_file_downloaded", [({}, True), ({util.ENV_LOCAL_RANK: "1"}, False)])
@patch("azureml.core.Run", MockRun)
def test_download_file_from_run(tmp_path: Path, dummy_env_vars: Dict[str, str], expect_file_downloaded: bool) -> None:
    dummy_filename = "filetodownload.txt"
    expected_file_path = tmp_path / dummy_filename

    # mock the method 'download_file' on the AML Run class and assert it gets called with the expected params
    mock_run = MockRun(run_id="id123")
    mock_run.download_file = MagicMock(return_value=None)  # type: ignore
    mock_run.download_file.side_effect = _mock_download_file

    with mock.patch.dict(os.environ, dummy_env_vars):
        _ = util._download_file_from_run(mock_run, dummy_filename, expected_file_path)

        if expect_file_downloaded:
            mock_run.download_file.assert_called_with(dummy_filename, output_file_path=str(expected_file_path),
                                                      _validate_checksum=False)
            assert expected_file_path.exists()
        else:
            assert not expected_file_path.exists()


def test_download_file_from_run_remote(tmp_path: Path) -> None:
    # This test will create a Run in your workspace (using only local compute)
    ws = DEFAULT_WORKSPACE.workspace
    experiment = Experiment(ws, AML_TESTS_EXPERIMENT)
    config = ScriptRunConfig(
        source_directory=".",
        command=["cd ."],  # command that does nothing
        compute_target="local"
    )
    run = experiment.submit(config)

    file_to_upload = tmp_path / "dummy_file.txt"
    file_contents = "Hello world"
    file_to_upload.write_text(file_contents)

    # This should store the file in outputs
    run.upload_file("dummy_file", str(file_to_upload))

    output_file_path = tmp_path / "downloaded_file.txt"
    assert not output_file_path.exists()

    start_time = time.perf_counter()
    _ = util._download_file_from_run(run, "dummy_file", output_file_path)
    end_time = time.perf_counter()
    time_dont_validate_checksum = end_time - start_time

    assert output_file_path.exists()
    assert output_file_path.read_text() == file_contents

    # Now delete the file and try again with _validate_checksum == True
    output_file_path.unlink()
    assert not output_file_path.exists()
    start_time = time.perf_counter()
    _ = util._download_file_from_run(run, "dummy_file", output_file_path, validate_checksum=True)
    end_time = time.perf_counter()
    time_validate_checksum = end_time - start_time

    assert output_file_path.exists()
    assert output_file_path.read_text() == file_contents

    logging.info(f"Time to download file without checksum: {time_dont_validate_checksum} vs time with"
                 f"validation {time_validate_checksum}.")


def test_download_run_file_during_run(tmp_path: Path) -> None:
    # This test will create a Run in your workspace (using only local compute)

    expected_file_path = tmp_path / "azureml-logs"
    # Check that at first the path to downloaded logs doesnt exist (will be created by the later test script)
    assert not expected_file_path.exists()

    ws = DEFAULT_WORKSPACE.workspace

    # call the script here
    extra_options = {
        "imports": """
from azureml.core import Run
from health_azure.utils import _download_files_from_run""",
        "args": """
    parser.add_argument("--output_path", type=str, required=True)
        """,
        "body": """
    output_path = Path(args.output_path)
    output_path.mkdir(exist_ok=True)

    run_ctx = Run.get_context()
    available_files = run_ctx.get_file_names()
    print(f"available files: {available_files}")
    first_file_name = available_files[0]
    output_file_path = output_path / first_file_name

    _download_files_from_run(run_ctx, output_path, prefix=first_file_name)

    print(f"Downloaded file {first_file_name} to location {output_file_path}")
        """
    }

    extra_args = ["--output_path", 'outputs']
    render_and_run_test_script(tmp_path, RunTarget.AZUREML, extra_options, extra_args, True)

    run = util.get_most_recent_run(run_recovery_file=tmp_path / himl.RUN_RECOVERY_FILE,
                                   workspace=ws)
    assert run.status == "Completed"


def test_is_global_rank_zero() -> None:
    with mock.patch.dict(os.environ, {util.ENV_NODE_RANK: "0", util.ENV_GLOBAL_RANK: "0", util.ENV_LOCAL_RANK: "0"}):
        assert not util.is_global_rank_zero()

    with mock.patch.dict(os.environ, {util.ENV_GLOBAL_RANK: "0", util.ENV_LOCAL_RANK: "0"}):
        assert not util.is_global_rank_zero()

    with mock.patch.dict(os.environ, {util.ENV_NODE_RANK: "0"}):
        assert util.is_global_rank_zero()


def test_is_local_rank_zero() -> None:
    # mock the environment variables
    with mock.patch.dict(os.environ, {}):
        assert util.is_local_rank_zero()

    with mock.patch.dict(os.environ, {util.ENV_GLOBAL_RANK: "1", util.ENV_LOCAL_RANK: "1"}):
        assert not util.is_local_rank_zero()


@pytest.mark.parametrize("dummy_recovery_id, expected_id_type", [
    ("expt:run_abc_1234", util.RunRecoveryId),
    ("['expt:abc_432','expt2:def_111']", util.RunRecoveryId),
    ("run_ghi_1234", util.RunId),
    ("['run_jkl_1234','run_mno_7654']", util.RunId)
])
def test_get_run_source(dummy_recovery_id: str,
                        expected_id_type: Union[util.RunId, util.RunRecoveryId]) -> None:
    arguments = ["", "--run", dummy_recovery_id]
    with patch.object(sys, "argv", arguments):

        run_source = util.AmlRunScriptConfig.parse_args()

        if isinstance(run_source.run, List):
            assert isinstance(run_source.run[0], expected_id_type)
        else:
            assert isinstance(run_source.run, expected_id_type)


@pytest.mark.parametrize("overwrite", [True, False])
@pytest.mark.parametrize("show_progress", [True, False])
def test_download_from_datastore(tmp_path: Path, overwrite: bool, show_progress: bool) -> None:
    """
    Test that download_from_datastore successfully downloads file from Blob Storage.
    Note that this will temporarily upload a file to the default datastore of the default workspace -
    (determined by either a config.json file, or by specifying workspace settings in the environment variables).
    After the test has completed, the blob will be deleted.
    """
    ws = DEFAULT_WORKSPACE.workspace
    default_datastore: AzureBlobDatastore = ws.get_default_datastore()
    dummy_file_content = "Hello world"
    local_data_path = tmp_path / "local_data"
    local_data_path.mkdir()
    test_data_path_remote = "test_data/abc"

    # Create dummy data files and upload to datastore (checking they are uploaded)
    dummy_filenames = []
    num_dummy_files = 2
    for i in range(num_dummy_files):
        dummy_filename = f"dummy_data_{i}.txt"
        dummy_filenames.append(dummy_filename)
        data_to_upload_path = local_data_path / dummy_filename
        data_to_upload_path.touch()
        data_to_upload_path.write_text(dummy_file_content)
    default_datastore.upload(str(local_data_path), test_data_path_remote, overwrite=False)
    existing_blobs = list(default_datastore.blob_service.list_blobs(prefix=test_data_path_remote,
                                                                    container_name=default_datastore.container_name))
    assert len(existing_blobs) == num_dummy_files

    # Check that the file doesn't currently exist at download location
    downloaded_data_path = tmp_path / "downloads"
    assert not downloaded_data_path.exists()

    # Now attempt to download
    util.download_from_datastore(default_datastore.name, test_data_path_remote, downloaded_data_path,
                                 aml_workspace=ws, overwrite=overwrite, show_progress=show_progress)
    expected_local_download_dir = downloaded_data_path / test_data_path_remote
    assert expected_local_download_dir.exists()
    expected_download_paths = [expected_local_download_dir / dummy_filename for dummy_filename in dummy_filenames]
    assert all([p.exists() for p in expected_download_paths])

    # Delete the file from Blob Storage
    container = default_datastore.container_name
    existing_blobs = list(default_datastore.blob_service.list_blobs(prefix=test_data_path_remote,
                                                                    container_name=container))
    for existing_blob in existing_blobs:
        default_datastore.blob_service.delete_blob(container_name=container, blob_name=existing_blob.name)


@pytest.mark.parametrize("overwrite", [True, False])
@pytest.mark.parametrize("show_progress", [True, False])
def test_upload_to_datastore(tmp_path: Path, overwrite: bool, show_progress: bool) -> None:
    """
    Test that upload_to_datastore successfully uploads a file to Blob Storage.
    Note that this will temporarily upload a file to the default datastore of the default workspace -
    (determined by either a config.json file, or by specifying workspace settings in the environment variables).
    After the test has completed, the blob will be deleted.
    """
    ws = DEFAULT_WORKSPACE.workspace
    default_datastore: AzureBlobDatastore = ws.get_default_datastore()
    container = default_datastore.container_name
    dummy_file_content = "Hello world"

    remote_data_dir = "test_data"
    dummy_file_name = Path("abc/uploaded_file.txt")
    expected_remote_path = Path(remote_data_dir) / dummy_file_name.name

    # check that the file doesnt already exist in Blob Storage
    existing_blobs = list(default_datastore.blob_service.list_blobs(prefix=str(expected_remote_path.as_posix()),
                                                                    container_name=container))
    assert len(existing_blobs) == 0

    # Create a dummy data file and upload to datastore
    data_to_upload_path = tmp_path / dummy_file_name
    data_to_upload_path.parent.mkdir(exist_ok=True, parents=True)
    data_to_upload_path.touch()
    data_to_upload_path.write_text(dummy_file_content)

    util.upload_to_datastore(default_datastore.name, data_to_upload_path.parent, Path(remote_data_dir),
                             aml_workspace=ws, overwrite=overwrite, show_progress=show_progress)
    existing_blobs = list(default_datastore.blob_service.list_blobs(prefix=str(expected_remote_path.as_posix()),
                                                                    container_name=container))
    assert len(existing_blobs) == 1

    # delete the blob from Blob Storage
    existing_blob: Blob = existing_blobs[0]
    default_datastore.blob_service.delete_blob(container_name=container, blob_name=existing_blob.name)


@pytest.mark.parametrize("arguments, run_id", [
    (["", "--run", "run_abc_123"], util.RunId("run_abc_123")),
    (["", "--run", "run_abc_123,run_def_456"], [util.RunId("run_abc_123"), util.RunId("run_def_456")]),
    (["", "--run", "expt_name:run_abc_123"], util.RunRecoveryId("expt_name:run_abc_123")),
])
def test_script_config_run_src(arguments: List[str], run_id: Union[List[util.RunId], util.RunId]) -> None:
    with patch.object(sys, "argv", arguments):
        script_config = util.AmlRunScriptConfig.parse_args()

        if isinstance(run_id, list):
            for script_config_run, expected_run in zip(script_config.run, run_id):
                assert script_config_run.id == expected_run.val
        else:
            assert script_config.run.id == run_id.id


@patch("health_azure.utils.download_files_from_run_id")
@patch("health_azure.utils.get_workspace")
def test_checkpoint_download(mock_get_workspace: MagicMock, mock_download_files: MagicMock) -> None:
    mock_workspace = MagicMock()
    mock_get_workspace.return_value = mock_workspace
    dummy_run_id = "run_def_456"
    prefix = "path/to/file"
    output_file_dir = Path("my_ouputs")
    util.download_checkpoints_from_run_id(dummy_run_id, prefix, output_file_dir, aml_workspace=mock_workspace)
    mock_download_files.assert_called_once_with(dummy_run_id, output_file_dir, prefix=prefix,
                                                workspace=mock_workspace, validate_checksum=True)


@pytest.mark.slow
def test_checkpoint_download_remote(tmp_path: Path) -> None:
    """
    Creates a large dummy file (around 250 MB) and ensures we can upload it to a Run and subsequently download
    with no issues, thus replicating the behaviour of downloading a large checkpoint file.
    """
    num_dummy_files = 1
    prefix = "outputs/checkpoints/"

    ws = DEFAULT_WORKSPACE.workspace
    experiment = Experiment(ws, AML_TESTS_EXPERIMENT)
    config = ScriptRunConfig(
        source_directory=".",
        command=["cd ."],  # command that does nothing
        compute_target="local"
    )
    run = experiment.submit(config)

    file_contents = "Hello world"
    for i in range(num_dummy_files):
        file_name = f"dummy_checkpoint_{i}.txt"
        large_file_path = tmp_path / file_name
        with open(str(large_file_path), "wb") as f_path:
            f_path.seek((1024 * 1024 * 240) - 1)
            f_path.write(bytearray(file_contents, encoding="UTF-8"))

        file_size = large_file_path.stat().st_size
        logging.info(f"File {i} size: {file_size}")

        local_path = str(large_file_path)
        run.upload_file(prefix + file_name, local_path)

    # Check the local dir is empty to begin with
    output_file_dir = tmp_path
    assert not (output_file_dir / prefix).exists()

    whole_file_path = prefix + file_name
    start_time = time.perf_counter()
    util.download_checkpoints_from_run_id(run.id, whole_file_path, output_file_dir, aml_workspace=ws)
    end_time = time.perf_counter()
    time_taken = end_time - start_time
    logging.info(f"Time taken to download file: {time_taken}")

    download_file_path = output_file_dir / prefix / "dummy_checkpoint_0.txt"
    assert (output_file_dir / prefix).is_dir()
    assert len(list((output_file_dir / prefix).iterdir())) == num_dummy_files
    with open(str(download_file_path), "rb") as f_path:
        for line in f_path:
            chunk = line.strip(b'\x00')
            if chunk:
                found_file_contents = chunk.decode("utf-8")
                break

    assert found_file_contents == file_contents

    # Delete the file downloaded file and check that download_checkpoints also works on a single checkpoint file
    download_file_path.unlink()
    assert not download_file_path.exists()

    util.download_checkpoints_from_run_id(run.id, whole_file_path, output_file_dir, aml_workspace=ws)
    assert download_file_path.exists()
    with open(str(download_file_path), "rb") as f_path:
        for line in f_path:
            chunk = line.strip(b'\x00')
            if chunk:
                found_file_contents = chunk.decode("utf-8")
                break
    assert found_file_contents == file_contents


@pytest.mark.parametrize(("available", "initialized", "expected_barrier_called"),
                         [(False, True, False),
                          (True, False, False),
                          (False, False, False),
                          (True, True, True)])
@pytest.mark.fast
def test_torch_barrier(available: bool,
                       initialized: bool,
                       expected_barrier_called: bool) -> None:
    distributed = mock.MagicMock()
    distributed.is_available.return_value = available
    distributed.is_initialized.return_value = initialized
    distributed.barrier = mock.MagicMock()
    with mock.patch.dict("sys.modules", {"torch": mock.MagicMock(distributed=distributed)}):
        util.torch_barrier()
        if expected_barrier_called:
            distributed.barrier.assert_called_once()
        else:
            assert distributed.barrier.call_count == 0


class ParamEnum(Enum):
    EnumValue1 = "1",
    EnumValue2 = "2"


class IllegalCustomTypeNoFromString(param.Parameter):
    def _validate(self, val: Any) -> None:
        super()._validate(val)


class IllegalCustomTypeNoValidate(util.CustomTypeParam):
    def from_string(self, x: str) -> Any:
        return x


class ParamClass(util.GenericConfig):
    name: str = param.String(None, doc="Name")
    seed: int = param.Integer(42, doc="Seed")
    flag: bool = param.Boolean(False, doc="Flag")
    not_flag: bool = param.Boolean(True, doc="Not Flag")
    number: float = param.Number(3.14)
    integers: List[int] = param.List(None, class_=int)
    optional_int: Optional[int] = param.Integer(None, doc="Optional int")
    optional_float: Optional[float] = param.Number(None, doc="Optional float")
    floats: List[float] = param.List(None, class_=float)
    tuple1: Tuple[int, float] = param.NumericTuple((1, 2.3), length=2, doc="Tuple")
    int_tuple: Tuple[int, int, int] = util.IntTuple((1, 1, 1), length=3, doc="Integer Tuple")
    enum: ParamEnum = param.ClassSelector(default=ParamEnum.EnumValue1, class_=ParamEnum, instantiate=False)
    readonly: str = param.String("Nope", readonly=True)
    _non_override: str = param.String("Nope")
    constant: str = param.String("Nope", constant=True)
    other_args = util.ListOrDictParam(None, doc="List or dictionary of other args")


class ClassFrom(param.Parameterized):
    foo = param.String("foo")
    bar = param.Integer(1)
    baz = param.String("baz")
    _private = param.String("private")
    constant = param.String("constant", constant=True)


class ClassTo(param.Parameterized):
    foo = param.String("foo2")
    bar = param.Integer(2)
    _private = param.String("private2")
    constant = param.String("constant2", constant=True)


class NotParameterized:
    foo = 1


@pytest.mark.fast
def test_overridable_parameter() -> None:
    """
    Test to check overridable parameters are correctly identified.
    """
    param_dict = ParamClass.get_overridable_parameters()
    assert "name" in param_dict
    assert "flag" in param_dict
    assert "not_flag" in param_dict
    assert "seed" in param_dict
    assert "number" in param_dict
    assert "integers" in param_dict
    assert "optional_int" in param_dict
    assert "optional_float" in param_dict
    assert "tuple1" in param_dict
    assert "int_tuple" in param_dict
    assert "enum" in param_dict
    assert "other_args" in param_dict

    assert "readonly" not in param_dict
    assert "_non_override" not in param_dict
    assert "constant" not in param_dict


@pytest.mark.fast
def test_parser_defaults() -> None:
    """
    Check that default values are created as expected, and that the non-overridable parameters
    are omitted.
    """
    defaults = vars(ParamClass.create_argparser().parse_args([]))
    assert defaults["seed"] == 42
    assert defaults["tuple1"] == (1, 2.3)
    assert defaults["int_tuple"] == (1, 1, 1)
    assert defaults["enum"] == ParamEnum.EnumValue1
    assert not defaults["flag"]
    assert defaults["not_flag"]
    assert "readonly" not in defaults
    assert "constant" not in defaults
    assert "_non_override" not in defaults
    # We can't test if all invalid cases are handled because argparse call sys.exit
    # upon errors.


def check_parsing_succeeds(arg: List[str], expected_key: str, expected_value: Any) -> None:
    parsed = ParamClass.parse_args(arg)
    assert getattr(parsed, expected_key) == expected_value


def check_parsing_fails(arg: List[str], expected_key: Optional[str] = None, expected_value: Optional[Any] = None
                        ) -> None:
    with pytest.raises(SystemExit) as e:
        ParamClass.parse_args(arg)
    assert e.type == SystemExit
    assert e.value.code == 2


@pytest.mark.fast
@pytest.mark.parametrize("args, expected_key, expected_value, expected_pass", [
    (["--name=foo"], "name", "foo", True),
    (["--seed", "42"], "seed", 42, True),
    (["--seed", ""], "seed", 42, True),
    (["--number", "2.17"], "number", 2.17, True),
    (["--number", ""], "number", 3.14, True),
    (["--integers", "1,2,3"], "integers", [1, 2, 3], True),
    (["--optional_int", ""], "optional_int", None, True),
    (["--optional_int", "2"], "optional_int", 2, True),
    (["--optional_float", ""], "optional_float", None, True),
    (["--optional_float", "3.14"], "optional_float", 3.14, True),
    (["--tuple1", "1,2"], "tuple1", (1, 2.0), True),
    (["--int_tuple", "1,2,3"], "int_tuple", (1, 2, 3), True),
    (["--enum=2"], "enum", ParamEnum.EnumValue2, True),
    (["--floats=1,2,3.14"], "floats", [1., 2., 3.14], True),
    (["--integers=1,2,3"], "integers", [1, 2, 3], True),
    (["--flag"], "flag", True, True),
    (["--no-flag"], None, None, False),
    (["--not_flag"], None, None, False),
    (["--no-not_flag"], "not_flag", False, True),
    (["--not_flag=false", "--no-not_flag"], None, None, False),
    (["--flag=Falsf"], None, None, False),
    (["--flag=Truf"], None, None, False),
    (["--other_args={'learning_rate': 0.5}"], "other_args", {'learning_rate': 0.5}, True),
])
def test_create_parser(args: List[str], expected_key: str, expected_value: Any, expected_pass: bool) -> None:
    """
    Check that parse_args works as expected, with both non default and default values.
    """
    if expected_pass:
        check_parsing_succeeds(args, expected_key, expected_value)
    else:
        check_parsing_fails(args)


@pytest.mark.fast
@pytest.mark.parametrize("flag, expected_value", [
    ('on', True), ('t', True), ('true', True), ('y', True), ('yes', True), ('1', True),
    ('off', False), ('f', False), ('false', False), ('n', False), ('no', False), ('0', False)
])
def test_parsing_bools(flag: str, expected_value: bool) -> None:
    """
    Check all the ways of passing in True and False, with and without the first letter capitialized
    """
    check_parsing_succeeds([f"--flag={flag}"], "flag", expected_value)
    check_parsing_succeeds([f"--flag={flag.capitalize()}"], "flag", expected_value)
    check_parsing_succeeds([f"--not_flag={flag}"], "not_flag", expected_value)
    check_parsing_succeeds([f"--not_flag={flag.capitalize()}"], "not_flag", expected_value)


@pytest.mark.fast
@patch("health_azure.utils.GenericConfig.report_on_overrides")
@patch("health_azure.utils.GenericConfig.validate")
def test_apply_overrides(mock_validate: MagicMock, mock_report_on_overrides: MagicMock) -> None:
    """
    Test that overrides are applied correctly, ond only to overridable parameters,
    """
    m = ParamClass()
    overrides = {"name": "newName", "int_tuple": (0, 1, 2)}
    actual_overrides = m.apply_overrides(overrides)
    assert actual_overrides == overrides
    assert all([x == i and isinstance(x, int) for i, x in enumerate(m.int_tuple)])
    assert m.name == "newName"
    # Attempt to change seed and constant, but the latter should be ignored.
    change_seed = {"seed": 123}
    old_constant = m.constant
    changes2 = m.apply_overrides({**change_seed, "constant": "Nothing"})  # type: ignore
    assert changes2 == change_seed
    assert m.seed == 123
    assert m.constant == old_constant

    # Check the call count of mock_validate and check it doesn't increase if should_validate is set to False
    # and that setting this flag doesn't affect on the outputs
    mock_validate_call_count = mock_validate.call_count
    actual_overrides = m.apply_overrides(values=overrides, should_validate=False)
    assert actual_overrides == overrides
    assert mock_validate.call_count == mock_validate_call_count

    # Check that report_on_overrides has not yet been called, but is called if keys_to_ignore is not None
    # and that setting this flag doesn't affect on the outputs
    assert mock_report_on_overrides.call_count == 0
    actual_overrides = m.apply_overrides(values=overrides, keys_to_ignore={"name"})
    assert actual_overrides == overrides
    assert mock_report_on_overrides.call_count == 1


def test_report_on_overrides() -> None:
    m = ParamClass()
    overrides = {"name": "newName", "int_tuple": (0, 1, 2)}
    m.report_on_overrides(overrides)


@pytest.mark.fast
@pytest.mark.parametrize("value_idx_0", [1.0, 1])
@pytest.mark.parametrize("value_idx_1", [2.0, 2])
@pytest.mark.parametrize("value_idx_2", [3.0, 3])
def test_int_tuple_validation(value_idx_0: Any, value_idx_1: Any, value_idx_2: Any) -> None:
    """
    Test integer tuple parameter is validated correctly.
    """
    m = ParamClass()
    val = (value_idx_0, value_idx_1, value_idx_2)
    if not all([isinstance(x, int) for x in val]):
        with pytest.raises(ValueError):
            m.int_tuple = (value_idx_0, value_idx_1, value_idx_2)
    else:
        m.int_tuple = (value_idx_0, value_idx_1, value_idx_2)


@pytest.mark.fast
def test_create_from_matching_params() -> None:
    """
    Test if Parameterized objects can be cloned by looking at matching fields.
    """
    class_from = ClassFrom()
    class_to = util.create_from_matching_params(class_from, cls_=ClassTo)
    assert isinstance(class_to, ClassTo)
    assert class_to.foo == "foo"
    assert class_to.bar == 1
    # Constant fields should not be touched
    assert class_to.constant == "constant2"
    # Private fields must be copied over.
    assert class_to._private == "private"
    # Baz is only present in the "from" object, and should not be copied to the new object
    assert not hasattr(class_to, "baz")

    with pytest.raises(ValueError) as ex:
        util.create_from_matching_params(class_from, NotParameterized)
    assert "subclass of param.Parameterized" in str(ex)
    assert "NotParameterized" in str(ex)


def test_parse_illegal_params() -> None:
    with pytest.raises(ValueError) as e:
        ParamClass(readonly="abc")
        assert "cannot be overridden" in str(e.value)


def test_parse_throw_if_unknown() -> None:
    with pytest.raises(ValueError) as e:
        ParamClass(throw_if_unknown_param=True, idontexist="hello")
        assert "parameters do not exist" in str(e.value)


@pytest.mark.parametrize("should_validate, expected_call_count", [(None, 0), (False, 0), (True, 1)])
@patch("health_azure.utils.GenericConfig.validate")
def test_config_validate(mock_validate: MagicMock, should_validate: Optional[bool], expected_call_count: int) -> None:
    _ = ParamClass(should_validate=should_validate)
    assert mock_validate.call_count == expected_call_count


def test_config_add_and_validate() -> None:
    config = ParamClass.parse_args([])
    assert config.name == "ParamClass"
    config.add_and_validate({"name": "foo"})
    assert config.name == "foo"

    assert hasattr(config, "new_property") is False
    config.add_and_validate({"new_property": "bar"})
    assert hasattr(config, "new_property") is True
    assert config.new_property == "bar"


class CustomType(util.CustomTypeParam):
    def __init__(self) -> None:
        pass

    def _validate(self) -> None:
        pass

    def from_string(self, x: str) -> None:
        pass


class IllegalParamClassNoString(util.GenericConfig):
    custom_type_no_from_string = IllegalCustomTypeNoFromString(
        None, doc="This should fail since from_string method is missing"
    )


def test_cant_parse_param_type() -> None:
    """
    Assert that a TypeError is raised when trying to add a custom type with no from_string method as an argument
    """
    with pytest.raises(TypeError) as e:
        IllegalParamClassNoString.parse_args([])
        assert "is not supported" in str(e.value)
