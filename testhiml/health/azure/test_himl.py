#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
Tests for hi-ml.
"""
import logging
import os
import pathlib
import subprocess
import sys
from pathlib import Path, PosixPath
from typing import Dict, List, Tuple
from unittest import mock
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from azureml.core import RunConfiguration, Workspace
from azureml.data.dataset_consumption_config import DatasetConsumptionConfig

import health.azure.himl as himl
from conftest import check_config_json
from health.azure.azure_util import EXPERIMENT_RUN_SEPARATOR
from health.azure.datasets import DatasetConfig, _input_dataset_key, _output_dataset_key
from testhiml.health.azure.test_data.make_tests import render_environment_yaml, render_test_script
from testhiml.health.azure.util import get_most_recent_run, repository_root

INEXPENSIVE_TESTING_CLUSTER_NAME = "lite-testing-ds2"

logger = logging.getLogger('test.health.azure')
logger.setLevel(logging.DEBUG)


# region Small fast local unit tests

@pytest.mark.fast
def test_submit_to_azure_if_needed_returns_immediately() -> None:
    """
    Test that himl.submit_to_azure_if_needed can be called, and returns immediately.
    """
    with mock.patch("sys.argv", ["", "--azureml"]):
        with pytest.raises(Exception) as ex:
            himl.submit_to_azure_if_needed(
                aml_workspace=None,
                workspace_config_path=None,
                entry_script=Path(__file__),
                compute_cluster_name="foo",
                conda_environment_file=Path("env.yaml"),
                snapshot_root_directory=Path(__file__).parent)
        assert "Cannot glean workspace config from parameters" in str(ex)
    with mock.patch("sys.argv", [""]):
        result = himl.submit_to_azure_if_needed(
            entry_script=Path(__file__),
            compute_cluster_name="foo",
            conda_environment_file=Path("env.yml"))
        assert isinstance(result, himl.AzureRunInfo)
        assert not result.is_running_in_azure
        assert result.run is None


@pytest.mark.fast
@patch("health.azure.himl.Run")
def test_write_run_recovery_file(mock_run: mock.MagicMock) -> None:
    # recovery file does not exist:
    mock_run.id = uuid4().hex
    mock_run.experiment.name = uuid4().hex
    expected_run_recovery_id = mock_run.experiment.name + EXPERIMENT_RUN_SEPARATOR + mock_run.id
    himl._write_run_recovery_file(mock_run)
    recovery_file_text = Path(himl.RUN_RECOVERY_FILE).read_text()
    assert expected_run_recovery_id == recovery_file_text
    # recovery file exists from above:
    mock_run.id = uuid4().hex
    mock_run.experiment.name = uuid4().hex
    himl._write_run_recovery_file(mock_run)
    recovery_file_text = Path(himl.RUN_RECOVERY_FILE).read_text()
    assert expected_run_recovery_id != recovery_file_text


@pytest.mark.fast
@patch("azureml.data.OutputFileDatasetConfig")
@patch("health.azure.himl.DatasetConsumptionConfig")
@patch("health.azure.himl.Workspace")
@patch("health.azure.himl.DatasetConfig")
def test_to_datasets(
        mock_dataset_config: mock.MagicMock,
        mock_workspace: mock.MagicMock,
        mock_dataset_consumption_config: mock.MagicMock,
        mock_output_file_dataset_config: mock.MagicMock) -> None:
    def to_input_dataset(workspace: Workspace, dataset_index: int, ) -> DatasetConsumptionConfig:
        return mock_dataset_consumption_config

    def to_output_dataset(workspace: Workspace, dataset_index: int, ) -> DatasetConsumptionConfig:
        return mock_output_file_dataset_config

    mock_dataset_consumption_config.name = "A Consumption Config"
    mock_output_file_dataset_config.name = "An Output File Dataset Config"
    mock_dataset_config.to_input_dataset = to_input_dataset
    mock_dataset_config.to_output_dataset = to_output_dataset
    cleaned_input_datasets = [mock_dataset_config]
    # TODO: There's something wrong here. We feed 2 datasets in, but only get one out. When creating the dictionary,
    # we should check if the names are unique. In practice, this is unlikely to happen because the names are
    # INPUT_0, INPUT_1, etc
    cleaned_output_datasets = [mock_dataset_config, mock_dataset_config]
    inputs, outputs = himl.convert_himl_to_azureml_datasets(
        cleaned_input_datasets=cleaned_input_datasets,
        cleaned_output_datasets=cleaned_output_datasets,
        workspace=mock_workspace)
    assert len(inputs) == 1
    assert len(outputs) == 1
    assert inputs[mock_dataset_consumption_config.name] == mock_dataset_consumption_config
    assert outputs[mock_output_file_dataset_config.name] == mock_output_file_dataset_config


@pytest.mark.fast
@patch("health.azure.himl.register_environment")
@patch("health.azure.himl.create_python_environment")
@patch("health.azure.himl.Workspace")
def test_create_run_configuration_fails(
        mock_workspace: mock.MagicMock,
        _: mock.MagicMock,
        __: mock.MagicMock,
        ) -> None:
    existing_compute_target = "this_does_exist"
    mock_workspace.compute_targets = {existing_compute_target: 123}
    with pytest.raises(ValueError) as e:
        himl.create_run_configuration(
            compute_cluster_name="b",
            workspace=mock_workspace)
    assert "One of the two arguments 'aml_environment_name' or 'conda_environment_file' must be given." == str(e.value)
    with pytest.raises(ValueError) as e:
        himl.create_run_configuration(
            conda_environment_file=Path(__file__),
            compute_cluster_name="b",
            workspace=mock_workspace)
    assert "Could not find the compute target b in the AzureML workspace" in str(e.value)
    assert existing_compute_target in str(e.value)


@pytest.mark.fast
@patch("health.azure.himl.DockerConfiguration")
@patch("health.azure.datasets.DatasetConfig.to_output_dataset")
@patch("health.azure.datasets.DatasetConfig.to_input_dataset")
@patch("health.azure.himl.Environment.get")
@patch("health.azure.himl.Workspace")
def test_create_run_configuration(
        mock_workspace: mock.MagicMock,
        mock_environment_get: mock.MagicMock,
        mock_to_input_dataset: mock.MagicMock,
        mock_to_output_dataset: mock.MagicMock,
        mock_docker_configuration: mock.MagicMock,
) -> None:
    existing_compute_target = "this_does_exist"
    mock_env_name = "Mock Env"
    mock_environment_get.return_value = mock_env_name
    mock_workspace.compute_targets = {existing_compute_target: 123}
    aml_input_dataset = MagicMock()
    aml_input_dataset.name = "dataset_in"
    aml_output_dataset = MagicMock()
    aml_output_dataset.name = "dataset_out"
    mock_to_input_dataset.return_value = aml_input_dataset
    mock_to_output_dataset.return_value = aml_output_dataset
    run_config = himl.create_run_configuration(
        workspace=mock_workspace,
        compute_cluster_name=existing_compute_target,
        aml_environment_name="foo",
        num_nodes=10,
        max_run_duration="1h",
        input_datasets=[DatasetConfig(name="input1")],
        output_datasets=[DatasetConfig(name="output1")],
        docker_shm_size="2g"
    )
    assert isinstance(run_config, RunConfiguration)
    assert run_config.target == existing_compute_target
    assert run_config.environment == mock_env_name
    assert run_config.node_count == 10
    assert run_config.mpi.node_count == 10
    assert run_config.max_run_duration_seconds == 60 * 60
    assert run_config.data == {"dataset_in": aml_input_dataset}
    assert run_config.output_data == {"dataset_out": aml_output_dataset}
    mock_docker_configuration.assert_called_once()
    run_config = himl.create_run_configuration(
        workspace=mock_workspace,
        compute_cluster_name=existing_compute_target,
        aml_environment_name="foo",
        num_nodes=10,
        input_datasets=[DatasetConfig(name="input1")],
        output_datasets=[DatasetConfig(name="output1")],
        docker_shm_size="2g"
    )
    assert run_config.max_run_duration_seconds is None


@pytest.mark.fast
def test_invalid_entry_script(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / uuid4().hex
    snapshot_dir.mkdir(exist_ok=False)
    ok_entry_script = snapshot_dir / "entry_script.py"
    ok_entry_script.write_text("print('hello world')\n")

    run_config = himl.create_script_run(
        entry_script=ok_entry_script,
        snapshot_root_directory=snapshot_dir,
        script_params=[])
    assert run_config.script == str(ok_entry_script.relative_to(snapshot_dir))

    problem_entry_script_dir = tmp_path / uuid4().hex
    problem_entry_script_dir.mkdir(exist_ok=False)
    problem_entry_script = problem_entry_script_dir / "entry_script.py"
    problem_entry_script.write_text("print('hello world')\n")

    with pytest.raises(ValueError) as e:
        himl.create_script_run(
            entry_script=problem_entry_script,
            snapshot_root_directory=snapshot_dir,
            script_params=[])
    assert "entry script must be inside of the snapshot root directory" in str(e)

    with mock.patch("sys.argv", ["foo"]):
        script_run = himl.create_script_run()
        assert script_run.source_directory == str(Path.cwd())
        assert script_run.script == "foo"
        assert script_run.arguments == []

    # Entry scripts where the path is not absolute should be left unchanged
    script_run = himl.create_script_run(entry_script="some_string", script_params=["--foo"])
    assert script_run.script == "some_string"
    assert script_run.arguments == ["--foo"]


@pytest.mark.fast
def test_get_script_params() -> None:
    expected_params = ["a string"]
    assert expected_params == himl._get_script_params(expected_params)
    with mock.patch("sys.argv", ["", "a string", "--azureml"]):
        assert expected_params == himl._get_script_params()
    with mock.patch("sys.argv", ["", "a string"]):
        assert expected_params == himl._get_script_params()


@pytest.mark.fast
@patch("health.azure.himl.Workspace.from_config")
@patch("health.azure.himl.get_authentication")
@patch("health.azure.himl.Workspace")
def test_get_workspace(
        mock_workspace: mock.MagicMock,
        mock_get_authentication: mock.MagicMock,
        mock_from_config: mock.MagicMock) -> None:
    workspace = himl._get_workspace(mock_workspace, None)
    assert workspace == mock_workspace
    mock_get_authentication.return_value = None
    _ = himl._get_workspace(None, Path(__file__))
    assert mock_from_config.called


@pytest.mark.fast
@patch("health.azure.himl.Run")
@patch("health.azure.himl.Workspace")
@patch("health.azure.himl._generate_azure_datasets")
@patch("health.azure.himl.is_running_in_azure")
def test_submit_to_azure_if_needed_azure_return(
        mock_is_running_in_azure: mock.MagicMock,
        mock_generate_azure_datasets: mock.MagicMock,
        mock_workspace: mock.MagicMock,
        mock_run: mock.MagicMock) -> None:
    mock_is_running_in_azure.return_value = True
    expected_run_info = himl.AzureRunInfo(
        run=mock_run,
        input_datasets=[],
        output_datasets=[],
        is_running_in_azure=True,
        output_folder=Path.cwd(),
        logs_folder=Path.cwd())
    mock_generate_azure_datasets.return_value = expected_run_info
    with mock.patch("sys.argv", ["", "--azureml"]):
        run_info = himl.submit_to_azure_if_needed(
            aml_workspace=mock_workspace,
            entry_script=Path(__file__),
            compute_cluster_name="foo",
            conda_environment_file=Path("env.yml"))
    assert run_info == expected_run_info


@pytest.mark.fast
@patch("health.azure.himl.DatasetConfig")
@patch("health.azure.himl.RUN_CONTEXT")
def test_generate_azure_datasets(
        mock_run_context: mock.MagicMock,
        mock_dataset_config: mock.MagicMock) -> None:
    mock_run_context.input_datasets = {}
    mock_run_context.output_datasets = {}
    for i in range(4):
        mock_run_context.input_datasets[_input_dataset_key(i)] = f"input_{i}"
        mock_run_context.output_datasets[_output_dataset_key(i)] = f"output_{i}"
    run_info = himl._generate_azure_datasets(
        cleaned_input_datasets=[mock_dataset_config] * 2,
        cleaned_output_datasets=[mock_dataset_config] * 3)
    assert run_info.is_running_in_azure
    assert len(run_info.input_datasets) == 2
    assert len(run_info.output_datasets) == 3
    for i, d in enumerate(run_info.input_datasets):
        assert isinstance(d, Path)
        assert str(d) == f"input_{i}"
    for i, d in enumerate(run_info.output_datasets):
        assert isinstance(d, Path)
        assert str(d) == f"output_{i}"


@pytest.mark.fast
def test_append_to_amlignore(tmp_path: Path) -> None:
    # If there is no .amlignore file before the test, there should be none afterwards
    amlignore_path = tmp_path / Path(uuid4().hex)
    with himl.append_to_amlignore(
            amlignore=amlignore_path,
            lines_to_append=["1st line", "2nd line"]):
        amlignore_text = amlignore_path.read_text()
    assert "1st line\n2nd line" == amlignore_text
    assert not amlignore_path.exists()

    # If there is no .amlignore file before the test, and there are no lines to append, then there should be no
    # .amlignore file during the test
    amlignore_path = tmp_path / Path(uuid4().hex)
    with himl.append_to_amlignore(
            amlignore=amlignore_path,
            lines_to_append=[]):
        amlignore_exists_during_test = amlignore_path.exists()
    assert not amlignore_exists_during_test
    assert not amlignore_path.exists()

    # If there is an empty .amlignore file before the test, it should be there afterwards
    amlignore_path = tmp_path / Path(uuid4().hex)
    amlignore_path.touch()
    with himl.append_to_amlignore(
            amlignore=amlignore_path,
            lines_to_append=["1st line", "2nd line"]):
        amlignore_text = amlignore_path.read_text()
    assert "1st line\n2nd line" == amlignore_text
    assert amlignore_path.exists()
    assert amlignore_path.read_text() == ""

    # If there is a .amlignore file before the test, it should be identical afterwards
    amlignore_path = tmp_path / Path(uuid4().hex)
    amlignore_path.write_text("0th line")
    with himl.append_to_amlignore(
            amlignore=amlignore_path,
            lines_to_append=["1st line", "2nd line"]):
        amlignore_text = amlignore_path.read_text()
    assert "0th line\n1st line\n2nd line" == amlignore_text
    amlignore_text = amlignore_path.read_text()
    assert "0th line" == amlignore_text


# endregion Small fast local unit tests


# region Elevate to AzureML unit tests

here = pathlib.Path(__file__).parent.resolve()


def spawn_and_monitor_subprocess(process: str, args: List[str],
                                 cwd: Path, env: Dict[str, str]) -> Tuple[int, List[str]]:
    """
    Helper function to spawn and monitor subprocesses.
    :param process: The name or path of the process to spawn.
    :param args: The args to the process.
    :param cwd: Working directory.
    :param env: The environment variables for the process (default is the environment variables of the parent).
    :return: Return code after the process has finished, and the list of lines that were written to stdout by the
    subprocess.
    """
    p = subprocess.Popen(
        [process] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=cwd,
        env=env)

    # Read and print all the lines that are printed by the subprocess
    stdout_lines = [line.decode('UTF-8').strip() for line in p.stdout]  # type: ignore

    logging.info("~~~~~~~~~~~~~~")
    logging.info("\n".join(stdout_lines))
    logging.info("~~~~~~~~~~~~~~")

    return p.wait(), stdout_lines


def render_test_scripts(path: Path, local: bool,
                        extra_options: Dict[str, str], extra_args: List[str]) -> Tuple[int, List[str]]:
    """
    Prepare test scripts, submit them, and return response.

    :param path: Where to build the test scripts.
    :param local: Local execution if True, else in AzureML.
    :param extra_options: Extra options for template rendering.
    :param extra_args: Extra command line arguments for calling script.
    :return: snapshot_root and response from spawn_and_monitor_subprocess.
    """
    repo_root = repository_root()

    environment_yaml_path = path / "environment.yml"
    latest_version_path = repo_root / "latest_version.txt"
    if latest_version_path.exists():
        latest_version = f"=={latest_version_path.read_text()}"
        logging.debug(f"pinning hi-ml to: {latest_version}")
    else:
        latest_version = ""
        logging.debug("not pinning hi-ml")
    render_environment_yaml(environment_yaml_path, latest_version)

    entry_script_path = path / "test_script.py"
    render_test_script(entry_script_path, extra_options, INEXPENSIVE_TESTING_CLUSTER_NAME, environment_yaml_path)

    score_args = [str(entry_script_path)]
    if not local:
        score_args.append("--azureml")
    score_args.extend(extra_args)

    env = dict(os.environ.items())

    with check_config_json(path):
        return spawn_and_monitor_subprocess(
            process=sys.executable,
            args=score_args,
            cwd=path,
            env=env)


@pytest.mark.parametrize("local", [True, False])
def test_invoking_hello_world(local: bool, tmp_path: Path) -> None:
    """
    Test invoking hello_world.py.and
    If running in AzureML - does not elevate itself to AzureML without any config.
    Else runs locally.
    :param local: Local execution if True, else in AzureML.
    :param tmp_path: PyTest test fixture for temporary path.
    """
    extra_options = {
        'workspace_config_path': 'None',
        'environment_variables': 'None'
    }
    extra_args = ["--message=hello_world"]
    code, stdout = render_test_scripts(tmp_path, local, extra_options, extra_args)
    captured = "\n".join(stdout)
    if local:
        assert code == 0
        assert "Successfully queued new run" not in captured
        assert 'The message was: hello_world' in captured
    else:
        assert code == 1
        assert "Cannot glean workspace config from parameters, and so not submitting to AzureML" in captured


@pytest.mark.parametrize("local", [True, False])
def test_invoking_hello_world_config(local: bool, tmp_path: Path) -> None:
    """
    Test that invoking hello_world.py elevates itself to AzureML with config.json.
    :param local: Local execution if True, else in AzureML.
    :param tmp_path: PyTest test fixture for temporary path.
    """
    extra_options = {
        'environment_variables': 'None'
    }
    extra_args = ["--message=hello_world"]
    code, stdout = render_test_scripts(tmp_path, local, extra_options, extra_args)
    captured = "\n".join(stdout)
    assert code == 0
    queuing_message = "Successfully queued run"
    execution_message = 'The message was: hello_world'
    if local:
        assert queuing_message not in captured
        assert execution_message in captured
    else:
        assert queuing_message in captured
        run = get_most_recent_run(run_recovery_file=tmp_path / himl.RUN_RECOVERY_FILE)
        assert run.status in ["Finalizing", "Completed"]
        log_root = tmp_path / "logs"
        log_root.mkdir(exist_ok=False)
        run.get_all_logs(destination=str(log_root))
        driver_log = log_root / "azureml-logs" / "70_driver_log.txt"
        log_text = driver_log.read_text()
        assert execution_message in log_text


@patch("health.azure.himl.submit_to_azure_if_needed")
def test_calling_script_directly(mock_submit_to_azure_if_needed: mock.MagicMock) -> None:
    with mock.patch("sys.argv", ["",
                                 "--workspace_config_path", "1",
                                 "--compute_cluster_name", "2",
                                 "--snapshot_root_directory", "3",
                                 "--entry_script", "4",
                                 "--conda_environment_file", "5"]):
        himl.main()
    assert mock_submit_to_azure_if_needed.call_args[1]["workspace_config_path"] == PosixPath("1")
    assert mock_submit_to_azure_if_needed.call_args[1]["compute_cluster_name"] == "2"
    assert mock_submit_to_azure_if_needed.call_args[1]["snapshot_root_directory"] == PosixPath("3")
    assert mock_submit_to_azure_if_needed.call_args[1]["entry_script"] == PosixPath("4")
    assert mock_submit_to_azure_if_needed.call_args[1]["conda_environment_file"] == PosixPath("5")

# endregion Elevate to AzureML unit tests
