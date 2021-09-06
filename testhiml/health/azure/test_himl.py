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
import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PosixPath
from typing import Dict, List, Tuple
from unittest import mock
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from _pytest.capture import CaptureFixture
from azureml._restclient.constants import RunStatus
from azureml.core import RunConfiguration, Workspace
from azureml.data.azure_storage_datastore import AzureBlobDatastore
from azureml.data.dataset_consumption_config import DatasetConsumptionConfig

import health.azure.himl as himl
from conftest import check_config_json
from health.azure.azure_util import EXPERIMENT_RUN_SEPARATOR, get_most_recent_run
from health.azure.datasets import DatasetConfig, _input_dataset_key, _output_dataset_key, get_datastore
from testhiml.health.azure.test_data.make_tests import render_environment_yaml, render_test_script
from testhiml.health.azure.util import DEFAULT_DATASTORE

INEXPENSIVE_TESTING_CLUSTER_NAME = "lite-testing-ds2"
EXPECTED_QUEUED = "This command will be run in AzureML:"
GITHUB_SHIBBOLETH = "GITHUB_RUN_ID"  # https://docs.github.com/en/actions/reference/environment-variables

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
                snapshot_root_directory=Path(__file__).parent)
        # N.B. This assert may fail when run locally since we may find a workspace_config_path through the call to
        # _find_file(CONDA_ENVIRONMENT_FILE) in submit_to_azure_if_needed
        if _is_running_in_github_pipeline():
            assert "No workspace config file given, nor can we find one" in str(ex)
    with mock.patch("sys.argv", [""]):
        result = himl.submit_to_azure_if_needed(
            entry_script=Path(__file__),
            compute_cluster_name="foo",
            conda_environment_file=Path("env.yml"))
        assert isinstance(result, himl.AzureRunInfo)
        assert not result.is_running_in_azure
        assert result.run is None


def _is_running_in_github_pipeline() -> bool:
    """
    :return: Is the test running in a pipeline/action on GitHub, i.e. not locally?
    """
    return GITHUB_SHIBBOLETH in os.environ


@pytest.mark.fast
@patch("health.azure.himl.Run")
def test_write_run_recovery_file(mock_run: mock.MagicMock) -> None:
    # recovery file does not exist:
    mock_run.id = uuid4().hex
    mock_run.experiment.name = uuid4().hex
    expected_run_recovery_id = mock_run.experiment.name + EXPERIMENT_RUN_SEPARATOR + mock_run.id
    recovery_file = Path(himl.RUN_RECOVERY_FILE)
    if recovery_file.exists():
        recovery_file.unlink()
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
    )
    assert run_config.max_run_duration_seconds is None
    assert run_config.mpi.node_count == 1
    assert not run_config.data
    assert not run_config.output_data


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
    workspace = himl.get_workspace(mock_workspace, None)
    assert workspace == mock_workspace
    mock_get_authentication.return_value = None
    _ = himl.get_workspace(None, Path(__file__))
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


@pytest.mark.fast
@pytest.mark.parametrize("wait_for_completion", [True, False])
@patch("health.azure.himl.Run")
@patch("health.azure.himl.ScriptRunConfig")
@patch("health.azure.himl.Experiment")
@patch("health.azure.himl.Workspace")
def test_submit_run(
        mock_workspace: mock.MagicMock,
        mock_experiment: mock.MagicMock,
        mock_script_run_config: mock.MagicMock,
        mock_run: mock.MagicMock,
        wait_for_completion: bool,
        capsys: CaptureFixture,
        ) -> None:
    mock_experiment.return_value.submit.return_value = mock_run
    mock_run.get_status.return_value = RunStatus.COMPLETED
    mock_run.status = RunStatus.COMPLETED
    mock_run.get_children.return_value = []
    an_experiment_name = "an experiment"
    _ = himl.submit_run(
        workspace=mock_workspace,
        experiment_name=an_experiment_name,
        script_run_config=mock_script_run_config,
        wait_for_completion=wait_for_completion,
        wait_for_completion_show_output=True,
    )
    out, err = capsys.readouterr()
    assert not err
    assert "Successfully queued run" in out
    assert "Experiment name and run ID are available" in out
    assert "Experiment URL" in out
    assert "Run URL" in out
    if wait_for_completion:
        assert "Waiting for the completion of the AzureML run" in out
        assert "AzureML completed" in out
        mock_run.get_status.return_value = RunStatus.UNAPPROVED
        mock_run.status = RunStatus.UNAPPROVED
        with pytest.raises(ValueError) as e:
            _ = himl.submit_run(
                workspace=mock_workspace,
                experiment_name=an_experiment_name,
                script_run_config=mock_script_run_config,
                wait_for_completion=wait_for_completion,
                wait_for_completion_show_output=True,
            )
        error_msg = str(e.value)
        out, err = capsys.readouterr()
        assert "runs failed" in error_msg
        assert "AzureML completed" not in out


@pytest.mark.fast
def test_str_to_path(tmp_path: Path) -> None:
    assert himl._str_to_path(tmp_path) == tmp_path
    assert himl._str_to_path(str(tmp_path)) == tmp_path


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
    found_file = himl._find_file(file_name, False)
    assert found_file
    with mock.patch.dict(os.environ, {"PYTHONPATH": str(python_root.absolute())}):
        found_file = himl._find_file(file_name)
        assert not found_file
    os.chdir(where_are_we_now)

# endregion Small fast local unit tests


# region Elevate to AzureML unit tests

here = pathlib.Path(__file__).parent.resolve()


class RunTarget(Enum):
    LOCAL = 1
    AZUREML = 2


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


def render_and_run_test_script(path: Path,
                               run_target: RunTarget,
                               extra_options: Dict[str, str],
                               extra_args: List[str],
                               expected_pass: bool,
                               suppress_config_creation: bool = False) -> str:
    """
    Prepare test scripts, submit them, and return response.

    :param path: Where to build the test scripts.
    :param run_target: Where to run the script.
    :param extra_options: Extra options for template rendering.
    :param extra_args: Extra command line arguments for calling script.
    :param expected_pass: Whether this call to subprocess is expected to be successful.
    :param suppress_config_creation: (Optional, defaults to False) do not create a config.json file if none exists
    :return: Either response from spawn_and_monitor_subprocess or run output if in AzureML.
    """
    # target hi-ml package version, if specified in an environment variable.
    version = ""
    run_requirements = False

    himl_wheel_filename = os.getenv('HIML_WHEEL_FILENAME', '')
    himl_test_pypi_version = os.getenv('HIML_TEST_PYPI_VERSION', '')
    himl_pypi_version = os.getenv('HIML_PYPI_VERSION', '')

    if not himl_wheel_filename:
        # If testing locally, can build the package into the "dist" folder and use that.
        dist_folder = Path.cwd().joinpath('dist')
        whls = sorted(list(dist_folder.glob('*.whl')))
        if len(whls) > 0:
            last_whl = whls[-1]
            himl_wheel_filename = str(last_whl)

    if himl_wheel_filename:
        # Testing against a private wheel.
        himl_wheel_filename_full_path = str(Path(himl_wheel_filename).resolve())
        extra_options['private_pip_wheel_path'] = f'Path("{himl_wheel_filename_full_path}")'
        print(f"Added private_pip_wheel_path: {himl_wheel_filename_full_path} option")
    elif himl_test_pypi_version:
        # Testing against test.pypi, add this as the pip_extra_index_url, and set the version.
        extra_options['pip_extra_index_url'] = "https://test.pypi.org/simple/"
        version = himl_test_pypi_version
        print(f"Added test.pypi: {himl_test_pypi_version} option")
    elif himl_pypi_version:
        # Testing against pypi, set the version.
        version = himl_pypi_version
        print(f"Added pypi: {himl_pypi_version} option")
    else:
        # No packages found, so copy the src folder as a fallback
        src_path = Path.cwd().joinpath('src')
        if src_path.is_dir():
            shutil.copytree(src=src_path / 'health', dst=path / 'health')
            run_requirements = True
            print("Copied 'src' folder.")

    environment_yaml_path = path / "environment.yml"
    render_environment_yaml(environment_yaml_path, version, run_requirements)

    entry_script_path = path / "test_script.py"
    render_test_script(entry_script_path, extra_options, INEXPENSIVE_TESTING_CLUSTER_NAME, environment_yaml_path)

    score_args = [str(entry_script_path)]
    if run_target == RunTarget.AZUREML:
        score_args.append("--azureml")
    score_args.extend(extra_args)

    env = dict(os.environ.items())

    def spawn() -> Tuple[int, List[str], Workspace]:
        code, stdout = spawn_and_monitor_subprocess(
            process=sys.executable,
            args=score_args,
            cwd=path,
            env=env)
        workspace = himl.get_workspace(aml_workspace=None, workspace_config_path=path / himl.WORKSPACE_CONFIG_JSON)
        return code, stdout, workspace

    if suppress_config_creation:
        code, stdout, workspace = spawn()
    else:
        with check_config_json(path):
            code, stdout, workspace = spawn()
    assert code == 0 if expected_pass else 1
    captured = "\n".join(stdout)

    if run_target == RunTarget.LOCAL or not expected_pass:
        assert EXPECTED_QUEUED not in captured
        return captured
    else:
        assert EXPECTED_QUEUED in captured
        run = get_most_recent_run(run_recovery_file=path / himl.RUN_RECOVERY_FILE,
                                  workspace=workspace)
        assert run.status == "Completed"
        log_root = path / "logs"
        log_root.mkdir(exist_ok=False)
        run.get_all_logs(destination=log_root)
        driver_log = log_root / "azureml-logs" / "70_driver_log.txt"
        log_text = driver_log.read_text()
        return log_text


@pytest.mark.parametrize("run_target", [RunTarget.LOCAL, RunTarget.AZUREML])
def test_invoking_hello_world_no_config(run_target: RunTarget, tmp_path: Path) -> None:
    """
    Test invoking rendered 'simple' / 'hello_world_template.txt'.and
    If running in AzureML - does not elevate itself to AzureML without any config.
    Else runs locally.
    :param run_target: Where to run the script.
    :param tmp_path: PyTest test fixture for temporary path.
    """
    message_guid = uuid4().hex
    extra_options = {
        'workspace_config_path': 'None',
        'args': 'parser.add_argument("-m", "--message", type=str, required=True, help="The message to print out")',
        'body': 'print(f"The message was: {args.message}")'
    }
    extra_args = [f"--message={message_guid}"]
    expected_output = f"The message was: {message_guid}"
    if run_target == RunTarget.LOCAL:
        output = render_and_run_test_script(tmp_path, run_target, extra_options, extra_args,
                                            run_target == RunTarget.LOCAL,
                                            suppress_config_creation=run_target == RunTarget.AZUREML)
        assert expected_output in output
    else:
        with pytest.raises(ValueError) as e:
            render_and_run_test_script(tmp_path, run_target, extra_options, extra_args, run_target == RunTarget.LOCAL,
                                       suppress_config_creation=run_target == RunTarget.AZUREML)
        assert "Cannot glean workspace config from parameters, and so not submitting to AzureML" in str(e.value)


@pytest.mark.parametrize("run_target", [RunTarget.LOCAL, RunTarget.AZUREML])
@pytest.mark.parametrize("use_package", [True, False])
def test_invoking_hello_world_config(run_target: RunTarget, use_package: bool, tmp_path: Path) -> None:
    """
    Test that invoking hello_world.py elevates itself to AzureML with config.json.
    Test against either the local src folder or a package. If running locally, ensure that there
    are no whl's in the dist folder, or that will be used.
    :param local: Local execution if True, else in AzureML.
    :param use_package: True to test against package, False to test against copy of src folder.
    :param tmp_path: PyTest test fixture for temporary path.
    """
    if not use_package and \
            not os.getenv('HIML_WHEEL_FILENAME', '') and \
            not os.getenv('HIML_TEST_PYPI_VERSION', '') and \
            not os.getenv('HIML_PYPI_VERSION', ''):
        # Running locally, no need to duplicate this test.
        return

    message_guid = uuid4().hex
    extra_options = {
        'args': 'parser.add_argument("-m", "--message", type=str, required=True, help="The message to print out")',
        'body': 'print(f"The message was: {args.message}")'
    }
    extra_args = [f"--message={message_guid}"]
    if use_package:
        output = render_and_run_test_script(tmp_path, run_target, extra_options, extra_args, True)
    else:
        with mock.patch.dict(os.environ, {"HIML_WHEEL_FILENAME": '',
                                          "HIML_TEST_PYPI_VERSION": '',
                                          "HIML_PYPI_VERSION": ''}):
            output = render_and_run_test_script(tmp_path, run_target, extra_options, extra_args, True)
    expected_output = f"The message was: {message_guid}"
    assert expected_output in output


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


def test_invoking_hello_world_no_private_pip_fails(tmp_path: Path) -> None:
    """
    Test that invoking rendered 'simple' / 'hello_world_template.txt' raises a FileNotFoundError on
    invalid private_pip_wheel_path.
    :param tmp_path: PyTest test fixture for temporary path.
    """
    extra_options: Dict[str, str] = {}
    extra_args: List[str] = []
    with mock.patch.dict(os.environ, {"HIML_WHEEL_FILENAME": 'not_a_known_file.whl'}):
        output = render_and_run_test_script(tmp_path, RunTarget.AZUREML, extra_options, extra_args, False)
    error_message_begin = "FileNotFoundError: Cannot add add_private_pip_wheel:"
    error_message_end = "not_a_known_file.whl, it is not a file."

    assert error_message_begin in output
    assert error_message_end in output


@pytest.mark.parametrize("run_target", [RunTarget.LOCAL, RunTarget.AZUREML])
def test_invoking_hello_world_env_var(run_target: RunTarget, tmp_path: Path) -> None:
    """
    Test that invoking rendered 'simple' / 'hello_world_template.txt' elevates itself to AzureML with config.json,
    and that environment variables are passed through.
    :param run_target: Where to run the script.
    :param tmp_path: PyTest test fixture for temporary path.
    """
    message_guid = uuid4().hex
    extra_options: Dict[str, str] = {
        'environment_variables': f"{{'message_guid': '{message_guid}'}}",
        'body': 'print(f"The message_guid env var was: {os.getenv(\'message_guid\')}")'
    }
    extra_args: List[str] = []
    output = render_and_run_test_script(tmp_path, run_target, extra_options, extra_args, True)
    expected_output = f"The message_guid env var was: {message_guid}"
    assert expected_output in output


def _create_test_file_in_blobstore(datastore: AzureBlobDatastore,
                                   filename: str, location: str, tmp_path: Path) -> str:
    # Create a dummy folder.
    dummy_data_folder = tmp_path / "dummy_data"
    dummy_data_folder.mkdir()

    # Create a dummy text file.
    dummy_txt_file = dummy_data_folder / filename
    message_guid = uuid4().hex
    dummy_txt_file.write_text(f"some test data: {message_guid}")

    # Upload dummy text file to blob storage
    datastore.upload_files(
        [str(dummy_txt_file.resolve())],
        relative_root=str(dummy_data_folder),
        target_path=location,
        overwrite=True,
        show_progress=True)

    dummy_txt_file_contents = dummy_txt_file.read_text()

    # Discard dummies
    dummy_txt_file.unlink()
    dummy_data_folder.rmdir()

    return dummy_txt_file_contents


@dataclass
class TestInputDataset:
    # Test file name. This will be populated with test data and uploaded to blob storage.
    filename: str
    # Name of container for this dataset in blob storage.
    blob_name: str
    # Local folder for this dataset when running locally.
    folder_name: Path
    # Contents of test file.
    contents: str = ""


@dataclass
class TestOutputDataset:
    # Name of container for this dataset in blob storage.
    blob_name: str
    # Local folder for this dataset when running locally or when testing after running in Azure.
    folder_name: Path


@pytest.mark.parametrize("run_target", [RunTarget.LOCAL, RunTarget.AZUREML])
def test_invoking_hello_world_datasets(run_target: RunTarget, tmp_path: Path) -> None:
    """
    Test that invoking rendered 'simple' / 'hello_world_template.txt' elevates itself to AzureML with config.json,
    and that datasets are mounted in all combinations.
    :param run_target: Where to run the script.
    :param tmp_path: PyTest test fixture for temporary path.
    """
    input_count = 4
    input_datasets = [TestInputDataset(
                          filename=f"{uuid4().hex}.txt",
                          blob_name=f"himl_dataset_test_input{i}",
                          folder_name=tmp_path / f"local_dataset_test_input{i}")
                      for i in range(0, input_count)]
    output_count = 3
    output_datasets = [TestOutputDataset(
                           blob_name=f"himl_dataset_test_output{i}",
                           folder_name=tmp_path / f"local_dataset_test_output{i}")
                       for i in range(0, output_count)]

    # Get default datastore
    with check_config_json(tmp_path):
        workspace = himl.get_workspace(aml_workspace=None,
                                       workspace_config_path=tmp_path / himl.WORKSPACE_CONFIG_JSON)
        datastore: AzureBlobDatastore = get_datastore(workspace=workspace,
                                                      datastore_name=DEFAULT_DATASTORE)

    # Create dummy txt files, one for each item in input_datasets.
    for input_dataset in input_datasets:
        input_dataset.contents = _create_test_file_in_blobstore(
            datastore=datastore,
            filename=input_dataset.filename,
            location=input_dataset.blob_name,
            tmp_path=tmp_path)

        if run_target == RunTarget.LOCAL:
            # For running locally, download the test files from blobstore
            downloaded = datastore.download(
                target_path=input_dataset.folder_name,
                prefix=f"{input_dataset.blob_name}/{input_dataset.filename}",
                overwrite=True,
                show_progress=True)
            assert downloaded == 1

            # Check that the input file is downloaded
            downloaded_dummy_txt_file = input_dataset.folder_name / input_dataset.blob_name / input_dataset.filename
            # Check it has expected contents
            assert input_dataset.contents == downloaded_dummy_txt_file.read_text()

    if run_target == RunTarget.LOCAL:
        for output_dataset in output_datasets:
            output_blob_folder = output_dataset.folder_name / output_dataset.blob_name
            output_blob_folder.mkdir(parents=True)
    else:
        # Check that these files are not already in the output folders.
        for input_dataset in input_datasets:
            for output_dataset in output_datasets:
                downloaded = datastore.download(
                    target_path=str(output_dataset.folder_name),
                    prefix=f"{output_dataset.blob_name}/{input_dataset.filename}",
                    overwrite=True,
                    show_progress=True)
                assert downloaded == 0

    # Format input_datasets for use in script.
    input_file_names = [
        f'("{input_dataset.filename}", "{input_dataset.blob_name}", Path("{str(input_dataset.folder_name)}"))'
        for input_dataset in input_datasets]
    script_input_datasets = ',\n        '.join(input_file_names)

    # Format output_datasets for use in script.
    output_file_names = [
        f'("{output_dataset.blob_name}", Path("{str(output_dataset.folder_name)}"))'
        for output_dataset in output_datasets]
    script_output_datasets = ',\n        '.join(output_file_names)

    directory_print = "print(f\"{str(input_folder_name)} contains these files: {input_folder_name.glob('*.txt')}\")"

    extra_options: Dict[str, str] = {
        'prequel': """
    target_folder = "foo"
        """,
        'ignored_folders': '[".config", ".mypy_cache", "hello_world_output"]',
        'default_datastore': f'"{DEFAULT_DATASTORE}"',
        'input_datasets': f"""[
            "{input_datasets[0].blob_name}",
            DatasetConfig(name="{input_datasets[1].blob_name}", datastore="{DEFAULT_DATASTORE}"),
            DatasetConfig(name="{input_datasets[2].blob_name}", datastore="{DEFAULT_DATASTORE}",
                          target_folder=target_folder),
            DatasetConfig(name="{input_datasets[3].blob_name}", datastore="{DEFAULT_DATASTORE}",
                          use_mounting=True),
        ]""",
        'output_datasets': f"""[
            "{output_datasets[0].blob_name}",
            DatasetConfig(name="{output_datasets[1].blob_name}", datastore="{DEFAULT_DATASTORE}"),
            DatasetConfig(name="{output_datasets[2].blob_name}", datastore="{DEFAULT_DATASTORE}",
                          use_mounting=False),
        ]""",
        'body': f"""
    input_datasets = [
        {script_input_datasets}
    ]
    output_datasets = [
        {script_output_datasets}
    ]
    for i, (filename, input_blob_name, input_folder_name) in enumerate(input_datasets):
        if input_folder_name.exists():
            {directory_print}
        input_folder = run_info.input_datasets[i] or input_folder_name / input_blob_name
        for j, (output_blob_name, output_folder_name) in enumerate(output_datasets):
            output_folder = run_info.output_datasets[j] or output_folder_name / output_blob_name
            file = input_folder / filename
            shutil.copy(file, output_folder)
            print(f"Copied file: {{file.name}} from {{input_blob_name}} to {{output_blob_name}}")
        """
    }
    extra_args: List[str] = []
    output = render_and_run_test_script(tmp_path, run_target, extra_options, extra_args, True)

    for input_dataset in input_datasets:
        for output_dataset in output_datasets:
            expected_output = \
                f"Copied file: {input_dataset.filename} from {input_dataset.blob_name} to {output_dataset.blob_name}"
            assert expected_output in output

            if run_target == RunTarget.AZUREML:
                # If test ran in Azure, need to download the outputs to check them.
                downloaded = datastore.download(
                    target_path=str(output_dataset.folder_name),
                    prefix=f"{output_dataset.blob_name}/{input_dataset.filename}",
                    overwrite=True,
                    show_progress=True)
                assert downloaded == 1

            output_dummy_txt_file = output_dataset.folder_name / output_dataset.blob_name / input_dataset.filename
            assert input_dataset.contents == output_dummy_txt_file.read_text()


# endregion Elevate to AzureML unit tests
