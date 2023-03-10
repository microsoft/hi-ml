#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
Tests for hi-ml-azure.
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
from random import randint
from ruamel.yaml import YAML
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock
from unittest.mock import MagicMock, create_autospec, patch, DEFAULT
from uuid import uuid4

import pytest
from _pytest.capture import CaptureFixture
from azure.ai.ml import Input, Output, MLClient
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.entities import Data, Job
from azure.ai.ml.entities._job.distribution import MpiDistribution, PyTorchDistribution
from azure.ai.ml.sweep import Choice
from azure.core.exceptions import ResourceNotFoundError
from azureml._restclient.constants import RunStatus
from azureml.core import ComputeTarget, Environment, RunConfiguration, ScriptRunConfig, Workspace
from azureml.data.azure_storage_datastore import AzureBlobDatastore
from azureml.data.dataset_consumption_config import DatasetConsumptionConfig
from azureml.dataprep.fuse.daemon import MountContext
from azureml.train.hyperdrive import HyperDriveConfig

import health_azure.himl as himl
from health_azure.datasets import (
    DatasetConfig,
    _input_dataset_key,
    _output_dataset_key,
    get_datastore,
    setup_local_datasets
)
from health_azure.utils import (
    DEFAULT_ENVIRONMENT_VARIABLES,
    ENV_EXPERIMENT_NAME,
    ENV_WORKSPACE_NAME,
    ENVIRONMENT_VERSION,
    EXPERIMENT_RUN_SEPARATOR,
    WORKSPACE_CONFIG_JSON,
    VALID_LOG_FILE_PATHS,
    JobStatus,
    check_config_json,
    get_most_recent_run,
    get_workspace,
    is_running_in_azure_ml,
    get_driver_log_file_text,
    get_ml_client,
)
from testazure.test_data.make_tests import render_environment_yaml, render_test_script
from testazure.utils_testazure import (
    DEFAULT_DATASTORE,
    USER_IDENTITY_TEST_DATASTORE,
    USER_IDENTITY_TEST_ASSET,
    USER_IDENTITY_TEST_ASSET_OUTPUT,
    USER_IDENTITY_TEST_FILE,
    TEST_DATA_ASSET_NAME,
    TEST_DATASTORE_NAME,
    change_working_directory,
    current_test_name,
    get_shared_config_json,
    repository_root
)

INEXPENSIVE_TESTING_CLUSTER_NAME = "lite-testing-ds2"
EXPECTED_QUEUED = "This command will be run in AzureML:"
GITHUB_SHIBBOLETH = "GITHUB_RUN_ID"  # https://docs.github.com/en/actions/reference/environment-variables
AZUREML_FLAG = himl.AZUREML_FLAG

TEST_ML_CLIENT = get_ml_client()

logger = logging.getLogger('test.health_azure')
logger.setLevel(logging.DEBUG)


# region Small fast local unit tests

@pytest.mark.fast
def test_submit_to_azure_if_needed_returns_immediately() -> None:
    """
    Test that himl.submit_to_azure_if_needed can be called, and returns immediately if not submitting anything
    to AzureML.
    """
    result = himl.submit_to_azure_if_needed(
        entry_script=Path(__file__),
        compute_cluster_name="foo")
    assert isinstance(result, himl.AzureRunInfo)
    assert not result.is_running_in_azure_ml
    assert result.run is None


def _is_running_in_github_pipeline() -> bool:
    """
    :return: Is the test running in a pipeline/action on GitHub, i.e. not locally?
    """
    return GITHUB_SHIBBOLETH in os.environ


@pytest.mark.fast
@patch("health_azure.himl.Run")
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


@pytest.fixture(scope="module")
def dummy_max_num_nodes_available() -> int:
    """
    Return a random integer between 2 and 10 that will represent the maximum number of
    nodes in our mock compute cluster
    """
    return randint(2, 10)


@pytest.fixture(scope="module")
def dummy_compute_cluster_name() -> str:
    """
    Returns a name for our mock Compute Target that will be used in multiple tests
    """
    return 'dummy_cluster'


@pytest.fixture(scope="module")
def mock_scale_settings(dummy_max_num_nodes_available: int) -> MagicMock:
    """
    Mock an Azure ML ScaleSettings object containing just the number of nodes a cluster
    can resize to
    """
    return MagicMock(maximum_node_count=dummy_max_num_nodes_available)


@pytest.fixture(scope="module")
def mock_compute_cluster(mock_scale_settings: MagicMock) -> MagicMock:
    """
    Mock an Azure ML ComputeTarget representing a compute cluster with property 'scale_settings'
    defined by our mock ScaleSettings object
    """
    return MagicMock(scale_settings=mock_scale_settings)


@pytest.fixture(scope="module")
def mock_workspace(mock_compute_cluster: MagicMock, dummy_compute_cluster_name: str) -> MagicMock:
    """
    Mock an Azure ML Workspace whose property compute_targets contains just our mock ComputeTarget object
    """
    return MagicMock(compute_targets={dummy_compute_cluster_name: mock_compute_cluster})


@pytest.mark.fast
def test_validate_num_nodes(dummy_max_num_nodes_available: int, mock_compute_cluster: MagicMock,
                            dummy_compute_cluster_name: str) -> None:
    # If number of requested nodes <= max available nodes, nothing should happen
    num_nodes_requested = dummy_max_num_nodes_available // 2
    himl.validate_num_nodes(mock_compute_cluster, num_nodes_requested)
    num_nodes_requested = dummy_max_num_nodes_available
    himl.validate_num_nodes(mock_compute_cluster, num_nodes_requested)

    # But if number of nodes requested > max available, a ValueError should be raised
    num_nodes_requested = randint(dummy_max_num_nodes_available + 1, 5000)
    expected_error_msg = f"You have requested {num_nodes_requested} nodes, which is more than your compute "
    f"cluster {dummy_compute_cluster_name}'s maximum of {dummy_max_num_nodes_available} nodes "
    with pytest.raises(ValueError, match=expected_error_msg):
        himl.validate_num_nodes(mock_compute_cluster, num_nodes_requested)


@pytest.mark.fast
def test_validate_compute_name(mock_workspace: MagicMock, dummy_compute_cluster_name: str) -> None:
    existing_compute_clusters = mock_workspace.compute_targets
    existent_compute_name = dummy_compute_cluster_name
    himl.validate_compute_name(existing_compute_clusters, existent_compute_name)

    nonexistent_compute_name = 'idontexist'
    assert nonexistent_compute_name not in existing_compute_clusters
    expected_error_msg = f"Could not find the compute target {nonexistent_compute_name} in the AzureML workspace."
    with pytest.raises(ValueError, match=expected_error_msg):
        himl.validate_compute_name(existing_compute_clusters, nonexistent_compute_name)


@pytest.mark.fast
@patch("health_azure.himl.validate_compute_name")
@patch("health_azure.himl.validate_num_nodes")
def test_validate_compute(mock_validate_num_nodes: MagicMock, mock_validate_compute_name: MagicMock,
                          mock_workspace: MagicMock, dummy_compute_cluster_name: str) -> None:
    def _raise_value_error(*args: Any) -> None:
        raise ValueError("A ValueError has been raised")

    def _raise_assertion_error(*args: Any) -> None:
        raise AssertionError("An AssertionError has been raised")

    # first mock the case where validate_num_nodes and validate_compute_name both return None
    mock_validate_compute_name.return_value = None
    mock_validate_num_nodes.return_value = None
    mock_num_available_nodes = 0
    himl.validate_compute_cluster(mock_workspace, dummy_compute_cluster_name, mock_num_available_nodes)
    assert mock_validate_num_nodes.call_count == 1
    assert mock_validate_compute_name.call_count == 1

    # now have validate_num_nodes raise an Assertionerror and check that calling validate_compute_cluster
    # raises the error
    mock_validate_num_nodes.side_effect = _raise_assertion_error
    with pytest.raises(AssertionError, match="An AssertionError has been raised"):
        himl.validate_compute_cluster(mock_workspace, dummy_compute_cluster_name, mock_num_available_nodes)
    assert mock_validate_num_nodes.call_count == 2
    assert mock_validate_compute_name.call_count == 2

    # now have validate_compute_name raise a ValueError and check that calling validate_compute_cluster raises the error
    mock_validate_compute_name.side_effect = _raise_value_error
    with pytest.raises(ValueError, match="A ValueError has been raised"):
        himl.validate_compute_cluster(mock_workspace, dummy_compute_cluster_name, mock_num_available_nodes)
    assert mock_validate_num_nodes.call_count == 2
    assert mock_validate_compute_name.call_count == 3


def test_validate_compute_real(tmp_path: Path) -> None:
    """
    Get a real Workspace object and attempt to validate a compute cluster from it, if any exist.
    This checks that the properties/ methods in the Azure ML SDK are consistent with those used in our
    codebase
    """
    with check_config_json(tmp_path, shared_config_json=get_shared_config_json()):
        workspace = get_workspace(aml_workspace=None,
                                  workspace_config_path=tmp_path / WORKSPACE_CONFIG_JSON)
    existing_compute_targets: Dict[str, ComputeTarget] = workspace.compute_targets
    if len(existing_compute_targets) == 0:
        return
    compute_target_name: str = list(existing_compute_targets)[0]
    compute_target: ComputeTarget = existing_compute_targets[compute_target_name]
    max_num_nodes = compute_target.scale_settings.maximum_node_count
    request_num_nodes = max_num_nodes - 1
    assert isinstance(compute_target_name, str)
    assert isinstance(compute_target, ComputeTarget)
    himl.validate_compute_cluster(workspace, compute_target_name, request_num_nodes)


@pytest.mark.fast
@patch("azureml.data.OutputFileDatasetConfig")
@patch("health_azure.himl.Workspace")
@patch("health_azure.himl.DatasetConfig")
def test_to_datasets(
        mock_dataset_config: mock.MagicMock,
        mock_workspace: mock.MagicMock,
        mock_output_file_dataset_config: mock.MagicMock) -> None:
    def to_input_dataset(workspace: Workspace, dataset_index: int, strictly_aml_v1: bool,
                         ml_client: Optional[MLClient] = None) -> DatasetConsumptionConfig:
        return mock_dataset_consumption_config

    def to_output_dataset(workspace: Workspace, dataset_index: int) -> DatasetConsumptionConfig:
        return mock_output_file_dataset_config

    mock_dataset_consumption_config = mock.create_autospec(DatasetConsumptionConfig)
    mock_dataset_consumption_config.__class__.return_value = DatasetConsumptionConfig
    mock_dataset_consumption_config.name = "A Consumption Config"
    mock_output_file_dataset_config.name = "An Output File Dataset Config"
    mock_dataset_config.to_input_dataset = to_input_dataset
    mock_dataset_config.to_output_dataset = to_output_dataset
    with pytest.raises(ValueError) as ex1:
        himl.convert_himl_to_azureml_datasets(
            cleaned_input_datasets=[mock_dataset_config, mock_dataset_config],
            cleaned_output_datasets=[],
            strictly_aml_v1=True,
            workspace=mock_workspace)
        assert "already an input dataset with name" in str(ex1)
    with pytest.raises(ValueError) as ex2:
        himl.convert_himl_to_azureml_datasets(
            cleaned_input_datasets=[mock_dataset_config, mock_dataset_config],
            cleaned_output_datasets=[],
            strictly_aml_v1=True,
            workspace=mock_workspace)
        assert "already an output dataset with name" in str(ex2)

    cleaned_input_datasets: List[DatasetConfig] = [mock_dataset_config]
    cleaned_output_datasets: List[DatasetConfig] = [mock_dataset_config]
    inputs, outputs = himl.convert_himl_to_azureml_datasets(
        cleaned_input_datasets=cleaned_input_datasets,
        cleaned_output_datasets=cleaned_output_datasets,
        strictly_aml_v1=True,
        workspace=mock_workspace)
    assert len(inputs) == 1
    assert len(outputs) == 1
    assert inputs[mock_dataset_consumption_config.name] == mock_dataset_consumption_config
    assert outputs[mock_output_file_dataset_config.name] == mock_output_file_dataset_config


@pytest.mark.fast
@patch("health_azure.himl.register_environment")
@patch("health_azure.himl.create_python_environment")
@patch("health_azure.himl.Workspace")
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
@patch("health_azure.himl.DockerConfiguration")
@patch("health_azure.datasets.DatasetConfig.to_output_dataset")
@patch("health_azure.datasets.DatasetConfig.to_input_dataset")
@patch("health_azure.himl.Environment.get")
@patch("health_azure.himl.Workspace")
def test_create_run_configuration(
        mock_workspace: MagicMock,
        mock_environment_get: MagicMock,
        mock_to_input_dataset: MagicMock,
        mock_to_output_dataset: MagicMock,
        mock_docker_configuration: MagicMock,
        dummy_max_num_nodes_available: MagicMock,
        mock_compute_cluster: MagicMock
) -> None:
    existing_compute_target = "this_does_exist"
    mock_env_name = "Mock Env"
    mock_environment_get.return_value = mock_env_name
    mock_workspace.compute_targets = {existing_compute_target: mock_compute_cluster}
    aml_input_dataset = create_autospec(DatasetConsumptionConfig)
    aml_input_dataset.name = "dataset_in"
    aml_output_dataset = create_autospec(DatasetConsumptionConfig)
    aml_output_dataset.name = "dataset_out"
    mock_to_input_dataset.return_value = aml_input_dataset
    mock_to_output_dataset.return_value = aml_output_dataset
    run_config = himl.create_run_configuration(
        workspace=mock_workspace,
        compute_cluster_name=existing_compute_target,
        aml_environment_name="foo",
        num_nodes=dummy_max_num_nodes_available - 1,
        max_run_duration="1h",
        input_datasets=[DatasetConfig(name="input1")],
        output_datasets=[DatasetConfig(name="output1")],
        docker_shm_size="2g",
        environment_variables={"foo": "bar"},
    )
    assert isinstance(run_config, RunConfiguration)
    assert run_config.target == existing_compute_target
    assert run_config.environment == mock_env_name
    assert run_config.node_count == dummy_max_num_nodes_available - 1
    assert run_config.mpi.node_count == dummy_max_num_nodes_available - 1
    assert run_config.max_run_duration_seconds == 60 * 60
    assert run_config.data == {"dataset_in": aml_input_dataset}
    assert run_config.output_data == {"dataset_out": aml_output_dataset}
    mock_docker_configuration.assert_called_once()
    assert run_config.environment_variables
    # Environment variables should be added to the default ones
    assert "foo" in run_config.environment_variables
    any_default_variable = "RSLEX_DIRECT_VOLUME_MOUNT"
    assert any_default_variable in DEFAULT_ENVIRONMENT_VARIABLES
    assert any_default_variable in run_config.environment_variables

    # Test run configuration default values
    run_config = himl.create_run_configuration(
        workspace=mock_workspace,
        compute_cluster_name=existing_compute_target,
        aml_environment_name="foo",
    )
    assert run_config.max_run_duration_seconds is None
    assert run_config.mpi.node_count == 1
    assert not run_config.data
    assert not run_config.output_data


def create_empty_conda_env(tmp_path: Path) -> Path:
    """Create an empty conda environment in a given folder, and returns its path."""
    conda_env_spec = dict(
        name="dummy_env",
        channels=["default"],
        dependencies=["pip=20.1.1", "python=3.7.3"]
    )
    conda_env_path = tmp_path / "dummy_conda_env.yml"
    yaml = YAML()
    yaml.dump(conda_env_spec, conda_env_path)
    assert conda_env_path.is_file()
    return conda_env_path


@pytest.mark.fast
@patch("azureml.core.Workspace")
@patch("health_azure.himl.create_python_environment")
def test_create_run_configuration_correct_env(mock_create_environment: MagicMock,
                                              mock_workspace: MagicMock,
                                              mock_compute_cluster: MagicMock,
                                              tmp_path: Path) -> None:
    mock_workspace.compute_targets = {"dummy_compute_cluster": mock_compute_cluster}

    # First ensure if environment.get returns None, that register_environment gets called
    mock_environment = MagicMock()
    # raise exception to esure that register gets called
    mock_environment.version = None

    mock_create_environment.return_value = mock_environment

    conda_env_path = create_empty_conda_env(tmp_path)
    with patch.object(mock_environment, "register") as mock_register:
        mock_register.return_value = mock_environment

        with patch("azureml.core.Environment.get") as mock_environment_get:  # type: ignore
            mock_environment_get.side_effect = Exception()
            run_config = himl.create_run_configuration(workspace=mock_workspace,
                                                       compute_cluster_name="dummy_compute_cluster",
                                                       conda_environment_file=conda_env_path)

            # check that mock_register has been called once with the expected args
            mock_register.assert_called_once_with(mock_workspace)
            assert mock_environment_get.call_count == 1

            # check that en environment is returned with the default ENVIRONMENT_VERSION (this overrides AML's
            # default 'Autosave' environment version)
            environment: Environment = run_config.environment
            assert environment.version == ENVIRONMENT_VERSION

            # when calling create_run_configuration again with the same conda environment file, Environment.get
            # should retrieve the registered version, hence we disable the side effect
            mock_environment_get.side_effect = None

            _ = himl.create_run_configuration(mock_workspace,
                                              "dummy_compute_cluster",
                                              conda_environment_file=conda_env_path)

            # check mock_register has still only been called once
            mock_register.assert_called_once()
            assert mock_environment_get.call_count == 2

    # Assert that a Conda env spec with no python version raises an exception
    conda_env_spec = dict(
        name="dummy_env",
        channels=["default"],
        dependencies=["pip=20.1.1"],
    )

    conda_env_path = tmp_path / "dummy_conda_env_no_python.yml"
    yaml = YAML()
    yaml.dump(conda_env_spec, conda_env_path)
    assert conda_env_path.is_file()

    with patch.object(mock_environment, "register") as mock_register:
        mock_register.return_value = mock_environment

        with patch("azureml.core.Environment.get") as mock_environment_get:  # type: ignore
            mock_environment_get.side_effect = Exception()

            with pytest.raises(Exception) as e:
                himl.create_run_configuration(mock_workspace,
                                              "dummy_compute_cluster",
                                              conda_environment_file=conda_env_path,
                                              )
                assert "you must specify the python version" in str(e)

    # check that when create_run_configuration is called, whatever is returned  from register_environment
    # is set as the new "environment" attribute of the run config
    with patch("health_azure.himl.register_environment") as mock_register_environment:
        for dummy_env in ["abc", 1, Environment("dummy_env")]:
            mock_register_environment.return_value = dummy_env

            with patch("azureml.core.Environment.get") as mock_environment_get:  # type: ignore
                mock_environment_get.side_effect = Exception()
                run_config = himl.create_run_configuration(mock_workspace,
                                                           "dummy_compute_cluster",
                                                           conda_environment_file=conda_env_path)
            assert run_config.environment == dummy_env

    subprocess.run


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
            script_params=[],
            entry_script=problem_entry_script,
            snapshot_root_directory=snapshot_dir,
        )
    assert "entry script must be inside of the snapshot root directory" in str(e)

    with mock.patch("sys.argv", ["foo"]):
        script_params = himl._get_script_params()
        script_run = himl.create_script_run(script_params)
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
    with mock.patch("sys.argv", ["", "a string", AZUREML_FLAG]):
        assert expected_params == himl._get_script_params()
    with mock.patch("sys.argv", ["", "a string"]):
        assert expected_params == himl._get_script_params()


@pytest.mark.fast
@patch("health_azure.himl.Workspace")
@patch("health_azure.himl._generate_azure_datasets")
@patch("health_azure.himl.RUN_CONTEXT")
def test_submit_to_azure_if_needed_azure_return(
        mock_run_context: mock.MagicMock,
        mock_generate_azure_datasets: mock.MagicMock,
        mock_workspace: mock.MagicMock,
) -> None:
    """
    When running in AzureML, the call to submit_to_azure_if_needed should return immediately, without trying to
    submit a new job.
    """
    # The presence of the "experiment" flag is the trigger to recognize an AzureML run.
    mock_run_context.experiment = mock.MagicMock(workspace=mock_workspace)
    assert is_running_in_azure_ml(himl.RUN_CONTEXT)
    expected_run_info = himl.AzureRunInfo(
        run=mock_run_context,
        input_datasets=[],
        output_datasets=[],
        mount_contexts=[],
        is_running_in_azure_ml=True,
        output_folder=Path.cwd() / himl.OUTPUT_FOLDER,
        logs_folder=Path.cwd() / himl.LOGS_FOLDER)
    mock_generate_azure_datasets.return_value = expected_run_info
    with mock.patch("sys.argv", ["", AZUREML_FLAG]):
        run_info = himl.submit_to_azure_if_needed(
            aml_workspace=mock_workspace,
            entry_script=Path(__file__),
            compute_cluster_name="foo",
            conda_environment_file=Path("env.yml"))
    assert run_info == expected_run_info


@pytest.mark.fast
@patch("health_azure.himl.DatasetConfig")
@patch("health_azure.himl.RUN_CONTEXT")
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
    assert run_info.is_running_in_azure_ml
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


class TestTagOption(Enum):
    TAGS = 1  # Set the tags parameter
    NO_TAGS = 2  # Set the tags parameter to None
    ARGS = 3  # script_run_config is a ScriptRunConfig with arguments
    HYPER = 4  # script_run_config is a HyperDriveConfig with arguments


@pytest.mark.fast
@pytest.mark.parametrize("wait_for_completion", [True, False])
@pytest.mark.parametrize("set_tags", [
    TestTagOption.TAGS, TestTagOption.NO_TAGS, TestTagOption.ARGS, TestTagOption.HYPER])
@patch("health_azure.himl.Run")
@patch("health_azure.himl.Experiment")
@patch("health_azure.himl.Workspace")
def test_submit_run(mock_workspace: mock.MagicMock,
                    mock_experiment: mock.MagicMock,
                    mock_run: mock.MagicMock,
                    wait_for_completion: bool,
                    set_tags: TestTagOption,
                    capsys: CaptureFixture
                    ) -> None:
    mock_experiment.return_value.submit.return_value = mock_run
    mock_run.get_status.return_value = RunStatus.COMPLETED
    mock_run.status = RunStatus.COMPLETED
    mock_run.get_children.return_value = []
    mock_tags = {'tag1': '1', 'tag2': '2'}
    mock_arguments = ["--arg1", "--message=\\'Hello World :-)\\'"]
    # Pretend to be a ScriptRunConfig
    mock_script_run_config = mock.MagicMock(spec=[ScriptRunConfig])
    mock_script_run_config.arguments = None
    tags: Optional[Dict[str, str]] = None
    if set_tags == TestTagOption.TAGS:
        tags = mock_tags
    else:
        if set_tags == TestTagOption.ARGS:
            mock_script_run_config.arguments = mock_arguments
        elif set_tags == TestTagOption.HYPER:
            # Pretend to be a HyperDriveConfig
            mock_script_run_config = mock.MagicMock(spec=[HyperDriveConfig])
            mock_script_run_config.run_config = MagicMock()
            mock_script_run_config.run_config.arguments = mock_arguments
    an_experiment_name = "an experiment"
    _ = himl.submit_run(
        workspace=mock_workspace,
        experiment_name=an_experiment_name,
        script_run_config=mock_script_run_config,
        tags=tags,
        wait_for_completion=wait_for_completion,
        wait_for_completion_show_output=True,
    )
    out, err = capsys.readouterr()
    assert not err
    assert "Successfully queued run" in out
    assert "Experiment name and run ID are available" in out
    assert "Experiment URL" in out
    assert "Run URL" in out
    if set_tags == TestTagOption.TAGS:
        mock_run.set_tags.assert_called_once_with(mock_tags)
    elif set_tags == TestTagOption.NO_TAGS:
        mock_run.set_tags.assert_called_once_with(None)
    else:
        mock_run.set_tags.assert_called_once_with({"commandline_args": " ".join(mock_arguments)})
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
def test_submit_run_v2(tmp_path: Path) -> None:
    def _mock_sweep(*args: Any, **kwargs: Any) -> MagicMock:
        assert kwargs.get("compute") == dummy_compute_target
        assert kwargs.get("max_total_trials") == len(values)
        assert kwargs.get("sampling_algorithm") == "grid"
        assert kwargs.get("primary_metric") == "val/loss"
        assert kwargs.get("goal") == "Minimize"
        return mock_command

    dummy_environment_name = "my_environment"
    dummy_environment = MagicMock()
    dummy_environment.name = dummy_environment_name

    dummy_input_data_name = "my_input_dataset"
    dummy_input_path = "path_to_my_input_data"
    dummy_inputs = {
        dummy_input_data_name: Input(  # type: ignore
            type=AssetTypes.URI_FOLDER,
            path=dummy_input_path,
            mode=InputOutputModes.MOUNT
        )
    }

    dummy_output_data_name = "my_output_dataset"
    dummy_output_path = "path_to_my_output_data"
    dummy_outputs = {
        dummy_output_data_name: Output(  # type: ignore
            type=AssetTypes.URI_FOLDER,
            path=dummy_output_path,
            mode=InputOutputModes.MOUNT
        )
    }

    dummy_root_directory = tmp_path
    dummy_entry_script = dummy_root_directory / "my_entry_script"
    dummy_experiment_name = "my_experiment"
    dummy_experiment_name = himl.effective_experiment_name(dummy_experiment_name, dummy_entry_script)
    dummy_entry_script.touch()

    dummy_script_params = ["--arg1=val1", "--arg2=val2", "--conda_env=some_path"]
    dummy_compute_target = "my_compute_target"
    dummy_display_name = "job_display_name"
    dummy_tags = {"tag1": "tag1value"}
    dummy_docker_shm_size = '1g'

    # job without hyperparameter sampling
    with patch("azure.ai.ml.MLClient") as mock_ml_client:
        with patch("health_azure.himl.command") as mock_command:
            himl.submit_run_v2(
                workspace=None,
                experiment_name=dummy_experiment_name,
                environment=dummy_environment,
                input_datasets_v2=dummy_inputs,
                output_datasets_v2=dummy_outputs,
                snapshot_root_directory=dummy_root_directory,
                entry_script=dummy_entry_script,
                script_params=dummy_script_params,
                compute_target=dummy_compute_target,
                tags=dummy_tags,
                docker_shm_size=dummy_docker_shm_size,
                workspace_config_path=None,
                ml_client=mock_ml_client,
                hyperparam_args=None,
                display_name=dummy_display_name,
            )

            expected_arg_str = " ".join(dummy_script_params)
            relative_entry_script = dummy_entry_script.relative_to(dummy_root_directory)
            expected_command = f"python {relative_entry_script} {expected_arg_str}"

            mock_command.assert_called_once_with(
                code=str(dummy_root_directory),
                command=expected_command,
                inputs=dummy_inputs,
                outputs=dummy_outputs,
                environment=dummy_environment_name + "@latest",
                compute=dummy_compute_target,
                experiment_name=dummy_experiment_name,
                tags=dummy_tags,
                shm_size=dummy_docker_shm_size,
                display_name=dummy_display_name,
                distribution=MpiDistribution(process_count_per_instance=1),
                instance_count=1,
                identity=None,
            )

            # job with hyperparameter sampling:
            mock_command.reset_mock()
            # when setting parameter sampling we update the value of mock_command
            mock_command.return_value = mock_command
            # we then update the value of mock_command again when calling sweep
            mock_command.sweep.side_effect = _mock_sweep

            values = [0.1, 0.5, 0.9]
            argument_name = "learning_rate"
            param_sampling = {argument_name: Choice(values)}  # type: ignore
            metric_name = "val/loss"

            dummy_hyperparam_args = {
                himl.MAX_TOTAL_TRIALS_ARG: len(values),
                himl.PARAM_SAMPLING_ARG: param_sampling,
                himl.SAMPLING_ALGORITHM_ARG: "grid",
                himl.PRIMARY_METRIC_ARG: metric_name,
                himl.GOAL_ARG: "Minimize"
            }

            # The hyperparameter to be altered should have been added to the command
            expected_command += " --learning_rate=${{inputs.learning_rate}}"

            himl.submit_run_v2(
                workspace=None,
                experiment_name=dummy_experiment_name,
                environment=dummy_environment,
                input_datasets_v2=dummy_inputs,
                output_datasets_v2=dummy_outputs,
                snapshot_root_directory=dummy_root_directory,
                entry_script=dummy_entry_script,
                script_params=dummy_script_params,
                compute_target=dummy_compute_target,
                tags=dummy_tags,
                docker_shm_size=dummy_docker_shm_size,
                workspace_config_path=None,
                ml_client=mock_ml_client,
                hyperparam_args=dummy_hyperparam_args,
                display_name=dummy_display_name,
            )

            # 'command' should be called with the same args
            print(mock_command.call)
            # command should be called once when initialising the command job and once when updating the param sampling
            assert mock_command.call_count == 2

            mock_command.assert_any_call(
                code=str(dummy_root_directory),
                command=expected_command,
                inputs=dummy_inputs,
                outputs=dummy_outputs,
                environment=dummy_environment_name + "@latest",
                compute=dummy_compute_target,
                experiment_name=dummy_experiment_name,
                tags=dummy_tags,
                shm_size=dummy_docker_shm_size,
                display_name=dummy_display_name,
                distribution=MpiDistribution(process_count_per_instance=1),
                instance_count=1,
                identity=None,
            )

            mock_command.assert_any_call(**param_sampling)
            mock_command.sweep.assert_called_once()
            assert mock_command.experiment_name == dummy_experiment_name

            dummy_entry_script_for_module = "-m Foo.bar run"
            expected_command = f"python {dummy_entry_script_for_module} {expected_arg_str}"

            himl.submit_run_v2(
                workspace=None,
                experiment_name=dummy_experiment_name,
                environment=dummy_environment,
                input_datasets_v2=dummy_inputs,
                output_datasets_v2=dummy_outputs,
                snapshot_root_directory=dummy_root_directory,
                entry_script=dummy_entry_script_for_module,
                script_params=dummy_script_params,
                compute_target=dummy_compute_target,
                tags=dummy_tags,
                docker_shm_size=dummy_docker_shm_size,
                workspace_config_path=None,
                ml_client=mock_ml_client,
                hyperparam_args=None,
                display_name=dummy_display_name,
            )

            mock_command.assert_any_call(
                code=str(dummy_root_directory),
                command=expected_command,
                inputs=dummy_inputs,
                outputs=dummy_outputs,
                environment=dummy_environment_name + "@latest",
                compute=dummy_compute_target,
                experiment_name=dummy_experiment_name,
                tags=dummy_tags,
                shm_size=dummy_docker_shm_size,
                display_name=dummy_display_name,
                distribution=MpiDistribution(process_count_per_instance=1),
                instance_count=1,
                identity=None,
            )


@pytest.mark.fast
def test_str_to_path(tmp_path: Path) -> None:
    assert himl._str_to_path(tmp_path) == tmp_path
    assert himl._str_to_path(str(tmp_path)) == tmp_path


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


def validate_v1_job_outputs(captured: str, run_path: Path, extra_options: Dict[str, Any]) -> str:
    """Assert that a v1 job/run has completed successfully.

    :param captured: The captured output from the test script.
    :param run_path: File path to where the run was executed.
    :raises ValueError: Raises a value error if missing expected output files.
    :return: Log text stored in the v1 log driver file.
    """
    assert EXPECTED_QUEUED in captured
    with check_config_json(run_path, shared_config_json=get_shared_config_json()):
        workspace = get_workspace(aml_workspace=None, workspace_config_path=run_path / WORKSPACE_CONFIG_JSON)

    run = get_most_recent_run(
        run_recovery_file=run_path / himl.RUN_RECOVERY_FILE,
        workspace=workspace,
    )
    if run.status not in [RunStatus.FAILED, RunStatus.COMPLETED, RunStatus.CANCELED]:
        run.wait_for_completion()
    assert run.status == "Completed"
    if "display_name" in extra_options:
        assert run.display_name == extra_options["display_name"], "Display name has not been set"

    # test error case mocking where no log file is present
    log_text_undownloaded = get_driver_log_file_text(run=run, download_file=False)
    assert log_text_undownloaded is None

    # TODO: upgrade to walrus operator when upgrading python version to 3.8+
    # if log_text := get_driver_log_file_text(run=run):
    log_text = get_driver_log_file_text(run=run)

    if log_text is None:
        raise ValueError(
            "The run does not contain any of the following log files: "
            f"{[log_file_path for log_file_path in VALID_LOG_FILE_PATHS]}"
        )

    return log_text


def validate_v2_job_outputs() -> str:
    # TODO: implement this using run recovery files, issue open here:
    # https://github.com/microsoft/hi-ml/issues/785
    return "Not implemented yet"


def render_and_run_test_script(path: Path,
                               run_target: RunTarget,
                               extra_options: Dict[str, Any],
                               extra_args: List[str],
                               expected_pass: bool,
                               suppress_config_creation: bool = False,
                               upload_package: bool = True) -> str:
    """
    Prepare test scripts, submit them, and return response.

    :param path: Where to build the test scripts.
    :param run_target: Where to run the script.
    :param extra_options: Extra options for template rendering.
    :param extra_args: Extra command line arguments for calling script.
    :param expected_pass: Whether this call to subprocess is expected to be successful.
    :param suppress_config_creation: (Optional, defaults to False) do not create a config.json file if none exists
    :param upload_package: If True, upload the current package version to the AzureML docker image. If False,
      skip uploading. Use only if the script does not use hi-ml.
    :return: Either response from spawn_and_monitor_subprocess or run output if in AzureML.
    """
    path.mkdir(exist_ok=True)
    # target hi-ml-azure package version, if specified in an environment variable.
    version = ""
    run_requirements = False

    himl_wheel_filename = os.getenv('HIML_AZURE_WHEEL_FILENAME', '')
    himl_test_pypi_version = os.getenv('HIML_AZURE_TEST_PYPI_VERSION', '')
    himl_pypi_version = os.getenv('HIML_AZURE_PYPI_VERSION', '')

    if not himl_wheel_filename:
        # If testing locally, can build the package into the "dist" folder and use that.
        dist_folder = Path.cwd().joinpath('dist')
        whls = sorted(list(dist_folder.glob('*.whl')))
        if len(whls) > 0:
            last_whl = whls[-1]
            himl_wheel_filename = str(last_whl)

    if himl_wheel_filename and upload_package:
        # Testing against a private wheel.
        himl_wheel_filename_full_path = str(Path(himl_wheel_filename).resolve())
        extra_options['private_pip_wheel_path'] = f'Path("{himl_wheel_filename_full_path}")'
        print(f"Added private_pip_wheel_path: {himl_wheel_filename_full_path} option")
    elif himl_test_pypi_version and upload_package:
        # Testing against test.pypi, add this as the pip_extra_index_url, and set the version.
        extra_options['pip_extra_index_url'] = "https://test.pypi.org/simple/"
        version = himl_test_pypi_version
        print(f"Added test.pypi: {himl_test_pypi_version} option")
    elif himl_pypi_version and upload_package:
        # Testing against pypi, set the version.
        version = himl_pypi_version
        print(f"Added pypi: {himl_pypi_version} option")
    else:
        # No packages found, so copy the src folder as a fallback
        src_path = repository_root() / "hi-ml-azure" / "src"
        if src_path.is_dir():
            shutil.copytree(src=src_path / 'health_azure', dst=path / 'health_azure')
            run_requirements = True
            print("Copied 'src' folder.")

    if run_target == RunTarget.AZUREML:
        extra_options["submit_to_azureml"] = 'True'

    # To easily identify the matching test run in AzureML, we set the display name to the test name.
    if "display_name" not in extra_options:
        extra_options["display_name"] = current_test_name()

    environment_yaml_path = path / "environment.yml"
    render_environment_yaml(environment_yaml_path, version, run_requirements, extra_options=extra_options)

    entry_script_path = path / "test_script.py"
    workspace_config_file_arg = "None" if suppress_config_creation else "WORKSPACE_CONFIG_JSON"
    render_test_script(entry_script_path, extra_options, INEXPENSIVE_TESTING_CLUSTER_NAME, environment_yaml_path,
                       workspace_config_file_arg=workspace_config_file_arg)

    score_args = [str(entry_script_path)]
    score_args.extend(extra_args)

    env = dict(os.environ.items())

    def spawn() -> Tuple[int, List[str]]:
        code, stdout = spawn_and_monitor_subprocess(
            process=sys.executable,
            args=score_args,
            cwd=path,
            env=env)
        return code, stdout
    print(f"Starting the script in {path}")
    if suppress_config_creation:
        code, stdout = spawn()
    else:
        with check_config_json(path, shared_config_json=get_shared_config_json()):
            code, stdout = spawn()
    captured = "\n".join(stdout)
    print(f"Script console output:\n{captured}")
    assert code == 0 if expected_pass else 1, f"Expected the script to {'pass' if expected_pass else 'fail'}, but " \
                                              f"got a return code {code}"

    if run_target == RunTarget.LOCAL or not expected_pass:
        if AZUREML_FLAG in extra_args:
            assert EXPECTED_QUEUED in captured
        else:
            assert EXPECTED_QUEUED not in captured
        return captured
    else:

        if "strictly_aml_v1" not in extra_options or extra_options["strictly_aml_v1"] == "True":
            return validate_v1_job_outputs(captured, path, extra_options)

        else:
            return validate_v2_job_outputs()


@pytest.mark.parametrize("run_target", [RunTarget.LOCAL, RunTarget.AZUREML])
def test_invoking_hello_world_no_config(run_target: RunTarget, tmp_path: Path) -> None:
    """
    Test invoking rendered 'simple' / 'hello_world_template.txt' when there is no config file in the current working
    directory, and no workspace is specified via environment variables. This should pass fine for local runs,
    but fail when trying to submit to AzureML.

    :param run_target: Where to run the script.
    :param tmp_path: PyTest test fixture for temporary path.
    """
    parser_args = "parser.add_argument('-m', '--message', type=str, required=True, help='The message to print out')"
    message_guid = uuid4().hex
    extra_options = {
        'workspace_config_file': 'None',
        'args': parser_args,
        'body': 'print(f"The message was: {args.message}")'
    }
    extra_args = [f"--message={message_guid}"]
    expected_output = f"The message was: {message_guid}"
    # Setting only one of the environment variables to an empty string should trigger the failure when submitting
    with patch.dict(os.environ, {ENV_WORKSPACE_NAME: ""}):
        if run_target == RunTarget.LOCAL:
            output = render_and_run_test_script(tmp_path, run_target, extra_options, extra_args,
                                                expected_pass=True,
                                                suppress_config_creation=True)
            assert expected_output in output
        else:
            output = render_and_run_test_script(tmp_path, run_target, extra_options, extra_args,
                                                expected_pass=False,
                                                suppress_config_creation=True)
            assert "Tried all ways of identifying the workspace" in output
            assert expected_output not in output


@pytest.mark.parametrize("run_target", [RunTarget.LOCAL, RunTarget.AZUREML])
@pytest.mark.parametrize("use_package", [True, False])
def test_invoking_hello_world_config(run_target: RunTarget, use_package: bool, tmp_path: Path) -> None:
    """
    Test that invoking hello_world.py elevates itself to AzureML with config.json.
    Test against either the local src folder or a package. If running locally, ensure that there
    are no whl's in the dist folder, or that will be used.

    :param run_target: Local execution if True, else in AzureML.
    :param use_package: True to test against package, False to test against copy of src folder.
    :param tmp_path: PyTest test fixture for temporary path.
    """
    if not use_package and \
            not os.getenv('HIML_AZURE_WHEEL_FILENAME', '') and \
            not os.getenv('HIML_AZURE_TEST_PYPI_VERSION', '') and \
            not os.getenv('HIML_AZURE_PYPI_VERSION', ''):
        # Running locally, no need to duplicate this test.
        return

    message_guid = uuid4().hex
    parser_args = "parser.add_argument('-m', '--message', type=str, required=True, help='The message to print out')"
    extra_options = {
        'args': parser_args,
        'body': 'print(f"The message was: {args.message}")',
        'display_name': current_test_name(),
    }
    extra_args = [f"--message={message_guid}"]
    if use_package:
        output = render_and_run_test_script(tmp_path, run_target, extra_options, extra_args, True)
    else:
        with mock.patch.dict(os.environ, {"HIML_AZURE_WHEEL_FILENAME": '',
                                          "HIML_AZURE_TEST_PYPI_VERSION": '',
                                          "HIML_AZURE_PYPI_VERSION": ''}):
            output = render_and_run_test_script(tmp_path, run_target, extra_options, extra_args, True)
    expected_output = f"The message was: {message_guid}"
    assert expected_output in output


def test_invoking_hello_world_using_azureml_flag(tmp_path: Path) -> None:
    """
    Test that invoking hello_world.py with the --azureml flag will submit to AzureML and not run locally.

    :param tmp_path: Pytest fixture for temporary path.
    """

    message_guid = uuid4().hex
    parser_args = "parser.add_argument('-m', '--message', type=str, required=True, help='The message to print out')"
    extra_options = {
        'args': parser_args,
        'body': 'print(f"The message was: {args.message}")',
        'submit_to_azureml': None,
    }
    extra_args = [f"--message={message_guid}", AZUREML_FLAG]
    output = render_and_run_test_script(tmp_path, RunTarget.LOCAL, extra_options, extra_args, True)
    expected_output = f"The message was: {message_guid}"
    assert expected_output in output


@patch("health_azure.himl.submit_to_azure_if_needed")
def test_calling_script_directly(mock_submit_to_azure_if_needed: mock.MagicMock) -> None:
    with mock.patch("sys.argv", ["",
                                 "--workspace_config_file", "1",
                                 "--compute_cluster_name", "2",
                                 "--snapshot_root_directory", "3",
                                 "--entry_script", "4",
                                 "--conda_environment_file", "5"]):
        himl.main()
    assert mock_submit_to_azure_if_needed.call_args[1]["workspace_config_file"] == PosixPath("1")
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
    with mock.patch.dict(os.environ, {"HIML_AZURE_WHEEL_FILENAME": 'not_a_known_file.whl'}):
        output = render_and_run_test_script(tmp_path, RunTarget.AZUREML, extra_options, extra_args, False)
    error_message_begin = "FileNotFoundError: Cannot add private wheel"
    assert error_message_begin in output


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
        "imports": """
import os
import sys""",
        'environment_variables': f"{{'message_guid': '{message_guid}'}}",
        'body': 'print(f"The message_guid env var was: {os.getenv(\'message_guid\')}")',
    }

    extra_args: List[str] = []
    output = render_and_run_test_script(tmp_path, run_target, extra_options, extra_args, True)
    expected_output = f"The message_guid env var was: {message_guid}"
    assert expected_output in output


def _assert_hello_world_files_exist(folder: Path) -> None:
    """Check if the .csv files in the hello_world dataset exist in the given folder."""
    files = []
    for file in folder.rglob("*.csv"):
        file_relative = file.relative_to(folder)
        logging.info(f"File {file_relative}: size: {file.stat().st_size}")
        files.append(str(file_relative))
    assert set(files) == {
        "dataset.csv",
        "train_and_test_data/metrics.csv",
        "train_and_test_data/metrics_aggregates.csv",
        "train_and_test_data/scalar_epoch_metrics.csv",
        "train_and_test_data/scalar_prediction_target_metrics.csv"
    }


@pytest.mark.timeout(120)
def test_mounting_and_downloading_dataset(tmp_path: Path) -> None:
    logging.info("creating config.json")
    with check_config_json(tmp_path, shared_config_json=get_shared_config_json()):
        logging.info("get_workspace")
        workspace = get_workspace(aml_workspace=None,
                                  workspace_config_path=tmp_path / WORKSPACE_CONFIG_JSON)
        # Loop inside the test because getting the workspace is quite time-consuming
        for use_mounting in [True, False]:
            logging.info(f"use_mounting={use_mounting}")
            action = "mount" if use_mounting else "download"
            target_path = tmp_path / action
            dataset_config = DatasetConfig(name="hello_world",
                                           use_mounting=use_mounting,
                                           target_folder=target_path)
            logging.info(f"ready to {action}")
            paths, mount_contexts = setup_local_datasets(
                dataset_configs=[dataset_config],
                strictly_aml_v1=True,
                aml_workspace=workspace
            )
            logging.info(f"{action} done")
            path = paths[0]
            assert path is not None
            _assert_hello_world_files_exist(path)
            for mount_context in mount_contexts:
                mount_context.stop()


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
    folder_name: Optional[Path] = None
    # Contents of test file.
    contents: str = ""
    # Local folder str
    local_folder: str = ""


@dataclass
class TestOutputDataset:
    # Name of container for this dataset in blob storage.
    blob_name: str
    # Local folder for this dataset when running locally or when testing after running in Azure.
    folder_name: Optional[Path] = None


@pytest.mark.parametrize(["run_target", "local_folder", "strictly_aml_v1"],
                         [(RunTarget.LOCAL, True, False),
                          (RunTarget.AZUREML, False, True),
                          ])
# Test with AML SDK v2 fails, logged as https://github.com/microsoft/hi-ml/issues/763
# (RunTarget.AZUREML, False, False)
def test_invoking_hello_world_datasets(run_target: RunTarget,
                                       local_folder: bool,
                                       strictly_aml_v1: bool,
                                       tmp_path: Path) -> None:
    """
    Test that invoking rendered 'simple' / 'hello_world_template.txt' elevates itself to AzureML with config.json,
    and that datasets are mounted in all combinations.

    :param run_target: Where to run the script.
    :param local_folder: True to use data in local folder when running locally, False to mount/download data.
    :param strictly_aml_v1: If True, use only AML v1 features. If False, use AML v2 features if available.
    :param tmp_path: PyTest test fixture for temporary path.
    """
    input_count = 5
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
    with check_config_json(tmp_path, shared_config_json=get_shared_config_json()):
        workspace = get_workspace(aml_workspace=None,
                                  workspace_config_path=tmp_path / WORKSPACE_CONFIG_JSON)
        datastore: AzureBlobDatastore = get_datastore(workspace=workspace,
                                                      datastore_name=DEFAULT_DATASTORE)

    # Create dummy txt files, one for each item in input_datasets.
    for input_dataset in input_datasets:
        input_dataset.contents = _create_test_file_in_blobstore(
            datastore=datastore,
            filename=input_dataset.filename,
            location=input_dataset.blob_name,
            tmp_path=tmp_path)

        if run_target == RunTarget.LOCAL and local_folder:
            # For running locally, download the test files from blobstore
            downloaded = datastore.download(
                target_path=input_dataset.folder_name,
                prefix=f"{input_dataset.blob_name}/{input_dataset.filename}",
                overwrite=True,
                show_progress=True)
            assert downloaded == 1

            # Check that the input file is downloaded
            assert input_dataset.folder_name is not None
            downloaded_dummy_txt_file = input_dataset.folder_name / input_dataset.blob_name / input_dataset.filename
            # Check it has expected contents
            assert input_dataset.contents == downloaded_dummy_txt_file.read_text()
            input_dataset.local_folder = f", local_folder='{input_dataset.folder_name / input_dataset.blob_name}'"

    if run_target == RunTarget.LOCAL:
        for output_dataset in output_datasets:
            assert output_dataset.folder_name is not None
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

    extra_options: Dict[str, str] = {
        'imports': """
import shutil
import sys
""",
        'prequel': """
    target_folders = ["foo", "bar"]
        """,
        'default_datastore': f'"{DEFAULT_DATASTORE}"',
        'input_datasets': f"""[
            "{input_datasets[0].blob_name}",
            DatasetConfig(name="{input_datasets[1].blob_name}",
                          datastore="{DEFAULT_DATASTORE}"{input_datasets[1].local_folder}),
            DatasetConfig(name="{input_datasets[2].blob_name}",
                          datastore="{DEFAULT_DATASTORE}",
                          target_folder=target_folders[0]{input_datasets[2].local_folder}),
            DatasetConfig(name="{input_datasets[3].blob_name}",
                          datastore="{DEFAULT_DATASTORE}",
                          use_mounting=True{input_datasets[3].local_folder}),
            DatasetConfig(name="{input_datasets[4].blob_name}",
                          datastore="{DEFAULT_DATASTORE}",
                          target_folder=target_folders[1],
                          use_mounting=True{input_datasets[4].local_folder}),
        ]""",
        'output_datasets': f"""[
            "{output_datasets[0].blob_name}",
            DatasetConfig(name="{output_datasets[1].blob_name}", datastore="{DEFAULT_DATASTORE}"),
            DatasetConfig(name="{output_datasets[2].blob_name}", datastore="{DEFAULT_DATASTORE}",
                          use_mounting=False),
        ]""",
        'strictly_aml_v1': str(strictly_aml_v1),
        'body': f"""
    input_datasets = [
        {script_input_datasets}
    ]
    output_datasets = [
        {script_output_datasets}
    ]
    for i, (filename, input_blob_name, input_folder_name) in enumerate(input_datasets):
        print(f"input_folder: {{run_info.input_datasets[i]}} or {{input_folder_name / input_blob_name}}")
        input_folder = run_info.input_datasets[i] or input_folder_name / input_blob_name
        for j, (output_blob_name, output_folder_name) in enumerate(output_datasets):
            print(f"output_folder: {{run_info.output_datasets[j]}} or {{output_folder_name / output_blob_name}}")
            output_folder = run_info.output_datasets[j] or output_folder_name / output_blob_name
            file = input_folder / filename
            shutil.copy(file, output_folder)
            print(f"Copied file: {{file.name}} from {{input_blob_name}} to {{output_blob_name}}")
        """,
    }
    extra_args: List[str] = []
    # Support for private wheels is only available in SDK v1, hence only upload the package when using v1.
    # Test script rendering will upload the full source folder instead.
    upload_package = strictly_aml_v1
    output = render_and_run_test_script(tmp_path, run_target, extra_options, extra_args,
                                        expected_pass=True, upload_package=upload_package)

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

            assert isinstance(output_dataset.folder_name, Path)
            output_dummy_txt_file = output_dataset.folder_name / output_dataset.blob_name / input_dataset.filename
            assert input_dataset.contents == output_dummy_txt_file.read_text()

# endregion Elevate to AzureML unit tests


def test_invoking_user_identity_datasets(tmp_path: Path) -> None:
    output_test_file_name = f"test_output_{uuid4().hex}.txt"
    extra_options: Dict[str, str] = {
        'imports': """
import shutil
import sys
import os
""",
        'default_datastore': f"'{USER_IDENTITY_TEST_DATASTORE}'",
        'input_datasets': f"['{USER_IDENTITY_TEST_ASSET}']",
        'output_datasets': f"['{USER_IDENTITY_TEST_ASSET_OUTPUT}']",
        'strictly_aml_v1': str(False),
        'identity_based_auth': str(True),
        'body': f"""

    input_folder = run_info.input_datasets[0]
    output_folder = run_info.output_datasets[0]

    print("input dir contents: " + str(os.listdir(input_folder)))

    output_dir_contents = os.listdir(output_folder)
    num_output_items = len(output_dir_contents)

    print("output dir contents before copying: " + str(output_dir_contents))
    print("Number of items in output dir before copying: " + str(num_output_items))

    input_file = input_folder / "{USER_IDENTITY_TEST_FILE}"
    output_file = output_folder / "{output_test_file_name}"

    print('input file: ' + str(input_file) + ', output file: ' + str(output_file))

    if os.path.exists(input_file):
        print("Input file exists")

    if not os.path.exists(output_file):
        print("Output file does not exist (yet)")

    print("Copying file...")
    shutil.copy(input_file, output_file)

    num_output_dir_items_after_copying = len(os.listdir(output_folder))

    assert num_output_dir_items_after_copying == num_output_items + 1, "Copied file not present in output dir"

    print("File successfully copied!")
        """,
    }

    extra_args: List[str] = []

    render_and_run_test_script(
        tmp_path,
        RunTarget.AZUREML,
        extra_options,
        extra_args,
        expected_pass=True,
        upload_package=False,
    )


@pytest.mark.fast
# Azure ML expects run_config to be instance of ScriptRunConfig
@patch("azureml.train.hyperdrive.runconfig.isinstance", return_value=True)
@pytest.mark.parametrize("num_crossval_splits, metric_name, cross_val_index_arg_name", [
    (-1, "val/loss", "cross_validation_split_index"),
    (0, "loss", "cross_validation_split_index"),
    (1, "val/acc", "split"),
    (5, "accuracy", "data_split")
])
def test_create_crossval_hyperdrive_config(_: MagicMock, num_crossval_splits: int, metric_name: str,
                                           cross_val_index_arg_name: str) -> None:
    if num_crossval_splits < 1:
        with pytest.raises(Exception):
            himl.create_crossval_hyperdrive_config(num_splits=num_crossval_splits,
                                                   cross_val_index_arg_name=cross_val_index_arg_name,
                                                   metric_name=metric_name)
    else:
        crossval_config = himl.create_crossval_hyperdrive_config(num_splits=num_crossval_splits,
                                                                 cross_val_index_arg_name=cross_val_index_arg_name,
                                                                 metric_name=metric_name)
        assert isinstance(crossval_config, HyperDriveConfig)
        assert crossval_config._primary_metric_config.get("name") == metric_name
        assert crossval_config._primary_metric_config.get("goal") == "minimize"
        assert crossval_config._max_total_runs == num_crossval_splits


def test_create_crossval_hyperparam_args_v2() -> None:
    num_splits = 3
    crossval_args = himl.create_crossval_hyperparam_args_v2(num_splits)
    assert isinstance(crossval_args, Dict)
    assert crossval_args[himl.MAX_TOTAL_TRIALS_ARG] == num_splits
    assert isinstance(crossval_args[himl.PARAM_SAMPLING_ARG], Dict)
    assert isinstance(crossval_args[himl.PARAM_SAMPLING_ARG]["crossval_index"], Choice)
    assert crossval_args[himl.PRIMARY_METRIC_ARG] == "val/loss"
    assert crossval_args[himl.SAMPLING_ALGORITHM_ARG] == "grid"
    assert crossval_args[himl.GOAL_ARG] == "Minimize"


def test_create_grid_hyperparam_args_v2() -> None:
    mock_values_float = [0.1, 0.2, 0.5]
    mock_arg_name_float = "float_number"
    mock_metric_name_float = mock_arg_name_float
    hparams_args_float = himl.create_grid_hyperparam_args_v2(mock_values_float, mock_arg_name_float,
                                                             mock_metric_name_float)
    assert isinstance(hparams_args_float, Dict)
    assert hparams_args_float[himl.MAX_TOTAL_TRIALS_ARG] == len(mock_values_float)
    assert isinstance(hparams_args_float[himl.PARAM_SAMPLING_ARG], Dict)
    assert isinstance(hparams_args_float[himl.PARAM_SAMPLING_ARG][mock_arg_name_float], Choice)
    assert hparams_args_float[himl.PRIMARY_METRIC_ARG] == mock_metric_name_float
    assert hparams_args_float[himl.SAMPLING_ALGORITHM_ARG] == "grid"
    assert hparams_args_float[himl.GOAL_ARG] == "Minimize"

    mock_values_str = ["a", "b", "c"]
    mock_arg_name_str = "letter"
    mock_metric_name_str = mock_arg_name_str
    hparam_args_str = himl.create_grid_hyperparam_args_v2(mock_values_str, mock_arg_name_str,
                                                          mock_metric_name_str)
    assert isinstance(hparam_args_str, Dict)
    assert hparam_args_str[himl.MAX_TOTAL_TRIALS_ARG] == len(mock_values_str)
    assert isinstance(hparam_args_str[himl.PARAM_SAMPLING_ARG], Dict)
    assert isinstance(hparam_args_str[himl.PARAM_SAMPLING_ARG][mock_arg_name_str], Choice)
    assert hparam_args_str[himl.PRIMARY_METRIC_ARG] == mock_metric_name_str
    assert hparam_args_str[himl.SAMPLING_ALGORITHM_ARG] == "grid"
    assert hparam_args_str[himl.GOAL_ARG] == "Minimize"


@pytest.mark.fast
@pytest.mark.parametrize("cross_validation_metric_name", [None, "accuracy"])
@patch("sys.argv")
@patch("health_azure.himl.exit")
def test_submit_to_azure_if_needed_with_hyperdrive(mock_sys_args: MagicMock,
                                                   mock_exit: MagicMock,
                                                   mock_compute_cluster: MagicMock,
                                                   cross_validation_metric_name: Optional[str],
                                                   ) -> None:
    """
    Test that himl.submit_to_azure_if_needed can be called, and returns immediately.
    """
    cross_validation_metric_name = cross_validation_metric_name or ""
    mock_sys_args.return_value = ["", AZUREML_FLAG]
    with patch("health_azure.himl.get_ml_client") as mock_get_ml_client:
        mock_ml_client = MagicMock()
        mock_get_ml_client.return_value = mock_ml_client
        with patch.object(Environment, "get", return_value="dummy_env"):
            mock_workspace = MagicMock()
            mock_workspace.compute_targets = {"foo": mock_compute_cluster}
            with patch("health_azure.datasets.setup_local_datasets") as mock_setup_local_datasets:
                mock_setup_local_datasets.return_value = [], []
                with patch("health_azure.himl.submit_run") as mock_submit_run:
                    with patch("health_azure.himl.HyperDriveConfig") as mock_hyperdrive_config:
                        crossval_config = himl.create_crossval_hyperdrive_config(
                            num_splits=2,
                            cross_val_index_arg_name="cross_val_split_index",
                            metric_name=cross_validation_metric_name)
                        himl.submit_to_azure_if_needed(
                            aml_workspace=mock_workspace,
                            ml_client=mock_ml_client,
                            entry_script=Path(__file__),
                            snapshot_root_directory=Path(__file__).parent,
                            compute_cluster_name="foo",
                            aml_environment_name="dummy_env",
                            submit_to_azureml=True,
                            hyperdrive_config=crossval_config,
                            strictly_aml_v1=True)
                        mock_submit_run.assert_called_once()
                        mock_hyperdrive_config.assert_called_once()


def test_get_data_asset_from_config() -> None:
    n_configs = 3
    test_dataset_configs = [
        DatasetConfig(
            name=TEST_DATA_ASSET_NAME,
            datastore=TEST_DATASTORE_NAME,
        ) for _ in range(n_configs)
    ]

    test_assets = [
        himl.get_data_asset_from_config(TEST_ML_CLIENT, test_dataset_config)
        for test_dataset_config in test_dataset_configs
    ]
    assert len(test_assets) == n_configs
    assert all([asset.name == TEST_DATA_ASSET_NAME for asset in test_assets])

    test_version = 1
    test_versioned_config = DatasetConfig(
        name=TEST_DATA_ASSET_NAME,
        datastore=TEST_DATASTORE_NAME,
        version=test_version,
    )
    test_versioned_asset = himl.get_data_asset_from_config(TEST_ML_CLIENT, test_versioned_config)
    assert test_versioned_asset.version == str(test_version)


@pytest.mark.fast
@pytest.mark.parametrize("already_exists", [True, False])
def test_create_v2_inputs(already_exists: bool) -> None:
    mock_ml_client = MagicMock()
    mock_data_name = "some_arbitrary_name"
    # These values are copied from an actual Data item
    mock_data_version = "1"
    mock_data_id = ("/subscriptions/123/resourceGroups/myrg/providers/Microsoft.MachineLearningServices/workspaces/"
                    f"myws/data/{mock_data_name}/versions/{mock_data_version}")
    mock_data_path = "azureml://subscriptions/123/resourcegroups/myrg/workspaces/myws/datastores/ds/paths/foldername/**"
    # This is normally "uri_folder", but we want to test if that value is passed through unchanged
    mock_data_type = "some_arbitrary_type"
    mock_asset = Data(
        name=mock_data_name,
        version=mock_data_version,
        id=mock_data_id,
        path=mock_data_path,
        type=mock_data_type,
    )
    if already_exists:
        mock_ml_client.data.get.return_value = mock_asset
    else:
        def mock_v2_data_get(*args: Any, **kwargs: Any) -> None:
            raise ResourceNotFoundError("dummy error")
        mock_ml_client.data.get.side_effect = mock_v2_data_get
        mock_ml_client.data.create_or_update.return_value = mock_asset

    for use_mounting in [True, False]:
        mock_input_dataconfigs = [DatasetConfig(
            name="dummy_dataset",
            use_mounting=use_mounting,
            version=int(mock_data_version)
        )]
        inputs = himl.create_v2_inputs(mock_ml_client, mock_input_dataconfigs)
        assert isinstance(inputs, Dict)
        assert len(inputs) == len(mock_input_dataconfigs)
        input_entry = inputs["INPUT_0"]
        assert isinstance(input_entry, Input)
        # This value should be passed through unchanged
        assert input_entry.type == mock_data_type
        assert input_entry.path == mock_data_path  # type: ignore
        if use_mounting:
            assert input_entry.mode == InputOutputModes.MOUNT
        else:
            assert input_entry.mode == InputOutputModes.DOWNLOAD


@pytest.mark.fast
@pytest.mark.parametrize("missing", [None, ""])
def test_create_v2_inputs_fails(missing: Any) -> None:

    with patch("health_azure.himl._get_or_create_v2_data_asset") as mock_get_or_create_v2_data_asset:
        mock_get_or_create_v2_data_asset.return_value = MagicMock(path=missing)
        mock_ml_client = MagicMock()
        mock_input_dataconfigs = [DatasetConfig(name="dummy_dataset")]
        with pytest.raises(ValueError, match="has no path"):
            himl.create_v2_inputs(mock_ml_client, mock_input_dataconfigs)


@pytest.mark.fast
def test_create_v2_outputs() -> None:
    test_dataset_config = DatasetConfig(
        name=TEST_DATA_ASSET_NAME,
        datastore=TEST_DATASTORE_NAME,
    )
    outputs = himl.create_v2_outputs(TEST_ML_CLIENT, [test_dataset_config])

    output_key = "OUTPUT_0"
    assert output_key in outputs
    assert len(outputs) == 1

    output = outputs[output_key]
    assert output.path is not None
    assert isinstance(output, Output)
    assert output.mode == InputOutputModes.MOUNT


def test_submit_to_azure_if_needed_v2() -> None:
    """
    Check that submit_run_v2 is called when submit_to_azure_if_needed is called, unless strictly_aml_v1 is
    set to True, in which case submit_run should be called instead
    """
    dummy_input_datasets: List[Optional[Path]] = []
    dummy_mount_contexts: List[MountContext] = []

    with patch.multiple(
        "health_azure.himl",
        health_azure_package_setup=DEFAULT,
        get_workspace=DEFAULT,
        get_ml_client=DEFAULT,
        create_run_configuration=DEFAULT,
        create_script_run=DEFAULT,
        append_to_amlignore=DEFAULT,
        exit=DEFAULT
    ) as mocks:
        mock_script_run = mocks["create_script_run"].return_value
        mock_script_run.script = "dummy_script"
        mock_script_run.source_directory = "dummy_dir"

        with patch("health_azure.himl.setup_local_datasets") as mock_setup_datasets:
            mock_setup_datasets.return_value = dummy_input_datasets, dummy_mount_contexts
            with patch("health_azure.himl.submit_run_v2") as mock_submit_run_v2:
                return_value = himl.submit_to_azure_if_needed(
                    workspace_config_file="mockconfig.json",
                    snapshot_root_directory="dummy",
                    submit_to_azureml=True,
                    strictly_aml_v1=False
                )
                mock_submit_run_v2.assert_called_once()
                assert return_value is None

            # Now supply strictly_aml_v1=True, and check that submit_run is called
            with patch("health_azure.himl.submit_run") as mock_submit_run:
                return_value = himl.submit_to_azure_if_needed(
                    workspace_config_file="mockconfig.json",
                    snapshot_root_directory="dummy",
                    submit_to_azureml=True,
                    strictly_aml_v1=True,
                )
                mock_submit_run.assert_called_once()
                assert return_value is None


@pytest.mark.parametrize("wait_for_completion", [True, False])
def test_submitting_script_with_sdk_v2(tmp_path: Path, wait_for_completion: bool) -> None:
    """
    Test that submitting a simple script can be run on AzureML when using the v2 SDK.
    It also tests the "wait_for_completion" parameter.
    """
    # Create a minimal script in a temp folder. Script
    test_script = tmp_path / "test_script.py"
    test_script.write_text("print('hello world')")
    shared_config_json = get_shared_config_json()
    conda_env_path = create_empty_conda_env(tmp_path)
    after_submission_called = False

    def after_submission(job: Job, ml_client: MLClient) -> None:
        nonlocal after_submission_called
        after_submission_called = True
        assert isinstance(job, Job)
        assert isinstance(ml_client, MLClient)
        # If waiting for completion then the job should be completed, otherwise it should be starting
        if wait_for_completion:
            assert job.status == JobStatus.COMPLETED.value
        else:
            assert job.status == JobStatus.STARTING.value

    with check_config_json(tmp_path, shared_config_json=shared_config_json),\
            change_working_directory(tmp_path), \
            pytest.raises(SystemExit):
        himl.submit_to_azure_if_needed(
            aml_workspace=None,
            experiment_name="test_submitting_script_with_sdk_v2",
            entry_script=test_script,
            compute_cluster_name=INEXPENSIVE_TESTING_CLUSTER_NAME,
            conda_environment_file=conda_env_path,
            snapshot_root_directory=tmp_path,
            submit_to_azureml=True,
            after_submission=after_submission,
            strictly_aml_v1=False,
            wait_for_completion=wait_for_completion,
        )

    assert after_submission_called, "after_submission callback was not called"


def test_submitting_script_with_sdk_v2_passes_display_name(tmp_path: Path) -> None:
    """
    Test that submission of a script with SDK v2 passes the display_name parameter to the "command" function
    that does the actual submission
    """
    # Create a minimal script in a temp folder.
    test_script = tmp_path / "test_script.py"
    test_script.write_text("print('hello world')")
    shared_config_json = get_shared_config_json()
    conda_env_path = create_empty_conda_env(tmp_path)
    display_name = "my_display_name"

    with check_config_json(tmp_path, shared_config_json=shared_config_json),\
            change_working_directory(tmp_path), \
            patch("health_azure.himl.command", side_effect=ValueError) as mock_command:
        with pytest.raises(ValueError):
            himl.submit_to_azure_if_needed(
                aml_workspace=None,
                entry_script=test_script,
                conda_environment_file=conda_env_path,
                snapshot_root_directory=tmp_path,
                submit_to_azureml=True,
                strictly_aml_v1=False,
                display_name=display_name
            )
        mock_command.assert_called_once()
        _, call_kwargs = mock_command.call_args
        assert call_kwargs.get("display_name") == display_name, "display_name was not passed to command"


def test_conda_env_missing(tmp_path: Path) -> None:
    """
    Test that submission fails if no Conda environment file is found.
    """
    # Create a minimal script in a temp folder.
    test_script = tmp_path / "test_script.py"
    test_script.write_text("print('hello world')")
    shared_config_json = get_shared_config_json()

    with check_config_json(tmp_path, shared_config_json=shared_config_json), \
            change_working_directory(tmp_path), \
            pytest.raises(ValueError, match="No conda environment file"):
        himl.submit_to_azure_if_needed(
            aml_workspace=None,
            experiment_name="test_conda_env_missing",
            entry_script=test_script,
            compute_cluster_name=INEXPENSIVE_TESTING_CLUSTER_NAME,
            snapshot_root_directory=tmp_path,
            submit_to_azureml=True,
        )


@pytest.mark.fast
def test_experiment_name() -> None:
    """Test the logic for choosing experiment names: Explicitly given experiment name should be used if provided,
    otherwise fall back to environment variables and thpwden script name."""
    # When the test suite runs on the Github, the environment variable "HIML_EXPERIMENT_NAME" will be set.
    # Remove it to test the default behaviour.
    with mock.patch.dict(os.environ):
        os.environ.pop(ENV_EXPERIMENT_NAME, None)
        assert himl.effective_experiment_name("explicit", Path()) == "explicit"
        assert himl.effective_experiment_name("", Path("from_script.py")) == "from_script"
        # Provide experiment names with special characters here that should be filtered out
        with mock.patch("health_azure.himl.to_azure_friendly_string", return_value="mock_return"):
            assert himl.effective_experiment_name("explicit", Path()) == "mock_return"
        assert himl.effective_experiment_name("explicit/", Path()) == "explicit_"
        with mock.patch.dict(os.environ, {ENV_EXPERIMENT_NAME: "name/from/env"}):
            assert himl.effective_experiment_name("explicit", Path()) == "name_from_env"
    with mock.patch.dict(os.environ, {ENV_EXPERIMENT_NAME: "name_from_env"}):
        assert himl.effective_experiment_name("explicit", Path()) == "name_from_env"
        assert himl.effective_experiment_name("", Path("from_script.py")) == "name_from_env"


@pytest.mark.fast
def test_submit_to_azure_v2_distributed() -> None:
    dummy_input_datasets: List[Optional[Path]] = []
    dummy_mount_contexts: List[MountContext] = []

    with patch.multiple(
        "health_azure.himl",
        health_azure_package_setup=DEFAULT,
        get_workspace=DEFAULT,
        get_ml_client=DEFAULT,
        create_run_configuration=DEFAULT,
        create_script_run=DEFAULT,
        append_to_amlignore=DEFAULT,
        exit=DEFAULT
    ) as _:

        with patch("health_azure.himl.setup_local_datasets") as mock_setup_datasets:
            mock_setup_datasets.return_value = dummy_input_datasets, dummy_mount_contexts
            with patch("health_azure.himl.submit_run_v2") as mock_submit_run_v2:
                # If num_nodes and processes_per_node are not passed to submit_to_azure_if_needed, then
                # submit_run_v2 should receive the default values of 1 and 1 respectively
                expected_num_nodes = 1
                _ = himl.submit_to_azure_if_needed(
                    workspace_config_file="mockconfig.json",
                    snapshot_root_directory="dummy",
                    submit_to_azureml=True,
                    strictly_aml_v1=False
                )

                _, call_kwargs = mock_submit_run_v2.call_args
                print(call_kwargs)
                assert call_kwargs.get("num_nodes") == expected_num_nodes
                assert call_kwargs.get("pytorch_processes_per_node") is None

                # If num_nodes and processes_per_node are both passed in to submit_to_azure_if_needed,
                # they should be passed through to submit_run_v2
                num_nodes = 3
                processes_per_node = 4
                _ = himl.submit_to_azure_if_needed(
                    workspace_config_file="mockconfig.json",
                    snapshot_root_directory="dummy",
                    submit_to_azureml=True,
                    strictly_aml_v1=False,
                    num_nodes=num_nodes,
                    pytorch_processes_per_node_v2=processes_per_node
                )
                _, call_kwargs = mock_submit_run_v2.call_args
                assert call_kwargs.get("num_nodes") == num_nodes
                assert call_kwargs.get("pytorch_processes_per_node") == processes_per_node

            # Single node job: The "distribution" argument of "command" should be set to None
            with patch("health_azure.himl.command") as mock_command:
                _ = himl.submit_to_azure_if_needed(
                    workspace_config_file="mockconfig.json",
                    entry_script=Path(__file__),
                    snapshot_root_directory=Path.cwd(),
                    submit_to_azureml=True,
                    strictly_aml_v1=False,
                )
                mock_command.assert_called_once()
                _, call_kwargs = mock_command.call_args
                assert call_kwargs.get("instance_count") == 1
                assert call_kwargs.get("distribution") == MpiDistribution(process_count_per_instance=1)

            with pytest.raises(ValueError, match="num_nodes must be >= 1"):
                _ = himl.submit_to_azure_if_needed(
                    workspace_config_file="mockconfig.json",
                    entry_script=Path(__file__),
                    snapshot_root_directory=Path.cwd(),
                    submit_to_azureml=True,
                    strictly_aml_v1=False,
                    num_nodes=0
                )

            with pytest.raises(ValueError, match="pytorch_processes_per_node must be >= 1"):
                _ = himl.submit_to_azure_if_needed(
                    workspace_config_file="mockconfig.json",
                    entry_script=Path(__file__),
                    snapshot_root_directory=Path.cwd(),
                    submit_to_azureml=True,
                    strictly_aml_v1=False,
                    num_nodes=1,
                    pytorch_processes_per_node_v2=0,
                )

            # When specifying the number of pytorch processes, a PyTorchDistribution job should be created
            num_nodes = 2
            num_processes = 3
            with patch("health_azure.himl.command") as mock_command:
                _ = himl.submit_to_azure_if_needed(
                    workspace_config_file="mockconfig.json",
                    entry_script=Path(__file__),
                    snapshot_root_directory=Path.cwd(),
                    submit_to_azureml=True,
                    strictly_aml_v1=False,
                    num_nodes=num_nodes,
                    pytorch_processes_per_node_v2=num_processes
                )
                mock_command.assert_called_once()
                _, call_kwargs = mock_command.call_args
                assert call_kwargs.get("instance_count") == num_nodes
                distribution = call_kwargs.get("distribution")
                assert isinstance(distribution, PyTorchDistribution)
                # pytorch_processes_per_node_v2=0 should be updated to 1
                assert distribution.process_count_per_instance == num_processes

            # If num_nodes is specified without processes_per_node, an MPI job should be created.
            with patch("health_azure.himl.command") as mock_command:
                _ = himl.submit_to_azure_if_needed(
                    workspace_config_file="mockconfig.json",
                    entry_script=Path(__file__),
                    snapshot_root_directory=Path.cwd(),
                    submit_to_azureml=True,
                    strictly_aml_v1=False,
                    num_nodes=num_nodes,
                )
                mock_command.assert_called_once()
                _, call_kwargs = mock_command.call_args
                assert call_kwargs.get("instance_count") == num_nodes
                distribution = call_kwargs.get("distribution")
                assert distribution == MpiDistribution(process_count_per_instance=1)


@pytest.mark.fast
def test_extract_v2_data_asset_from_env_vars() -> None:
    valid_mock_environment = {
        "AZURE_ML_INPUT_INPUT_0": "input_0",
        "AZURE_ML_OUTPUT_OUTPUT_0": "output_0",
    }

    with patch.dict(os.environ, valid_mock_environment):
        input_dataset_0 = himl._extract_v2_data_asset_from_env_vars(0, "INPUT_")
        output_dataset_0 = himl._extract_v2_data_asset_from_env_vars(0, "OUTPUT_")
        assert input_dataset_0 == Path("input_0")
        assert output_dataset_0 == Path("output_0")

        with pytest.raises(ValueError):
            himl._extract_v2_data_asset_from_env_vars(5, "OUTPUT_")

    valid_mock_environment = {
        "AZURE_ML_INPUT_INPUT_2": "input_2",
        "AZURE_ML_INPUT_INPUT_1": "input_1",
        "AZURE_ML_INPUT_INPUT_0": "input_0",
        "AZURE_ML_INPUT_INPUT_3": "input_3",
    }

    with patch.dict(os.environ, valid_mock_environment):
        input_datasets = [
            himl._extract_v2_data_asset_from_env_vars(i, "INPUT_")
            for i in range(len(valid_mock_environment))
        ]
        assert input_datasets == [Path("input_0"), Path("input_1"), Path("input_2"), Path("input_3")]
