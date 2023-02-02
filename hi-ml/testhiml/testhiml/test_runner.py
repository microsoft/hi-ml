#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from contextlib import contextmanager
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
from unittest.mock import patch, MagicMock, DEFAULT, create_autospec

import pytest
from _pytest.capture import SysCapture
from azureml.train.hyperdrive import HyperDriveConfig

from health_azure import AzureRunInfo, DatasetConfig
from health_azure.paths import ENVIRONMENT_YAML_FILE_NAME
from health_ml.configs.hello_world import HelloWorld  # type: ignore
from health_ml.deep_learning_config import WorkflowParams
from health_ml.experiment_config import DEBUG_DDP_ENV_VAR, DebugDDPOptions
from health_ml.lightning_container import LightningContainer
from health_ml.runner import Runner
from health_ml.utils.common_utils import change_working_directory
from health_ml.utils.fixed_paths import repository_root_directory


@pytest.fixture
def mock_runner(tmp_path: Path) -> Runner:
    """A test fixture that creates a Runner object in a temporary folder.
    """

    return Runner(project_root=tmp_path)


@contextmanager
def change_working_folder_and_add_environment(tmp_path: Path) -> Generator:
    # Use a special simplified environment file only for the tests here. Copy that to a temp folder, then let the runner
    # start in that temp folder.
    env_file = repository_root_directory() / "hi-ml" / "testhiml" / ENVIRONMENT_YAML_FILE_NAME
    shutil.copy(env_file, tmp_path)
    with change_working_directory(tmp_path):
        yield


@pytest.mark.parametrize("model_name, cluster, num_nodes, should_raise_value_error", [
    ("HelloWorld", "dummyCluster", 1, False),
    ("", "", None, True),
    ("HelloWorld", "", None, False),
    ("a", None, 0, True),
    (None, "b", 10, True),
    ("HelloWorld", "b", 10, False)
])
def test_parse_and_load_model(mock_runner: Runner, model_name: Optional[str], cluster: Optional[str],
                              num_nodes: Optional[int], should_raise_value_error: bool) -> None:
    """
    Test that command line args are parsed, a LightningContainer is instantiated with the expected attributes
    and a ParserResult object is returned, with the expected attributes. If model_name cannot be found in the
    namespace (i.e. the config does not exist) a ValueError should be raised
    """
    dummy_args = [""]
    if model_name is not None:
        dummy_args.append(f"--model={model_name}")
    if cluster is not None:
        dummy_args.append(f"--cluster={cluster}")
    if num_nodes is not None:
        dummy_args.append(f"--num_nodes={num_nodes}")

    with patch.object(sys, "argv", new=dummy_args):
        if should_raise_value_error:
            with pytest.raises(ValueError) as ve:
                mock_runner.parse_and_load_model()
                assert "Parameter 'model' needs to be set" in str(ve)
        else:
            parser_result = mock_runner.parse_and_load_model()
            # if model, cluster or num_nodes are provdided in command line args, the corresponding attributes of
            # the LightningContainer will be set accordingly and they will be dropped from ParserResult during
            # parse_overrides_and_apply
            assert parser_result.args.get("model") is None
            assert parser_result.args.get("cluster") is None
            assert parser_result.args.get("num_nodes") is None

            assert isinstance(mock_runner.lightning_container, LightningContainer)
            assert mock_runner.lightning_container.initialized
            assert mock_runner.lightning_container.model_name == model_name


@pytest.mark.parametrize("debug_ddp", ["OFF", "INFO", "DETAIL"])
def test_ddp_debug_flag(debug_ddp: DebugDDPOptions, mock_runner: Runner) -> None:
    model_name = "HelloWorld"
    arguments = ["", f"--debug_ddp={debug_ddp}", f"--model={model_name}"]
    with patch("health_ml.runner.submit_to_azure_if_needed") as mock_submit_to_azure_if_needed:
        with patch("health_ml.runner.get_workspace"):
            with patch("health_ml.runner.Runner.run_in_situ"):
                with patch.object(sys, "argv", arguments):
                    mock_runner.run()
        mock_submit_to_azure_if_needed.assert_called_once()
        assert mock_submit_to_azure_if_needed.call_args[1]["environment_variables"][DEBUG_DDP_ENV_VAR] == debug_ddp


def test_additional_aml_run_tags(mock_runner: Runner) -> None:
    model_name = "HelloWorld"
    arguments = ["", f"--model={model_name}", "--cluster=foo"]
    with patch("health_ml.runner.submit_to_azure_if_needed") as mock_submit_to_azure_if_needed:
        with patch("health_ml.runner.check_conda_environment"):
            with patch("health_ml.runner.get_workspace"):
                with patch("health_ml.runner.get_ml_client"):
                    with patch("health_ml.runner.Runner.run_in_situ"):
                        with patch.object(sys, "argv", arguments):
                            mock_runner.run()
        mock_submit_to_azure_if_needed.assert_called_once()
        assert "commandline_args" in mock_submit_to_azure_if_needed.call_args[1]["tags"]
        assert "tag" in mock_submit_to_azure_if_needed.call_args[1]["tags"]
        assert "max_epochs" in mock_submit_to_azure_if_needed.call_args[1]["tags"]


def test_additional_environment_variables(mock_runner: Runner) -> None:
    model_name = "HelloWorld"
    arguments = ["", f"--model={model_name}", "--cluster=foo"]
    with patch.multiple(
        "health_ml.runner",
        submit_to_azure_if_needed=DEFAULT,
        check_conda_environment=DEFAULT,
        get_workspace=DEFAULT,
        get_ml_client=DEFAULT,
    ) as mocks:
        with patch("health_ml.runner.Runner.run_in_situ"):
            with patch("health_ml.runner.Runner.parse_and_load_model"):
                with patch("health_ml.runner.Runner.validate"):
                    with patch.object(sys, "argv", arguments):
                        mock_container = create_autospec(LightningContainer)
                        mock_container.get_additional_environment_variables = MagicMock(return_value={"foo": "bar"})
                        mock_runner.lightning_container = mock_container
                        mock_runner.run()
        mocks["submit_to_azure_if_needed"].assert_called_once()
        mock_env_vars = mocks["submit_to_azure_if_needed"].call_args[1]["environment_variables"]
        assert DEBUG_DDP_ENV_VAR in mock_env_vars
        assert "foo" in mock_env_vars
        assert mock_env_vars["foo"] == "bar"


def test_run(mock_runner: Runner) -> None:
    model_name = "HelloWorld"
    arguments = ["", f"--model={model_name}"]
    with patch("health_ml.runner.Runner.run_in_situ") as mock_run_in_situ:
        with patch("health_ml.runner.get_workspace"):
            with patch.object(sys, "argv", arguments):
                model_config, azure_run_info = mock_runner.run()
        mock_run_in_situ.assert_called_once()

    assert model_config is not None  # for pyright
    assert model_config.model_name == model_name
    assert azure_run_info.run is None
    assert len(azure_run_info.input_datasets) == len(azure_run_info.output_datasets) == 0


@patch("health_ml.runner.choose_conda_env_file")
@patch("health_ml.runner.get_workspace")
@pytest.mark.fast
def test_submit_to_azureml_if_needed(mock_get_workspace: MagicMock,
                                     mock_get_env_files: MagicMock,
                                     mock_runner: Runner
                                     ) -> None:
    def _mock_dont_submit_to_aml(input_datasets: List[DatasetConfig],
                                 submit_to_azureml: bool, strictly_aml_v1: bool,  # type: ignore
                                 environment_variables: Dict[str, Any],  # type: ignore
                                 default_datastore: Optional[str],  # type: ignore
                                 ) -> AzureRunInfo:
        datasets_input = [d.target_folder for d in input_datasets] if input_datasets else []
        return AzureRunInfo(input_datasets=datasets_input,
                            output_datasets=[],
                            mount_contexts=[],
                            run=None,
                            is_running_in_azure_ml=False,
                            output_folder=None,  # type: ignore
                            logs_folder=None)  # type: ignore

    mock_get_env_files.return_value = Path("some_file.txt")

    mock_default_datastore = MagicMock()
    mock_default_datastore.name.return_value = "dummy_datastore"
    mock_get_workspace.get_default_datastore.return_value = mock_default_datastore

    with patch("health_ml.runner.create_dataset_configs") as mock_create_datasets:
        mock_create_datasets.return_value = []
        with patch("health_ml.runner.submit_to_azure_if_needed") as mock_submit_to_aml:
            mock_submit_to_aml.side_effect = _mock_dont_submit_to_aml
            mock_runner.lightning_container = LightningContainer()
            run_info = mock_runner.submit_to_azureml_if_needed()
            assert isinstance(run_info, AzureRunInfo)
            assert run_info.input_datasets == []
            assert run_info.is_running_in_azure_ml is False
            assert run_info.output_folder is None


def test_crossvalidation_flag() -> None:
    """
    Checks the basic use of the flags that trigger cross validation
    :return:
    """
    container = HelloWorld()
    assert not container.is_crossvalidation_parent_run
    assert not container.is_crossvalidation_child_run
    container.crossval_count = 5
    container.crossval_index = None
    assert container.is_crossvalidation_parent_run
    assert not container.is_crossvalidation_child_run
    container.crossval_index = 0
    assert container.is_crossvalidation_child_run
    container.validate()
    # Try all valid values
    for i in range(container.crossval_count):
        container.crossval_index = i
        container.validate()
    # Validation should fail if the cross validation index is out of bounds
    container.crossval_index = container.crossval_count
    with pytest.raises(ValueError):
        container.validate()


def test_crossval_config() -> None:
    """
    Check if the flags to trigger Hyperdrive runs work as expected.
    """
    mock_tuning_config = "foo"
    container = HelloWorld()
    with patch("health_ml.configs.hello_world.HelloWorld.get_parameter_tuning_config",
               return_value=mock_tuning_config):
        # Without any flags set, no Hyperdrive config should be returned
        assert container.get_hyperdrive_config() is None
        # To trigger a hyperparameter search, the commandline flag for hyperdrive must be present
        container.hyperdrive = True
        assert container.get_hyperdrive_config() == mock_tuning_config
        # Triggering cross validation works by just setting crossval_count
        container.hyperdrive = False
        container.crossval_count = 2
        assert container.is_crossvalidation_parent_run
        crossval_config = container.get_hyperdrive_config()
        assert isinstance(crossval_config, HyperDriveConfig)


@pytest.mark.fast
def test_crossval_argument_names() -> None:
    """
    Cross validation uses hardcoded argument names, check if they match the field names
    """
    container = HelloWorld()
    crossval_count = 8
    crossval_index = 5
    random_seed = 4711
    container.crossval_count = crossval_count
    container.crossval_index = crossval_index
    container.random_seed = random_seed
    assert getattr(container, container.CROSSVAL_INDEX_ARG_NAME) == crossval_index
    assert getattr(container, container.RANDOM_SEED_ARG_NAME) == random_seed


def test_submit_to_azure_hyperdrive(mock_runner: Runner) -> None:
    """
    Test if the hyperdrive configurations are passed to the submission function if using cross validation.
    """
    crossval_count = 2
    _test_hyperdrive_submission(mock_runner,
                                commandline_arg=f"--crossval_count={crossval_count}",
                                expected_argument_name=WorkflowParams.CROSSVAL_INDEX_ARG_NAME,
                                expected_argument_values=list(map(str, range(crossval_count))))


def test_submit_to_azure_differents_seeds(mock_runner: Runner) -> None:
    """
    Test if the hyperdrive configurations are passed to the submission function if running with dfferent seeds.
    """
    num_seeds = 2
    _test_hyperdrive_submission(mock_runner,
                                commandline_arg=f"--different_seeds={num_seeds}",
                                expected_argument_name=WorkflowParams.RANDOM_SEED_ARG_NAME,
                                expected_argument_values=list(map(str, range(num_seeds))))


def _test_hyperdrive_submission(mock_runner: Runner,
                                commandline_arg: str,
                                expected_argument_name: str,
                                expected_argument_values: List[str]) -> None:
    model_name = "HelloWorld"
    arguments = ["", f"--model={model_name}", "--cluster=foo", commandline_arg, "--strictly_aml_v1=True"]
    # Use a special simplified environment file only for the tests here. Copy that to a temp folder, then let the runner
    # start in that temp folder.
    with change_working_folder_and_add_environment(mock_runner.project_root):
        with patch("health_ml.runner.Runner.run_in_situ") as mock_run_in_situ:
            with patch("health_ml.runner.get_workspace"):
                with patch("health_ml.runner.get_ml_client"):
                    with patch.object(sys, "argv", arguments):
                        with patch("health_ml.runner.submit_to_azure_if_needed") as mock_submit_to_aml:
                            mock_runner.run()
            mock_run_in_situ.assert_called_once()
            mock_submit_to_aml.assert_called_once()
            # call_args is a tuple of (args, kwargs)
            call_kwargs = mock_submit_to_aml.call_args[1]
            # Submission to AzureML should have been turned on because a cluster name was supplied
            assert mock_runner.experiment_config.cluster == "foo"
            assert call_kwargs["submit_to_azureml"]
            # Check details of the Hyperdrive config
            hyperdrive_config = call_kwargs["hyperdrive_config"]
            parameter_space = hyperdrive_config._generator_config["parameter_space"]
            assert expected_argument_name in parameter_space
            assert parameter_space[expected_argument_name] == ["choice", [expected_argument_values]]


def test_submit_to_azure_docker(mock_runner: Runner) -> None:
    """
    Test if the docker arguments are passed through to the submission function.
    """
    model_name = "HelloWorld"
    docker_shm_size = "100k"
    arguments = ["", f"--model={model_name}", "--cluster=foo", f"--docker_shm_size={docker_shm_size}"]
    # Use a special simplified environment file only for the tests here. Copy that to a temp folder, then let the runner
    # start in that temp folder.
    with change_working_folder_and_add_environment(mock_runner.project_root):
        with patch("health_ml.runner.Runner.run_in_situ") as mock_run_in_situ:
            with patch("health_ml.runner.get_ml_client"):
                with patch("health_ml.runner.get_workspace"):
                    with patch.object(sys, "argv", arguments):
                        with patch("health_ml.runner.submit_to_azure_if_needed") as mock_submit_to_aml:
                            mock_runner.run()
            mock_run_in_situ.assert_called_once()
            mock_submit_to_aml.assert_called_once()
            # call_args is a tuple of (args, kwargs)
            call_kwargs = mock_submit_to_aml.call_args[1]
            # Submission to AzureML should have been turned on because a cluster name was supplied
            assert mock_runner.experiment_config.docker_shm_size == docker_shm_size
            assert call_kwargs["docker_shm_size"] == docker_shm_size


def test_runner_help(mock_runner: Runner, capsys: SysCapture) -> None:
    """Test if the runner outputs default values correctly then using --help
    """
    arguments = ["", "--help"]
    with pytest.raises(SystemExit):
        with patch.object(sys, "argv", arguments):
            mock_runner.run()
    stdout: str = capsys.readouterr().out  # type: ignore
    # There are at least 3 parameters in ExperimentConfig that should print with defaults
    assert stdout.count("(default: ") > 3


def test_run_hello_world(mock_runner: Runner) -> None:
    """Test running a model end-to-end via the commandline runner
    """
    model_name = "HelloWorld"
    arguments = ["", f"--model={model_name}"]
    with patch("health_ml.runner.get_workspace") as mock_get_workspace:
        with patch.object(sys, "argv", arguments):
            mock_runner.run()
        # get_workspace should not be called when using the runner outside AzureML, to not go through the
        # time-consuming auth
        mock_get_workspace.assert_not_called()
        # Summary.txt is written at start, the other files during inference
        expected_files = ["experiment_summary.txt", "test_mae.txt", "test_mse.txt"]
        for file in expected_files:
            assert (mock_runner.lightning_container.outputs_folder / file).is_file(), f"Missing file: {file}"


def test_invalid_args(mock_runner: Runner) -> None:
    """Test if invalid commandline arguments raise an error.
    """
    invalid_arg = "--no_such_argument"
    arguments = ["", "--model=HelloWorld", invalid_arg]
    with patch.object(sys, "argv", arguments):
        with pytest.raises(ValueError) as ex:
            mock_runner.run()
        assert "Unknown arguments" in str(ex)
        assert invalid_arg in str(ex)


def test_invalid_profiler(mock_runner: Runner) -> None:
    """Test if invalid profiler commandline arguments raise an error.
    """
    invalid_profile = "--pl_profiler=foo"
    arguments = ["", "--model=HelloWorld", invalid_profile]
    with patch.object(sys, "argv", arguments):
        with pytest.raises(ValueError) as ex:
            mock_runner.run()
        assert "Unsupported profiler." in str(ex)


def test_custom_datastore_outside_aml(mock_runner: Runner) -> None:
    model_name = "HelloWorld"
    datastore = "foo"
    arguments = ["", f"--datastore={datastore}", f"--model={model_name}"]
    with patch("health_ml.runner.submit_to_azure_if_needed") as mock_submit_to_azure_if_needed:
        with patch("health_ml.runner.get_workspace"):
            with patch("health_ml.runner.Runner.run_in_situ"):
                with patch.object(sys, "argv", arguments):
                    mock_runner.run()
        mock_submit_to_azure_if_needed.assert_called_once()
        assert mock_submit_to_azure_if_needed.call_args[1]["default_datastore"] == datastore
