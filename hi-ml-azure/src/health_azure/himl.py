#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

"""
Wrapper functions for running local Python scripts on Azure ML.

See examples/elevate_this.py for a very simple 'hello world' example of use.
"""

import logging
import os
import re
import sys
from argparse import ArgumentParser
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

from azure.ai.ml import MLClient, Input, Output, command
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.entities import Data, Job, Command, Sweep
from azure.ai.ml.entities import Environment as EnvironmentV2
from azure.ai.ml.entities._job.distribution import MpiDistribution, PyTorchDistribution

from azure.ai.ml.sweep import Choice
from azureml._base_sdk_common import user_agent
from azureml.core import ComputeTarget, Environment, Experiment, Run, RunConfiguration, ScriptRunConfig, Workspace
from azureml.core.runconfig import DockerConfiguration, MpiConfiguration
from azureml.data import OutputFileDatasetConfig
from azureml.data.dataset_consumption_config import DatasetConsumptionConfig
from azureml.train.hyperdrive import HyperDriveConfig, GridParameterSampling, PrimaryMetricGoal, choice
from azureml.dataprep.fuse.daemon import MountContext

from health_azure.amulet import (ENV_AMLT_DATAREFERENCE_DATA, ENV_AMLT_DATAREFERENCE_OUTPUT, is_amulet_job)
from health_azure.package_setup import health_azure_package_setup
from health_azure.utils import (ENV_EXPERIMENT_NAME, create_python_environment, create_run_recovery_id,
                                find_file_in_parent_to_pythonpath,
                                is_run_and_child_runs_completed, is_running_in_azure_ml, register_environment,
                                run_duration_string_to_seconds, to_azure_friendly_string, RUN_CONTEXT, get_workspace,
                                PathOrString, DEFAULT_ENVIRONMENT_VARIABLES, get_ml_client,
                                create_python_environment_v2, register_environment_v2, V2_INPUT_DATASET_PATTERN,
                                V2_OUTPUT_DATASET_PATTERN, wait_for_job_completion)
from health_azure.datasets import (DatasetConfig, StrOrDatasetConfig, setup_local_datasets,
                                   _input_dataset_key, _output_dataset_key, _replace_string_datasets)


logger = logging.getLogger('health_azure')
logger.setLevel(logging.DEBUG)

AML_IGNORE_FILE = ".amlignore"
AZUREML_FLAG = "--azureml"
CONDA_ENVIRONMENT_FILE = "environment.yml"
LOGS_FOLDER = "logs"
OUTPUT_FOLDER = "outputs"
RUN_RECOVERY_FILE = "most_recent_run.txt"
SDK_NAME = "innereye"
SDK_VERSION = "2.0"

DEFAULT_DOCKER_BASE_IMAGE = "mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.2-cudnn8-ubuntu18.04"
DEFAULT_DOCKER_SHM_SIZE = "100g"

# hyperparameter search args
PARAM_SAMPLING_ARG = "parameter_sampling"
MAX_TOTAL_TRIALS_ARG = "max_total_trials"
PRIMARY_METRIC_ARG = "primary_metric"
SAMPLING_ALGORITHM_ARG = "sampling_algorithm"
GOAL_ARG = "goal"


@dataclass
class AzureRunInfo:
    """
    This class stores all information that a script needs to run inside and outside of AzureML. It is return
    from `submit_to_azure_if_needed`, where the return value depends on whether the script is inside or outside
    AzureML.

    Please check the source code for detailed documentation for all fields.
    """
    input_datasets: List[Optional[Path]]
    """A list of folders that contain all the datasets that the script uses as inputs. Input datasets must be
     specified when calling `submit_to_azure_if_needed`. Here, they are made available as Path objects. If no input
     datasets are specified, the list is empty."""
    output_datasets: List[Optional[Path]]
    """A list of folders that contain all the datasets that the script uses as outputs. Output datasets must be
         specified when calling `submit_to_azure_if_needed`. Here, they are made available as Path objects. If no output
         datasets are specified, the list is empty."""
    mount_contexts: List[MountContext]
    """A list of mount contexts for input datasets when running outside AzureML. There will be a mount context
    for each input dataset where there is no local_folder, there is a workspace, and use_mounting is set.
    This list is maintained only to prevent exit from these contexts until the RunInfo object is deleted."""
    run: Optional[Run]
    """An AzureML Run object if the present script is executing inside AzureML, or None if outside of AzureML.
    The Run object has methods to log metrics, upload files, etc."""
    is_running_in_azure_ml: bool
    """If True, the present script is executing inside AzureML. If False, outside AzureML."""
    output_folder: Path
    """The output folder into which all script outputs should be written, if they should be later available in the
    AzureML portal. Files written to this folder will be uploaded to blob storage at the end of the script run."""
    logs_folder: Path
    """The folder into which all log files (for example, tensorboard) should be written. All files written to this
    folder will be uploaded to blob storage regularly during the script run."""


def validate_num_nodes(compute_cluster: ComputeTarget, num_nodes: int) -> None:
    """
    Check that the user hasn't requested more nodes than the maximum number of nodes allowed by
    their compute cluster

    :param compute_cluster: An AML ComputeTarget representing the cluster whose upper node limit
        should be checked
    :param num_nodes: The number of nodes that the user has requested
    """
    max_cluster_nodes: int = compute_cluster.scale_settings.maximum_node_count
    if num_nodes > max_cluster_nodes:
        raise ValueError(
            f"You have requested {num_nodes} nodes, which is more than your compute cluster "
            f"({compute_cluster.name})'s maximum of {max_cluster_nodes} nodes.")


def validate_compute_name(existing_compute_targets: Dict[str, ComputeTarget], compute_target_name: str) -> None:
    """
    Check that a specified compute target is one of the available existing compute targets in a Workspace

    :param existing_compute_targets: A list of AML ComputeTarget objects available to a given AML Workspace
    :param compute_cluster_name: The name of the specific compute target whose name to look up in existing
        compute targets
    """
    if compute_target_name not in existing_compute_targets:
        raise ValueError(f"Could not find the compute target {compute_target_name} in the AzureML workspace. ",
                         f"Existing compute targets: {list(existing_compute_targets)}")


def validate_compute_cluster(workspace: Workspace, compute_cluster_name: str, num_nodes: int) -> None:
    """
    Check that both the specified compute cluster exists in the given Workspace, and that it has enough
    nodes to spin up the requested number of nodes

    :param existing_compute_clusters: A list of AML ComputeTarget objects in a given AML Workspace
    :param compute_cluster_name: The name of the specific compute cluster whose properties should be checked
    :param num_nodes: The number of nodes that the user has requested
    """
    existing_compute_clusters: Dict[str, ComputeTarget] = workspace.compute_targets
    validate_compute_name(existing_compute_clusters, compute_cluster_name)
    compute_cluster = existing_compute_clusters[compute_cluster_name]
    validate_num_nodes(compute_cluster, num_nodes)


def create_run_configuration(workspace: Workspace,
                             compute_cluster_name: str,
                             conda_environment_file: Optional[Path] = None,
                             aml_environment_name: str = "",
                             environment_variables: Optional[Dict[str, str]] = None,
                             pip_extra_index_url: str = "",
                             private_pip_wheel_path: Optional[Path] = None,
                             docker_base_image: str = "",
                             docker_shm_size: str = "",
                             num_nodes: int = 1,
                             max_run_duration: str = "",
                             input_datasets: Optional[List[DatasetConfig]] = None,
                             output_datasets: Optional[List[DatasetConfig]] = None,
                             ) -> RunConfiguration:
    """
    Creates an AzureML run configuration, that contains information about environment, multi node execution, and
    Docker.

    :param workspace: The AzureML Workspace to use.
    :param aml_environment_name: The name of an AzureML environment that should be used to submit the script. If not
        provided, an environment will be created from the arguments to this function (conda_environment_file,
        pip_extra_index_url, environment_variables, docker_base_image)
    :param max_run_duration: The maximum runtime that is allowed for this job in AzureML. This is given as a
        floating point number with a string suffix s, m, h, d for seconds, minutes, hours, day. Examples: '3.5h', '2d'
    :param compute_cluster_name: The name of the AzureML cluster that should run the job. This can be a cluster with
        CPU or GPU machines.
    :param conda_environment_file: The conda configuration file that describes which packages are necessary for your
        script to run.
    :param environment_variables: The environment variables that should be set when running in AzureML.
    :param docker_base_image: The Docker base image that should be used when creating a new Docker image.
    :param docker_shm_size: The Docker shared memory size that should be used when creating a new Docker image.
    :param pip_extra_index_url: If provided, use this PIP package index to find additional packages when building
        the Docker image.
    :param private_pip_wheel_path: If provided, add this wheel as a private package to the AzureML workspace.
    :param conda_environment_file: The file that contains the Conda environment definition.
    :param input_datasets: The script will consume all data in folder in blob storage as the input. The folder must
        exist in blob storage, in the location that you gave when creating the datastore. Once the script has run, it
        will also register the data in this folder as an AzureML dataset.
    :param output_datasets: The script will create a temporary folder when running in AzureML, and while the job writes
        data to that folder, upload it to blob storage, in the data store.
    :param num_nodes: The number of nodes to use in distributed training on AzureML.
    :return:
    """
    run_config = RunConfiguration()

    if aml_environment_name:
        run_config.environment = Environment.get(workspace, aml_environment_name)
    elif conda_environment_file:
        # Create an AzureML environment, then check if it exists already. If it exists, use the registered
        # environment, otherwise register the new environment.
        new_environment = create_python_environment(
            conda_environment_file=conda_environment_file,
            pip_extra_index_url=pip_extra_index_url,
            workspace=workspace,
            private_pip_wheel_path=private_pip_wheel_path,
            docker_base_image=docker_base_image)
        conda_deps = new_environment.python.conda_dependencies
        if conda_deps.get_python_version() is None:
            raise ValueError("If specifying a conda environment file, you must specify the python version within it")
        registered_env = register_environment(workspace, new_environment)
        run_config.environment = registered_env
    else:
        raise ValueError("One of the two arguments 'aml_environment_name' or 'conda_environment_file' must be given.")

    # By default, include several environment variables that work around known issues in the software stack
    run_config.environment_variables = {**DEFAULT_ENVIRONMENT_VARIABLES, **(environment_variables or {})}

    if docker_shm_size:
        run_config.docker = DockerConfiguration(use_docker=True, shm_size=docker_shm_size)

    validate_compute_cluster(workspace, compute_cluster_name, num_nodes)

    run_config.target = compute_cluster_name

    if max_run_duration:
        run_config.max_run_duration_seconds = run_duration_string_to_seconds(max_run_duration)

    # Create MPI configuration for distributed jobs (unless num_splits > 1, in which case
    # an AML HyperdriveConfig is instantiated instead
    if num_nodes > 1:
        distributed_job_config = MpiConfiguration(node_count=num_nodes)
        run_config.mpi = distributed_job_config
        run_config.framework = "Python"
        run_config.communicator = "IntelMpi"
        run_config.node_count = distributed_job_config.node_count

    if input_datasets or output_datasets:
        inputs, outputs = convert_himl_to_azureml_datasets(cleaned_input_datasets=input_datasets or [],
                                                           cleaned_output_datasets=output_datasets or [],
                                                           workspace=workspace,
                                                           strictly_aml_v1=True
                                                           )
        run_config.data = inputs
        run_config.output_data = outputs

    return run_config


def create_grid_hyperdrive_config(values: List[str],
                                  argument_name: str,
                                  metric_name: str) -> HyperDriveConfig:
    """
    Creates an Azure ML HyperDriveConfig object that runs a simple grid search. The Hyperdrive job will run one child
    job for each of the values provided in `values`, and each child job will have a suffix added to the commandline
    like `--argument_name value`.

    Note: this config expects that a metric is logged in your training script([see here](
    https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters#log-metrics-for-hyperparameter-tuning))
    that will be monitored by Hyperdrive. The name of this metric is given by `metric_name`.

    :param values: The list of values to try for the commandline argument given by `argument_name`.
    :param argument_name: The name of the commandline argument that each of the child runs gets, to
        indicate which value they should work on.
    :param metric_name: The name of the metric that the HyperDriveConfig will compare runs by. Please note that it is
        your responsibility to make sure a metric with this name is logged to the Run in your training script
    :return: an Azure ML HyperDriveConfig object
    """
    logging.info(f"Creating a HyperDriveConfig. Please note that this expects to find the specified "
                 f"metric '{metric_name}' logged to AzureML from your training script (for example, using the "
                 f"AzureMLLogger with Pytorch Lightning)")
    parameter_dict = {
        argument_name: choice(values),
    }
    return HyperDriveConfig(
        run_config=ScriptRunConfig(""),
        hyperparameter_sampling=GridParameterSampling(parameter_dict),
        primary_metric_name=metric_name,
        primary_metric_goal=PrimaryMetricGoal.MINIMIZE,
        max_total_runs=len(values)
    )


def create_grid_hyperparam_args_v2(values: List[Any],
                                   argument_name: str,
                                   metric_name: str) -> Dict[str, Any]:
    """
    Create a dictionary of arguments to create an Azure ML v2 SDK Sweep job.

    :param values: The list of values to try for the commandline argument given by `argument_name`.
    :param argument_name: The name of the commandline argument that each of the child runs gets, to
        indicate which value they should work on.
    :param metric_name: The name of the metric that the sweep job will compare runs by. Please note that it is
        your responsibility to make sure a metric with this name is logged to the Run in your training script
    :return: A dictionary of arguments and values to pass in to the command job.
    """
    param_sampling = {argument_name: Choice(values)}
    hyperparam_args = {
        MAX_TOTAL_TRIALS_ARG: len(values),
        PARAM_SAMPLING_ARG: param_sampling,
        SAMPLING_ALGORITHM_ARG: "grid",
        PRIMARY_METRIC_ARG: metric_name,
        GOAL_ARG: "Minimize"
    }
    return hyperparam_args


def create_crossval_hyperdrive_config(num_splits: int,
                                      cross_val_index_arg_name: str = "crossval_index",
                                      metric_name: str = "val/loss") -> HyperDriveConfig:
    """
    Creates an Azure ML HyperDriveConfig object for running cross validation. Note: this config expects a metric
    named <metric_name> to be logged in your training script([see here](
    https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters#log-metrics-for-hyperparameter-tuning))

    :param num_splits: The number of splits for k-fold cross validation
    :param cross_val_index_arg_name: The name of the commandline argument that each of the child runs gets, to
        indicate which split they should work on.
    :param metric_name: The name of the metric that the HyperDriveConfig will compare runs by. Please note that it is
        your responsibility to make sure a metric with this name is logged to the Run in your training script
    :return: an Azure ML HyperDriveConfig object
    """
    return create_grid_hyperdrive_config(values=list(map(str, range(num_splits))),
                                         argument_name=cross_val_index_arg_name,
                                         metric_name=metric_name)


def create_crossval_hyperparam_args_v2(num_splits: int,
                                       cross_val_index_arg_name: str = "crossval_index",
                                       metric_name: str = "val/loss") -> Dict[str, Any]:
    """
    Create a dictionary of arguments to create an Azure ML v2 SDK Sweep job.

    :param num_splits: The number of splits for k-fold cross validation
    :param cross_val_index_arg_name: The name of the commandline argument that each of the child runs gets, to
        indicate which split they should work on.
    :param metric_name: The name of the metric that the HyperDriveConfig will compare runs by. Please note that it is
        your responsibility to make sure a metric with this name is logged to the Run in your training script
    :return: A dictionary of arguments and values to pass in to the command job.
    """
    return create_grid_hyperparam_args_v2(values=list(map(str, range(num_splits))),
                                          argument_name=cross_val_index_arg_name,
                                          metric_name=metric_name)


def create_script_run(
    script_params: List[str],
    snapshot_root_directory: Optional[Path] = None,
    entry_script: Optional[PathOrString] = None,
) -> ScriptRunConfig:
    """
    Creates an AzureML ScriptRunConfig object, that holds the information about the snapshot, the entry script, and
    its arguments.

    :param script_params: A list of parameter to pass on to the script as it runs in AzureML. Required arg. Script
        parameters can be generated using the ``_get_script_params()`` function.
    :param snapshot_root_directory: The directory that contains all code that should be packaged and sent to AzureML.
        All Python code that the script uses must be copied over.
    :param entry_script: The script that should be run in AzureML. If None, the current main Python file will be
        executed.
    :return:
    """
    if snapshot_root_directory is None:
        print("No snapshot root directory given. All files in the current working directory will be copied to AzureML.")
        snapshot_root_directory = Path.cwd()
    else:
        print(f"All files in this folder will be copied to AzureML: {snapshot_root_directory}")
    if entry_script is None:
        entry_script = Path(sys.argv[0])
        print("No entry script given. The current main Python file will be executed in AzureML.")
    elif isinstance(entry_script, str):
        entry_script = Path(entry_script)
    if entry_script.is_absolute():
        try:
            # The entry script always needs to use Linux path separators, even when submitting from Windows
            entry_script_relative = entry_script.relative_to(snapshot_root_directory).as_posix()
        except ValueError:
            raise ValueError("The entry script must be inside of the snapshot root directory. "
                             f"Snapshot root: {snapshot_root_directory}, entry script: {entry_script}")
    else:
        entry_script_relative = str(entry_script)
    print(f"This command will be run in AzureML: {entry_script_relative} {' '.join(script_params)}")
    return ScriptRunConfig(
        source_directory=str(snapshot_root_directory),
        script=entry_script_relative,
        arguments=script_params)


def _generate_input_dataset_command(input_datasets_v2: Dict[str, Input]) -> str:
    """
    Generate command line arguments to pass AML v2 data assets into a script

    :param input_datasets_v2: A dictionary of Input objects that have been passed into the AML command
    :return: A string representing the input datasets that the script should expect
    """
    input_cmd = ""
    for i, (input_data_name, input_dataset_v2) in enumerate(input_datasets_v2.items()):
        input_name = f"INPUT_{i}"
        input_str = "${{inputs." + f"{input_name}" + "}}"
        input_cmd += f" --{input_name}={input_str}"
    return input_cmd


def _generate_output_dataset_command(output_datasets_v2: Dict[str, Output]) -> str:
    """
    Generate command line arguments to pass AML v2 outputs into a script

    :param output_datasets_v2: A dictionary of Output objects that have been passed into the AML command
    :return: A string representing the output values that the script should expect
    """
    output_cmd = ""
    for i, (output_data_name, output_dataset_v2) in enumerate(output_datasets_v2.items()):
        output_name = f"OUTPUT_{i}"
        output_str = "${{outputs." + f"{output_name}" + "}}"
        output_cmd += f" --{output_name}={output_str}"
    return output_cmd


def get_display_name_v2(tags: Optional[Dict[str, Any]] = None) -> str:
    """
    If the command line argument 'tag' is provided, return its value to be set as the job's display name.
    Empty spaces in the tag will be replaced with hyphens, otherwise AML treats it as multiple statements.
    Otherwise return an empty string.

    :param tags: An optional dictionary of tag names and values to be provided to the job.
    :return: A string either containing the value of tag, or else empty.
    """
    if tags is None:
        return ""
    tag = tags.get("tag", "")
    display_name = tag.replace(" ", "-")
    return display_name


def effective_experiment_name(experiment_name: Optional[str],
                              entry_script: Optional[PathOrString] = None) -> str:
    """Choose the experiment name to use for the run. If provided in the environment variable HIML_EXPERIMENT_NAME,
    then use that. Otherwise, use the argument `experiment_name`, or fall back to the default based on the
    entry point script.

    :param experiment_name: The name of the AzureML experiment in which the run should be submitted.
    :param entry_script: The script that should be run in AzureML.
    :return: The effective experiment name to use, based on the fallback rules above.
    """
    value_from_env = os.environ.get(ENV_EXPERIMENT_NAME, "")
    if value_from_env:
        raw_value = value_from_env
    elif experiment_name:
        raw_value = experiment_name
    elif entry_script is not None:
        raw_value = Path(entry_script).stem
    else:
        raise ValueError("No experiment name provided, and no entry script provided. ")
    cleaned_value = to_azure_friendly_string(raw_value)
    assert cleaned_value is not None, "Expecting an actual string"
    return cleaned_value


def submit_run_v2(workspace: Optional[Workspace],
                  experiment_name: str,
                  environment: EnvironmentV2,
                  input_datasets_v2: Optional[Dict[str, Input]] = None,
                  output_datasets_v2: Optional[Dict[str, Output]] = None,
                  snapshot_root_directory: Optional[Path] = None,
                  entry_script: Optional[PathOrString] = None,
                  script_params: Optional[List[str]] = None,
                  compute_target: Optional[str] = None,
                  tags: Optional[Dict[str, str]] = None,
                  docker_shm_size: str = "",
                  wait_for_completion: bool = False,
                  workspace_config_path: Optional[PathOrString] = None,
                  ml_client: Optional[MLClient] = None,
                  hyperparam_args: Optional[Dict[str, Any]] = None,
                  num_nodes: int = 1,
                  pytorch_processes_per_node: Optional[int] = None,
                  ) -> Job:
    """
    Starts a v2 AML Job on a given workspace by submitting a command

    :param workspace: The AzureML workspace to use.
    :param experiment_name: The name of the experiment that will be used or created. If the experiment name contains
        characters that are not valid in Azure, those will be removed.
    :param environment: An AML v2 Environment object.
    :param input_datasets_v2: An optional dictionary of Inputs to pass in to the command.
    :param output_datasets_v2: An optional dictionary of Outputs to pass in to the command.
    :param snapshot_root_directory: The directory that contains all code that should be packaged and sent to AzureML.
        All Python code that the script uses must be copied over.
    :param entry_script: The script that should be run in AzureML.
    :param script_params: A list of parameter to pass on to the script as it runs in AzureML.
    :param compute_target: Optional name of a compute target in Azure ML to submit the job to. If None, will run
        locally.
    :param tags: A dictionary of string key/value pairs, that will be added as metadata to the run. If set to None,
        a default metadata field will be added that only contains the commandline arguments that started the run.
    :param docker_shm_size: The Docker shared memory size that should be used when creating a new Docker image.
    :param wait_for_completion: If False (the default) return after the run is submitted to AzureML, otherwise wait for
        the completion of this run (if True).
    :param workspace_config_path: If not provided with an AzureML Workspace, then load one given the information in this
        config
    :param ml_client: An Azure MLClient object for interacting with Azure resources.
    :param hyperparam_args: A dictionary of hyperparameter search args to pass into a sweep job.
    :param num_nodes: The number of nodes to use for the job in AzureML. The value must be 1 or greater.
    :param pytorch_processes_per_node: For plain PyTorch multi-GPU processing: The number of processes per node.
        If supplied, it will run a command job with the "pytorch" framework (rather than "Python"), and using "nccl"
        as the communication backend.
    :return: An AzureML Run object.
    """
    if ml_client is None:
        if workspace is not None:
            ml_client = get_ml_client(
                subscription_id=workspace.subscription_id,
                resource_group=workspace.resource_group,
                workspace_name=workspace.name
            )
        elif workspace_config_path is not None:
            ml_client = get_ml_client(workspace_config_path=workspace_config_path)
        else:
            raise ValueError("Either workspace or workspace_config_path must be specified to connect to the Workspace")

    assert compute_target is not None, "No compute_target has been provided"
    assert entry_script is not None, "No entry_script has been provided"
    snapshot_root_directory = snapshot_root_directory or Path.cwd()
    root_dir = Path(snapshot_root_directory)
    entry_script = Path(entry_script).relative_to(root_dir).as_posix()

    script_params = script_params or []
    cmd = " ".join(["python", str(entry_script), *script_params])

    if input_datasets_v2:
        cmd += _generate_input_dataset_command(input_datasets_v2)
    else:
        input_datasets_v2 = {}

    if output_datasets_v2:
        cmd += _generate_output_dataset_command(output_datasets_v2)
    else:
        output_datasets_v2 = {}

    job_to_submit: Union[Command, Sweep]
    display_name = get_display_name_v2(tags)

    # number of nodes and processes per node cannot be less than one
    if num_nodes < 1:
        raise ValueError("num_nodes must be >= 1")
    num_nodes = num_nodes if num_nodes >= 1 else 1
    if pytorch_processes_per_node is not None:
        if pytorch_processes_per_node < 1:
            raise ValueError("pytorch_processes_per_node must be >= 1")

    def create_command_job(cmd: str) -> Command:
        if pytorch_processes_per_node is None:
            if num_nodes > 1:
                distribution: Any = MpiDistribution(process_count_per_instance=1)
            else:
                # An empty dictionary for single node jobs would be in line with the type annotations on the
                # 'command' function, but this is not recognized by the SDK. So we need to pass None instead.
                distribution = None
        else:
            distribution = PyTorchDistribution(process_count_per_instance=pytorch_processes_per_node)
        return command(
            code=str(snapshot_root_directory),
            command=cmd,
            inputs=input_datasets_v2,
            outputs=output_datasets_v2,
            environment=environment.name + "@latest",
            compute=compute_target,
            experiment_name=experiment_name,
            tags=tags or {},
            shm_size=docker_shm_size,
            display_name=display_name,
            instance_count=num_nodes,
            distribution=distribution,  # type: ignore
        )

    if hyperparam_args:
        param_sampling = hyperparam_args[PARAM_SAMPLING_ARG]

        for sample_param, choices in param_sampling.items():
            input_datasets_v2[sample_param] = choices.values[0]
            cmd += f" --{sample_param}=" + "${{inputs." + sample_param + "}}"

        command_job = create_command_job(cmd)

        del hyperparam_args[PARAM_SAMPLING_ARG]
        # override command with parameter expressions
        command_job = command_job(
            **param_sampling,
        )

        job_to_submit = command_job.sweep(
            compute=compute_target,  # AML docs suggest setting this here although already passed to command
            **hyperparam_args
        )

        # AML docs state to reset certain properties here which aren't picked up from the
        # underlying command such as experiment name and max_total_trials
        job_to_submit.experiment_name = experiment_name
        job_to_submit.set_limits(max_total_trials=hyperparam_args.get(MAX_TOTAL_TRIALS_ARG, None))

    else:
        job_to_submit = create_command_job(cmd)

    returned_job = ml_client.jobs.create_or_update(job_to_submit)
    logging.info(f"URL to job: {returned_job.services['Studio'].endpoint}")  # type: ignore
    if wait_for_completion:
        print("Waiting for the completion of the AzureML job.")
        wait_for_job_completion(ml_client, job_name=returned_job.name)
        print("AzureML job completed.")
        # After waiting, ensure that the caller gets the latest version job object
        returned_job = ml_client.jobs.get(returned_job.name)
    return returned_job


def download_job_outputs_logs(ml_client: MLClient,
                              job_name: str,
                              file_to_download_path: str = "",
                              download_dir: Optional[PathOrString] = None) -> None:
    """
    Download output files from an mlflow job. Outputs will be downloaded to a folder named
    `<download_dir>/<job_name>` where download_dir is either provided to this function,
    or is "outputs". If a single file is required, the path to this file within the job can
    be specified with 'file_to_download_path'

    :param ml_client: An MLClient object.
    :param job_name: The name (id) of the job to download output files from.
    :param file_to_download_path: An optional path to a single file/folder to download.
    :param download_dir: An optional folder into which to download the run files.
    """

    download_dir = Path(download_dir) if download_dir else Path("outputs")
    download_dir = download_dir / job_name
    ml_client.jobs.download(job_name, output_name=file_to_download_path, download_path=download_dir)


def submit_run(workspace: Workspace,
               experiment_name: str,
               script_run_config: Union[ScriptRunConfig, HyperDriveConfig],
               tags: Optional[Dict[str, str]] = None,
               wait_for_completion: bool = False,
               wait_for_completion_show_output: bool = False,
               ) -> Run:
    """
    Starts an AzureML run on a given workspace, via the script_run_config.

    :param workspace: The AzureML workspace to use.
    :param experiment_name: The name of the experiment that will be used or created. If the experiment name contains
        characters that are not valid in Azure, those will be removed.
    :param script_run_config: The settings that describe which script should be run.
    :param tags: A dictionary of string key/value pairs, that will be added as metadata to the run. If set to None,
        a default metadata field will be added that only contains the commandline arguments that started the run.
    :param wait_for_completion: If False (the default) return after the run is submitted to AzureML, otherwise wait for
        the completion of this run (if True).
    :param wait_for_completion_show_output: If wait_for_completion is True this parameter indicates whether to show the
        run output on sys.stdout.
    :return: An AzureML Run object.
    """
    cleaned_experiment_name = to_azure_friendly_string(experiment_name)
    experiment = Experiment(workspace=workspace, name=cleaned_experiment_name)
    user_agent.append(SDK_NAME, SDK_VERSION)
    run = experiment.submit(script_run_config)
    if tags is None:
        if hasattr(script_run_config, 'arguments') and \
                script_run_config.arguments is not None:
            # It is probably a ScriptRunConfig
            tags = {"commandline_args": " ".join(script_run_config.arguments)}
        elif hasattr(script_run_config, 'run_config') and \
                hasattr(script_run_config.run_config, 'arguments') and \
                script_run_config.run_config.arguments is not None:
            # It is probably a HyperDriveConfig
            tags = {"commandline_args": " ".join(script_run_config.run_config.arguments)}
    run.set_tags(tags)

    _write_run_recovery_file(run)

    # These need to be 'print' not 'logging.info' so that the calling script sees them outside AzureML
    print("\n==============================================================================")
    print(f"Successfully queued run number {run.number} (ID {run.id}) in experiment {run.experiment.name}")
    print(f"Experiment name and run ID are available in file {RUN_RECOVERY_FILE}")
    print(f"Experiment URL: {run.experiment.get_portal_url()}")
    print(f"Run URL: {run.get_portal_url()}")
    print("==============================================================================\n")
    if wait_for_completion:
        print("Waiting for the completion of the AzureML run.")
        run.wait_for_completion(show_output=wait_for_completion_show_output,
                                wait_post_processing=True,
                                raise_on_error=True)
        if not is_run_and_child_runs_completed(run):
            raise ValueError(f"Run {run.id} in experiment {run.experiment.name} or one of its child "
                             "runs failed.")
        print("AzureML completed.")
    return run


def _str_to_path(s: Optional[PathOrString]) -> Optional[Path]:
    if isinstance(s, str):
        return Path(s)
    return s


def create_v2_inputs(ml_client: MLClient, input_datasets: List[DatasetConfig]) -> Dict[str, Input]:
    """
    Create a dictionary of Azure ML v2 Input objects, required for passing input data in to an AML job

    :param ml_client: An MLClient object.
    :param input_datasets: A list of DatasetConfigs to convert to Inputs.
    :return: A dictionary in the format "input_name": Input.
    """
    inputs: Dict[str, Input] = {}
    for i, input_dataset in enumerate(input_datasets):
        input_name = f"INPUT_{i}"
        version = input_dataset.version or 1
        data_asset: Data = ml_client.data.get(input_dataset.name, version=str(version))
        data_path = data_asset.id or ""
        # Note that there are alternative formats that the input path can take, such as:
        # v1_datastore_path = f"azureml://datastores/{input_dataset.datastore}/paths/<path_to_dataset>"
        # v2_dataset_path = f"azureml:{input_dataset.name}:1"

        inputs[input_name] = Input(  # type: ignore
            type=AssetTypes.URI_FOLDER,
            path=data_path,
            mode=InputOutputModes.MOUNT,
        )
    return inputs


def create_v2_outputs(output_datasets: List[DatasetConfig]) -> Dict[str, Output]:
    """
    Create a dictionary of Azure ML v2 Output objects, required for passing output data in to an AML job

    :param output_datasets: A list of DatasetConfigs to convert to Outputs.
    :return: A dictionary in the format "output_name": Output.
    """
    outputs = {}
    for i, output_dataset in enumerate(output_datasets):
        output_name = f"OUTPUT_{i}"
        v1_datastore_path = f"azureml://datastores/{output_dataset.datastore}/paths/{output_dataset.name}"
        # Note that there are alternative formats that the output path can take, such as:
        # v2_data_asset_path = f"azureml:{output_dataset.name}@latest"
        outputs[output_name] = Output(  # type: ignore
            type=AssetTypes.URI_FOLDER,
            path=v1_datastore_path,
            mode=InputOutputModes.DIRECT,
        )
    return outputs


def submit_to_azure_if_needed(  # type: ignore
        compute_cluster_name: str = "",
        entry_script: Optional[PathOrString] = None,
        aml_workspace: Optional[Workspace] = None,
        workspace_config_file: Optional[PathOrString] = None,
        ml_client: Optional[MLClient] = None,
        snapshot_root_directory: Optional[PathOrString] = None,
        script_params: Optional[List[str]] = None,
        conda_environment_file: Optional[PathOrString] = None,
        aml_environment_name: str = "",
        experiment_name: Optional[str] = None,
        environment_variables: Optional[Dict[str, str]] = None,
        pip_extra_index_url: str = "",
        private_pip_wheel_path: Optional[PathOrString] = None,
        docker_base_image: str = DEFAULT_DOCKER_BASE_IMAGE,
        docker_shm_size: str = DEFAULT_DOCKER_SHM_SIZE,
        ignored_folders: Optional[List[PathOrString]] = None,
        default_datastore: str = "",
        input_datasets: Optional[List[StrOrDatasetConfig]] = None,
        output_datasets: Optional[List[StrOrDatasetConfig]] = None,
        num_nodes: int = 1,
        wait_for_completion: bool = False,
        wait_for_completion_show_output: bool = False,
        max_run_duration: str = "",
        submit_to_azureml: Optional[bool] = None,
        tags: Optional[Dict[str, str]] = None,
        after_submission: Optional[Union[Callable[[Run], None], Callable[[Job, MLClient], None]]] = None,
        hyperdrive_config: Optional[HyperDriveConfig] = None,
        hyperparam_args: Optional[Dict[str, Any]] = None,
        strictly_aml_v1: bool = False,
        pytorch_processes_per_node_v2: Optional[int] = None,
) -> AzureRunInfo:  # pragma: no cover
    """
    Submit a folder to Azure, if needed and run it.
    Use the commandline flag --azureml to submit to AzureML, and leave it out to run locally.

    :param after_submission: A function that will be called directly after submitting the job to AzureML.
        Use this to, for example, add additional tags or print information about the run.
        When using AzureML SDK V1, the only argument to this function is the Run object that was just submitted.
        When using AzureML SDK V2, the arguments are (Job, MLClient).
    :param tags: A dictionary of string key/value pairs, that will be added as metadata to the run. If set to None,
        a default metadata field will be added that only contains the commandline arguments that started the run.
    :param aml_environment_name: The name of an AzureML environment that should be used to submit the script. If not
        provided, an environment will be created from the arguments to this function.
    :param max_run_duration: The maximum runtime that is allowed for this job in AzureML. This is given as a
        floating point number with a string suffix s, m, h, d for seconds, minutes, hours, day. Examples: '3.5h', '2d'
    :param experiment_name: The name of the AzureML experiment in which the run should be submitted. If omitted,
        this is created based on the name of the current script.
    :param entry_script: The script that should be run in AzureML
    :param compute_cluster_name: The name of the AzureML cluster that should run the job. This can be a cluster with
        CPU or GPU machines.
    :param conda_environment_file: The conda configuration file that describes which packages are necessary for your
        script to run.
    :param aml_workspace: There are two optional parameters used to glean an existing AzureML Workspace. The simplest is
        to pass it in as a parameter.
    :param workspace_config_file: The 2nd option is to specify the path to the config.json file downloaded from the
        Azure portal from which we can retrieve the existing Workspace.
    :param ml_client: An Azure MLClient object for interacting with Azure resources.
    :param snapshot_root_directory: The directory that contains all code that should be packaged and sent to AzureML.
        All Python code that the script uses must be copied over.
    :param ignored_folders: A list of folders to exclude from the snapshot when copying it to AzureML.
    :param script_params: A list of parameter to pass on to the script as it runs in AzureML. If empty (or None, the
        default) these will be copied over from sys.argv, omitting the --azureml flag.
    :param environment_variables: The environment variables that should be set when running in AzureML.
    :param docker_base_image: The Docker base image that should be used when creating a new Docker image.
        The list of available images can be found here: https://github.com/Azure/AzureML-Containers
        The default image is `mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.2-cudnn8-ubuntu18.04`
    :param docker_shm_size: The Docker shared memory size that should be used when creating a new Docker image.
        Default value is '100g'.
    :param pip_extra_index_url: If provided, use this PIP package index to find additional packages when building
        the Docker image.
    :param private_pip_wheel_path: If provided, add this wheel as a private package to the AzureML workspace.
    :param default_datastore: The data store in your AzureML workspace, that points to your training data in blob
        storage. This is described in more detail in the README.
    :param input_datasets: The script will consume all data in folder in blob storage as the input. The folder must
        exist in blob storage, in the location that you gave when creating the datastore. Once the script has run, it
        will also register the data in this folder as an AzureML dataset.
    :param output_datasets: The script will create a temporary folder when running in AzureML, and while the job writes
        data to that folder, upload it to blob storage, in the data store.
    :param num_nodes: The number of nodes to use in distributed training on AzureML. When using a value > 1, multiple
        nodes in AzureML will be started. If `pytorch_processes_per_node_v2=None`, the job will be submitted
        as a multi-node MPI job, with 1 process per node. This is suitable for PyTorch Lightning jobs.
        If `pytorch_processes_per_node_v2` is not None,
        a job with framework "PyTorch" and communication backend "nccl" will be started.
        `pytorch_processes_per_node_v2` will guide the number of processes per node. This is suitable for plain PyTorch
        training jobs without the use of frameworks like PyTorch Lightning.
    :param wait_for_completion: If False (the default) return after the run is submitted to AzureML, otherwise wait for
        the completion of this run (if True).
    :param wait_for_completion_show_output: If wait_for_completion is True this parameter indicates whether to show the
        run output on sys.stdout.
    :param submit_to_azureml: If True, the codepath to create an AzureML run will be executed. If False, the codepath
        for local execution (i.e., return immediately) will be executed. If not provided (None), submission to AzureML
        will be triggered if the commandline flag '--azureml' is present in sys.argv
    :param hyperdrive_config: A configuration object for Hyperdrive (hyperparameter search).
    :param strictly_aml_v1: If True, use Azure ML SDK v1. Otherwise, attempt to use Azure ML SDK v2.
    :param pytorch_processes_per_node_v2: For plain PyTorch multi-GPU processing: The number of processes per node. This
        is only supported with AML SDK v2, and ignored in v1. If supplied, the job will be submitted as using the
        "pytorch" framework (rather than "Python"), and using "nccl" as the communication backend.
    :return: If the script is submitted to AzureML then we terminate python as the script should be executed in AzureML,
        otherwise we return a AzureRunInfo object.
    """
    health_azure_package_setup()
    workspace_config_path = _str_to_path(workspace_config_file)
    snapshot_root_directory = _str_to_path(snapshot_root_directory)
    cleaned_input_datasets = _replace_string_datasets(input_datasets or [],
                                                      default_datastore_name=default_datastore)
    cleaned_output_datasets = _replace_string_datasets(output_datasets or [],
                                                       default_datastore_name=default_datastore)

    # The present function will most likely be called from the script once it is running in AzureML.
    # The '--azureml' flag will not be present anymore, but we don't want to rely on that. From Run.get_context we
    # can infer if the present code is running in AzureML.
    in_azure = is_running_in_azure_ml(RUN_CONTEXT)
    if in_azure:
        if strictly_aml_v1:
            return _generate_azure_datasets(cleaned_input_datasets, cleaned_output_datasets)
        else:
            return _generate_v2_azure_datasets(cleaned_input_datasets, cleaned_output_datasets)
    # This codepath is reached when executing outside AzureML. Here we first check if a script submission to AzureML
    # is necessary. If not, return to the caller for local execution.
    if submit_to_azureml is None:
        submit_to_azureml = AZUREML_FLAG in sys.argv[1:]
    if not submit_to_azureml:
        # Set the environment variables for local execution.
        environment_variables = {
            **DEFAULT_ENVIRONMENT_VARIABLES,
            **(environment_variables or {})
        }

        for k, v in environment_variables.items():
            os.environ[k] = v

        output_folder = Path.cwd() / OUTPUT_FOLDER
        output_folder.mkdir(exist_ok=True)

        logs_folder = Path.cwd() / LOGS_FOLDER
        logs_folder.mkdir(exist_ok=True)

        mounted_input_datasets, mount_contexts = setup_local_datasets(cleaned_input_datasets,
                                                                      strictly_aml_v1,
                                                                      aml_workspace=aml_workspace,
                                                                      ml_client=ml_client,
                                                                      workspace_config_path=workspace_config_path)

        return AzureRunInfo(
            input_datasets=mounted_input_datasets,
            output_datasets=[d.local_folder for d in cleaned_output_datasets],
            mount_contexts=mount_contexts,
            run=None,
            is_running_in_azure_ml=False,
            output_folder=output_folder,
            logs_folder=logs_folder
        )

    if snapshot_root_directory is None:
        print(f"No snapshot root directory given. Uploading all files in the current directory {Path.cwd()}")
        snapshot_root_directory = Path.cwd()

    workspace = get_workspace(aml_workspace, workspace_config_path)
    print(f"Loaded AzureML workspace {workspace.name}")

    if conda_environment_file is None:
        conda_environment_file = find_file_in_parent_to_pythonpath(CONDA_ENVIRONMENT_FILE)
        if conda_environment_file is None:
            raise ValueError(f"No conda environment file {CONDA_ENVIRONMENT_FILE} found in {Path.cwd()} "
                             "or any parent directory.")
        print(f"Using the Conda environment from this file: {conda_environment_file}")
    conda_environment_file = _str_to_path(conda_environment_file)

    amlignore_path = snapshot_root_directory / AML_IGNORE_FILE
    lines_to_append = [str(path) for path in (ignored_folders or [])]
    script_params = _get_script_params(script_params)

    with append_to_amlignore(amlignore=amlignore_path, lines_to_append=lines_to_append):
        if strictly_aml_v1:
            run_config = create_run_configuration(
                workspace=workspace,
                compute_cluster_name=compute_cluster_name,
                aml_environment_name=aml_environment_name,
                conda_environment_file=conda_environment_file,
                environment_variables=environment_variables,
                pip_extra_index_url=pip_extra_index_url,
                private_pip_wheel_path=_str_to_path(private_pip_wheel_path),
                docker_base_image=docker_base_image,
                docker_shm_size=docker_shm_size,
                num_nodes=num_nodes,
                max_run_duration=max_run_duration,
                input_datasets=cleaned_input_datasets,
                output_datasets=cleaned_output_datasets,
            )

            script_run_config = create_script_run(
                script_params=script_params,
                snapshot_root_directory=snapshot_root_directory,
                entry_script=entry_script,
            )
            script_run_config.run_config = run_config

            if hyperdrive_config:
                config_to_submit: Union[ScriptRunConfig, HyperDriveConfig] = hyperdrive_config
                config_to_submit._run_config = script_run_config
            else:
                config_to_submit = script_run_config

            run = submit_run(workspace=workspace,
                             experiment_name=effective_experiment_name(experiment_name, script_run_config.script),
                             script_run_config=config_to_submit,
                             tags=tags,
                             wait_for_completion=wait_for_completion,
                             wait_for_completion_show_output=wait_for_completion_show_output)
            if after_submission is not None:
                after_submission(run)  # type: ignore
        else:

            if conda_environment_file is None:
                raise ValueError("Argument 'conda_environment_file' must be specified when using AzureML v2")
            environment = create_python_environment_v2(
                conda_environment_file=conda_environment_file,
                docker_base_image=docker_base_image
            )
            if entry_script is None:
                entry_script = Path(sys.argv[0])

            ml_client = get_ml_client(ml_client=ml_client, aml_workspace=workspace)
            registered_env = register_environment_v2(environment, ml_client)
            input_datasets_v2 = create_v2_inputs(ml_client, cleaned_input_datasets)
            output_datasets_v2 = create_v2_outputs(cleaned_output_datasets)

            job = submit_run_v2(workspace=workspace,
                                input_datasets_v2=input_datasets_v2,
                                output_datasets_v2=output_datasets_v2,
                                experiment_name=effective_experiment_name(experiment_name, entry_script),
                                environment=registered_env,
                                snapshot_root_directory=snapshot_root_directory,
                                entry_script=entry_script,
                                script_params=script_params,
                                compute_target=compute_cluster_name,
                                tags=tags,
                                docker_shm_size=docker_shm_size,
                                wait_for_completion=wait_for_completion,
                                hyperparam_args=hyperparam_args,
                                num_nodes=num_nodes,
                                pytorch_processes_per_node=pytorch_processes_per_node_v2,
                                )
            if after_submission is not None:
                after_submission(job, ml_client)  # type: ignore

    exit(0)


def _write_run_recovery_file(run: Run) -> None:
    """
    Write the run recovery file

    :param run: The AzureML run to save as a recovery checkpoint.
    """
    recovery_id = create_run_recovery_id(run)
    recovery_file = Path(RUN_RECOVERY_FILE)
    if recovery_file.exists():
        recovery_file.unlink()
    recovery_file.write_text(recovery_id)


def convert_himl_to_azureml_datasets(
    cleaned_input_datasets: List[DatasetConfig],
    cleaned_output_datasets: List[DatasetConfig],
    workspace: Workspace,
    strictly_aml_v1: bool
) -> Tuple[Dict[str, DatasetConsumptionConfig], Dict[str, OutputFileDatasetConfig]]:
    """
    Convert the cleaned input and output datasets into dictionaries of DatasetConsumptionConfigs for use in AzureML.

    :param cleaned_input_datasets: The list of input DatasetConfigs
    :param cleaned_output_datasets: The list of output DatasetConfigs
    :param workspace: The AzureML workspace
    :param strictly_aml_v1: If True, use Azure ML SDK v1 to attempt to find or create and reigster the dataset.
        Otherwise, attempt to use Azure ML SDK v2.
    :return: The input and output dictionaries of DatasetConsumptionConfigs.
    """
    inputs = {}
    for index, input_dataset in enumerate(cleaned_input_datasets):
        consumption = input_dataset.to_input_dataset(index, workspace, strictly_aml_v1=strictly_aml_v1)
        if isinstance(consumption, DatasetConsumptionConfig):
            data_name = consumption.name  # type: ignore
            if data_name in inputs:
                raise ValueError(f"There is already an input dataset with name '{data_name}' set up?")
            inputs[data_name] = consumption
        elif isinstance(consumption, Input):
            inputs[input_dataset.name] = consumption
        else:
            raise ValueError(f"Unrecognised input data type: {type(consumption)}")
    outputs = {}
    for index, output_dataset in enumerate(cleaned_output_datasets):
        out = output_dataset.to_output_dataset(workspace=workspace, dataset_index=index)
        if out.name in outputs:
            raise ValueError(f"There is already an output dataset with name '{out.name}' set up?")
        outputs[out.name] = out
    return inputs, outputs


def _get_script_params(script_params: Optional[List[str]] = None) -> List[str]:
    """
    If script parameters are given then return them, otherwise derive them from sys.argv

    :param script_params: The optional script parameters
    :return: The given script parameters or ones derived from sys.argv
    """
    if script_params:
        return script_params
    return [p for p in sys.argv[1:] if p != AZUREML_FLAG]


def _generate_azure_datasets(
        cleaned_input_datasets: List[DatasetConfig],
        cleaned_output_datasets: List[DatasetConfig]) -> AzureRunInfo:
    """
    Generate returned datasets when running in AzureML.

    :param cleaned_input_datasets: The list of input dataset configs
    :param cleaned_output_datasets: The list of output dataset configs
    :return: The AzureRunInfo containing the AzureML input and output dataset lists etc.
    """
    if is_amulet_job():
        input_data_mount_folder = Path(os.environ[ENV_AMLT_DATAREFERENCE_DATA])
        logging.info(f"Path to mounted data: {ENV_AMLT_DATAREFERENCE_DATA}: {str(input_data_mount_folder)}")
        returned_input_datasets = [input_data_mount_folder / input_dataset.name for input_dataset in
                                   cleaned_input_datasets]

        output_data_mount_folder = Path(os.environ[ENV_AMLT_DATAREFERENCE_OUTPUT])
        logging.info(f"Path to output datasets: {output_data_mount_folder}")
        returned_output_datasets = [output_data_mount_folder / output_dataset.name for output_dataset in
                                    cleaned_output_datasets]
        logging.info(f"Stitched returned input datasets: {returned_input_datasets}")
        logging.info(f"Stitched returned output datasets: {returned_output_datasets}")
    else:
        returned_input_datasets = [Path(RUN_CONTEXT.input_datasets[_input_dataset_key(index)])
                                   for index in range(len(cleaned_input_datasets))]
        returned_output_datasets = [Path(RUN_CONTEXT.output_datasets[_output_dataset_key(index)])
                                    for index in range(len(cleaned_output_datasets))]
    return AzureRunInfo(
        input_datasets=returned_input_datasets,  # type: ignore
        output_datasets=returned_output_datasets,  # type: ignore
        mount_contexts=[],
        run=RUN_CONTEXT,
        is_running_in_azure_ml=True,
        output_folder=Path.cwd() / OUTPUT_FOLDER,
        logs_folder=Path.cwd() / LOGS_FOLDER)


def _get_dataset_names_from_string(sys_arg: str, pattern: str) -> Path:
    dataset_string = re.split(pattern, sys_arg)[-1]
    dataset_path = Path(dataset_string)
    return dataset_path


def _extract_v2_inputs_outputs_from_args() -> Tuple[List[Path], List[Path]]:
    """
    Extract all command line arguments of the format INPUT_i=path_to_input or OUTPUT_i=path_to_output (where i is any
    integer) and return a list of the Paths for each.

    :return: A list of Input paths and a list of Output paths
    """
    returned_input_datasets: List[Path] = []
    returned_output_datasets: List[Path] = []

    for sys_arg in sys.argv:
        if re.match(V2_INPUT_DATASET_PATTERN, sys_arg):
            returned_input_datasets += [_get_dataset_names_from_string(sys_arg, V2_INPUT_DATASET_PATTERN)]
        if re.match(V2_OUTPUT_DATASET_PATTERN, sys_arg):
            returned_output_datasets += [_get_dataset_names_from_string(sys_arg, V2_OUTPUT_DATASET_PATTERN)]
    return returned_input_datasets, returned_output_datasets


def _generate_v2_azure_datasets(cleaned_input_datasets: List[DatasetConfig],
                                cleaned_output_datasets: List[DatasetConfig]) -> AzureRunInfo:
    """
    Generate returned datasets when running in AzureML. Assumes this is v2 Job, so we need to get
    the input datasets from the command line args

    :param cleaned_input_datasets: The list of input dataset configs
    :param cleaned_output_datasets: The list of output dataset configs
    :return: The AzureRunInfo containing the AzureML input and output dataset lists etc.
    """
    returned_input_datasets, returned_output_datasets = _extract_v2_inputs_outputs_from_args()

    return AzureRunInfo(
        input_datasets=returned_input_datasets,  # type: ignore
        output_datasets=returned_output_datasets,  # type: ignore
        mount_contexts=[],
        run=RUN_CONTEXT,
        is_running_in_azure_ml=True,
        output_folder=Path.cwd() / OUTPUT_FOLDER,
        logs_folder=Path.cwd() / LOGS_FOLDER)


@contextmanager
def append_to_amlignore(lines_to_append: List[str], amlignore: Optional[Path] = None) -> Generator:
    """
    Context manager that appends lines to the .amlignore file, and reverts to the previous contents after leaving
    the context.
    If the file does not exist yet, it will be created, the contents written, and deleted when leaving the context.

    :param lines_to_append: The text lines that should be added at the enund of the .amlignore file
    :param amlignore: The path of the .amlignore file that should be modified. If not given, the function
        looks for a file in the current working directory.
    """
    if amlignore is None:
        amlignore = Path.cwd() / AML_IGNORE_FILE
    amlignore_exists_already = amlignore.exists()
    old_contents = amlignore.read_text() if amlignore_exists_already else ""
    new_lines = old_contents.splitlines() + lines_to_append
    new_text = "\n".join(new_lines)
    if new_text:
        amlignore.write_text(new_text)
    yield
    if amlignore_exists_already:
        amlignore.write_text(old_contents)
    elif new_text:
        amlignore.unlink()


def main() -> None:
    """
    Handle submit_to_azure if called from the command line.
    """
    parser = ArgumentParser()
    parser.add_argument("-p", "--workspace_config_file", type=str, required=False, help="AzureML workspace config file")
    parser.add_argument("-c", "--compute_cluster_name", type=str, required=True, help="AzureML cluster name")
    parser.add_argument("-y", "--snapshot_root_directory", type=str, required=True,
                        help="Root of snapshot to upload to AzureML")
    parser.add_argument("-t", "--entry_script", type=str, required=True,
                        help="The script to run in AzureML")
    parser.add_argument("-d", "--conda_environment_file", type=str, required=True, help="The environment to use")

    args = parser.parse_args()

    submit_to_azure_if_needed(
        workspace_config_file=Path(args.workspace_config_file),
        compute_cluster_name=args.compute_cluster_name,
        snapshot_root_directory=Path(args.snapshot_root_directory),
        entry_script=Path(args.entry_script),
        conda_environment_file=Path(args.conda_environment_file),
    )


if __name__ == "__main__":
    main()  # pragma: no cover
