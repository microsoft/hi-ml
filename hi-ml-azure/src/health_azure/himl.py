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
import sys
import warnings
from argparse import ArgumentParser
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union

from azureml._base_sdk_common import user_agent
from azureml.core import ComputeTarget, Environment, Experiment, Run, RunConfiguration, ScriptRunConfig, Workspace
from azureml.core.runconfig import DockerConfiguration, MpiConfiguration
from azureml.data import OutputFileDatasetConfig
from azureml.data.dataset_consumption_config import DatasetConsumptionConfig
from azureml.train.hyperdrive import HyperDriveConfig, GridParameterSampling, PrimaryMetricGoal, choice
from azureml.dataprep.fuse.daemon import MountContext

from health_azure.utils import (create_python_environment, create_run_recovery_id, find_file_in_parent_to_pythonpath,
                                is_run_and_child_runs_completed, is_running_in_azure_ml, register_environment,
                                run_duration_string_to_seconds, to_azure_friendly_string, RUN_CONTEXT, get_workspace,
                                PathOrString, DEFAULT_ENVIRONMENT_VARIABLES)
from health_azure.datasets import (DatasetConfig, StrOrDatasetConfig, _input_dataset_key, _output_dataset_key,
                                   _replace_string_datasets, setup_local_datasets)

logger = logging.getLogger('health_azure')
logger.setLevel(logging.DEBUG)

AML_IGNORE_FILE = ".amlignore"
AZUREML_COMMANDLINE_FLAG = "--azureml"
CONDA_ENVIRONMENT_FILE = "environment.yml"
LOGS_FOLDER = "logs"
OUTPUT_FOLDER = "outputs"
RUN_RECOVERY_FILE = "most_recent_run.txt"
SDK_NAME = "innereye"
SDK_VERSION = "2.0"


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
                                                           workspace=workspace)
        run_config.data = inputs
        run_config.output_data = outputs

    return run_config


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
    logging.info(f"Creating a HyperDriveConfig. Please note that this expects to find the specified "
                 f"metric '{metric_name}' logged to AzureML from your training script (for example, using the "
                 f"AzureMLLogger with Pytorch Lightning)")
    parameter_dict = {
        cross_val_index_arg_name: choice(list(range(num_splits))),
    }
    return HyperDriveConfig(
        run_config=ScriptRunConfig(""),
        hyperparameter_sampling=GridParameterSampling(parameter_dict),
        primary_metric_name=metric_name,
        primary_metric_goal=PrimaryMetricGoal.MINIMIZE,
        max_total_runs=num_splits
    )


def create_script_run(snapshot_root_directory: Optional[Path] = None,
                      entry_script: Optional[PathOrString] = None,
                      script_params: Optional[List[str]] = None) -> ScriptRunConfig:
    """
    Creates an AzureML ScriptRunConfig object, that holds the information about the snapshot, the entry script, and
    its arguments.

    :param entry_script: The script that should be run in AzureML.
    :param snapshot_root_directory: The directory that contains all code that should be packaged and sent to AzureML.
        All Python code that the script uses must be copied over.
    :param script_params: A list of parameter to pass on to the script as it runs in AzureML. If empty (or None, the
        default) these will be copied over from sys.argv, omitting the --azureml flag.
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
    script_params = _get_script_params(script_params)
    print(f"This command will be run in AzureML: {entry_script_relative} {' '.join(script_params)}")
    return ScriptRunConfig(
        source_directory=str(snapshot_root_directory),
        script=entry_script_relative,
        arguments=script_params)


def submit_run(workspace: Workspace,
               experiment_name: str,
               script_run_config: Union[ScriptRunConfig, HyperDriveConfig],
               tags: Optional[Dict[str, str]] = None,
               wait_for_completion: bool = False,
               wait_for_completion_show_output: bool = False, ) -> Run:
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


def submit_to_azure_if_needed(  # type: ignore
        compute_cluster_name: str = "",
        entry_script: Optional[PathOrString] = None,
        aml_workspace: Optional[Workspace] = None,
        workspace_config_file: Optional[PathOrString] = None,
        snapshot_root_directory: Optional[PathOrString] = None,
        script_params: Optional[List[str]] = None,
        conda_environment_file: Optional[PathOrString] = None,
        aml_environment_name: str = "",
        experiment_name: Optional[str] = None,
        environment_variables: Optional[Dict[str, str]] = None,
        pip_extra_index_url: str = "",
        private_pip_wheel_path: Optional[PathOrString] = None,
        docker_base_image: str = "",
        docker_shm_size: str = "",
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
        after_submission: Optional[Callable[[Run], None]] = None,
        hyperdrive_config: Optional[HyperDriveConfig] = None,
        create_output_folders: bool = True,
) -> AzureRunInfo:  # pragma: no cover
    """
    Submit a folder to Azure, if needed and run it.
    Use the commandline flag --azureml to submit to AzureML, and leave it out to run locally.

    :param after_submission: A function that will be called directly after submitting the job to AzureML. The only
        argument to this function is the run that was just submitted. Use this to, for example, add additional tags
        or print information about the run.
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
    :param snapshot_root_directory: The directory that contains all code that should be packaged and sent to AzureML.
        All Python code that the script uses must be copied over.
    :param ignored_folders: A list of folders to exclude from the snapshot when copying it to AzureML.
    :param script_params: A list of parameter to pass on to the script as it runs in AzureML. If empty (or None, the
        default) these will be copied over from sys.argv, omitting the --azureml flag.
    :param environment_variables: The environment variables that should be set when running in AzureML.
    :param docker_base_image: The Docker base image that should be used when creating a new Docker image.
    :param docker_shm_size: The Docker shared memory size that should be used when creating a new Docker image.
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
    :param num_nodes: The number of nodes to use in distributed training on AzureML.
    :param wait_for_completion: If False (the default) return after the run is submitted to AzureML, otherwise wait for
        the completion of this run (if True).
    :param wait_for_completion_show_output: If wait_for_completion is True this parameter indicates whether to show the
        run output on sys.stdout.
    :param submit_to_azureml: If True, the codepath to create an AzureML run will be executed. If False, the codepath
        for local execution (i.e., return immediately) will be executed. If not provided (None), submission to AzureML
        will be triggered if the commandline flag '--azureml' is present in sys.argv
    :param hyperdrive_config: A configuration object for Hyperdrive (hyperparameter search).
    :param create_output_folders: If True (default), create folders "outputs" and "logs" in the current working folder.
    :return: If the script is submitted to AzureML then we terminate python as the script should be executed in AzureML,
        otherwise we return a AzureRunInfo object.
    """
    _package_setup()
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
        return _generate_azure_datasets(cleaned_input_datasets, cleaned_output_datasets)

    # This codepath is reached when executing outside AzureML. Here we first check if a script submission to AzureML
    # is necessary. If not, return to the caller for local execution.
    if submit_to_azureml is None:
        submit_to_azureml = AZUREML_COMMANDLINE_FLAG in sys.argv[1:]
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

        mounted_input_datasets, mount_contexts = setup_local_datasets(aml_workspace,
                                                                      workspace_config_path,
                                                                      cleaned_input_datasets)

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
        print(f"Using the Conda environment from this file: {conda_environment_file}")
    conda_environment_file = _str_to_path(conda_environment_file)

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
        output_datasets=cleaned_output_datasets
    )
    script_run_config = create_script_run(snapshot_root_directory=snapshot_root_directory,
                                          entry_script=entry_script,
                                          script_params=script_params)
    script_run_config.run_config = run_config

    if hyperdrive_config:
        config_to_submit: Union[ScriptRunConfig, HyperDriveConfig] = hyperdrive_config
        config_to_submit._run_config = script_run_config
    else:
        config_to_submit = script_run_config

    effective_experiment_name = experiment_name or Path(script_run_config.script).stem

    amlignore_path = snapshot_root_directory / AML_IGNORE_FILE
    lines_to_append = [str(path) for path in (ignored_folders or [])]
    with append_to_amlignore(
            amlignore=amlignore_path,
            lines_to_append=lines_to_append):
        run = submit_run(workspace=workspace,
                         experiment_name=effective_experiment_name,
                         script_run_config=config_to_submit,
                         tags=tags,
                         wait_for_completion=wait_for_completion,
                         wait_for_completion_show_output=wait_for_completion_show_output)

    if after_submission is not None:
        after_submission(run)
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
        workspace: Workspace) -> Tuple[Dict[str, DatasetConsumptionConfig], Dict[str, OutputFileDatasetConfig]]:
    """
    Convert the cleaned input and output datasets into dictionaries of DatasetConsumptionConfigs for use in AzureML.

    :param cleaned_input_datasets: The list of input DatasetConfigs
    :param cleaned_output_datasets: The list of output DatasetConfigs
    :param workspace: The AzureML workspace
    :return: The input and output dictionaries of DatasetConsumptionConfigs.
    """
    inputs = {}
    for index, d in enumerate(cleaned_input_datasets):
        consumption = d.to_input_dataset(workspace=workspace, dataset_index=index)
        if consumption.name in inputs:
            raise ValueError(f"There is already an input dataset with name '{consumption.name}' set up?")
        inputs[consumption.name] = consumption
    outputs = {}
    for index, d in enumerate(cleaned_output_datasets):
        out = d.to_output_dataset(workspace=workspace, dataset_index=index)
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
    return [p for p in sys.argv[1:] if p != AZUREML_COMMANDLINE_FLAG]


def _generate_azure_datasets(
        cleaned_input_datasets: List[DatasetConfig],
        cleaned_output_datasets: List[DatasetConfig]) -> AzureRunInfo:
    """
    Generate returned datasets when running in AzureML.

    :param cleaned_input_datasets: The list of input dataset configs
    :param cleaned_output_datasets: The list of output dataset configs
    :return: The AzureRunInfo containing the AzureML input and output dataset lists etc.
    """
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


def _package_setup() -> None:
    """
    Set up the Python packages where needed. In particular, reduce the logging level for some of the used
    libraries, which are particularly talkative in DEBUG mode. Usually when running in DEBUG mode, we want
    diagnostics about the model building itself, but not for the underlying libraries.
    It also adds workarounds for known issues in some packages.
    """
    # The adal package creates a logging.info line each time it gets an authentication token, avoid that.
    logging.getLogger('adal-python').setLevel(logging.WARNING)
    # Azure core prints full HTTP requests even in INFO mode
    logging.getLogger('azure').setLevel(logging.WARNING)
    # PyJWT prints out warnings that are beyond our control
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="jwt")
    # Urllib3 prints out connection information for each call to write metrics, etc
    logging.getLogger('urllib3').setLevel(logging.INFO)
    logging.getLogger('msrest').setLevel(logging.INFO)
    # AzureML prints too many details about logging metrics
    logging.getLogger('azureml').setLevel(logging.INFO)


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
