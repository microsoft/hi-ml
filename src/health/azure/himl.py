#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
Wrapper functions for running local Python scripts on Azure ML.

See examples/elevate_this.py for a very simple 'hello world' example of use.
"""

import logging
import sys
import warnings
from argparse import ArgumentParser
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union

from azureml.core import Environment, Experiment, Run, RunConfiguration, ScriptRunConfig, Workspace
from azureml.core.runconfig import DockerConfiguration, MpiConfiguration
from azureml.data import OutputFileDatasetConfig
from azureml.data.dataset_consumption_config import DatasetConsumptionConfig
from azureml.train.hyperdrive import HyperDriveConfig

from health.azure.azure_util import (create_python_environment, create_run_recovery_id,
                                     get_authentication, is_run_and_child_runs_completed, register_environment,
                                     run_duration_string_to_seconds,
                                     to_azure_friendly_string)
from health.azure.datasets import (DatasetConfig, StrOrDatasetConfig, _input_dataset_key, _output_dataset_key,
                                   _replace_string_datasets)

logger = logging.getLogger('health.azure')
logger.setLevel(logging.DEBUG)

RUN_RECOVERY_FILE = "most_recent_run.txt"
WORKSPACE_CONFIG_JSON = "config.json"
AZUREML_COMMANDLINE_FLAG = "--azureml"
RUN_CONTEXT = Run.get_context()
OUTPUT_FOLDER = "outputs"
LOGS_FOLDER = "logs"
AML_IGNORE_FILE = ".amlignore"

PathOrString = Union[Path, str]


@dataclass
class AzureRunInformation:
    input_datasets: List[Optional[Path]]
    output_datasets: List[Optional[Path]]
    run: Optional[Run]
    # If True, the present code is running inside of AzureML.
    is_running_in_azure: bool
    # In Azure, this would be the "outputs" folder. In local runs: "." or create a timestamped folder.
    # The folder that we create here must be added to .amlignore
    output_folder: Path
    logs_folder: Path


def is_running_in_azure(aml_run: Run = RUN_CONTEXT) -> bool:
    """
    Returns True if the given run is inside of an AzureML machine, or False if it is a machine outside AzureML.
    When called without arguments, this functions returns True if the present code is running in AzureML.
    :param aml_run: The run to check. If omitted, use the default run in RUN_CONTEXT
    :return: True if the given run is inside of an AzureML machine, or False if it is a machine outside AzureML.
    """
    return hasattr(aml_run, 'experiment')


def get_or_create_environment(workspace: Workspace,
                              aml_environment_name: str,
                              conda_environment_file: Optional[Path],
                              environment_variables: Optional[Dict[str, str]],
                              pip_extra_index_url: str,
                              docker_base_image: str,
                              ) -> Environment:
    """
    Gets an existing AzureML environment from the workspace (choosing by name), or get one based on the contents
    of a Conda environment file, environment variables, pip and docker settings. Either one of the arguments
    `aml_environment` and `conda_environment_file` must be provided.
    :param workspace: The AzureML workspace to work in.
    :param aml_environment_name: The name of an existing AzureML environment that should be read. If this is empty, the
    environment is created based on conda_environment_file.
    :param conda_environment_file: The Conda environment.yml file that should be used for environment creation. If this
    is empty, an existing environment is retrieved via the name given in aml_environment.
    :param environment_variables: A dictionary with environment variables that should used in the AzureML environment.
    This is only used if conda_environment_file is given.
    :param pip_extra_index_url: The value to use for pip's --extra-index-url argument, to read additional packages.
    :param docker_base_image: The Docker base image to use. If not given, docker will not be used. Use the fully
    qualified image locator, e.g. "mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.2-cudnn8-ubuntu18.04", without the
    trailing ":<version number>"
    :return: An AzureML Environment object.
    """
    if aml_environment_name:
        # TODO: Split off version
        return Environment.get(workspace, aml_environment_name)
    elif conda_environment_file:
        environment = create_python_environment(workspace=workspace,
                                                conda_environment_file=conda_environment_file,
                                                pip_extra_index_url=pip_extra_index_url,
                                                docker_base_image=docker_base_image,
                                                environment_variables=environment_variables)
        if docker_base_image:
            return register_environment(workspace, environment)
        return environment
    else:
        raise ValueError("One of the two arguments 'aml_environment' or 'conda_environment_file' must be given.")


def create_run_configuration(workspace: Workspace,
                             compute_cluster_name: str,
                             conda_environment_file: Optional[Path] = None,
                             aml_environment_name: str = "",
                             environment_variables: Optional[Dict[str, str]] = None,
                             pip_extra_index_url: str = "",
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
    :param conda_environment_file: The file that contains the Conda environment definition.
    :param input_datasets: The script will consume all data in folder in blob storage as the input. The folder must
    exist in blob storage, in the location that you gave when creating the datastore. Once the script has run, it will
    also register the data in this folder as an AzureML dataset.
    :param output_datasets: The script will create a temporary folder when running in AzureML, and while the job writes
    data to that folder, upload it to blob storage, in the data store.

    :param num_nodes: The number of nodes to use in distributed training on AzureML.
    :return:
    """
    existing_compute_clusters = workspace.compute_targets
    if compute_cluster_name not in existing_compute_clusters:
        raise ValueError(f"Could not find the compute target {compute_cluster_name} in the AzureML workspace. ",
                         f"Existing clusters: {list(existing_compute_clusters.keys())}")

    run_config = RunConfiguration()
    if docker_shm_size and docker_base_image:
        run_config.docker = DockerConfiguration(use_docker=True, shm_size=docker_shm_size)
    # elif docker_shm_size or docker_base_image:
    #     raise ValueError("To enable docker, you need to provide both arguments 'docker_shm_size' and "
    #                      "'docker_base_image'")
    else:
        if docker_base_image and docker_shm_size:
            run_config.docker = DockerConfiguration(use_docker=False, shm_size=docker_shm_size)
        else:
            run_config.docker.use_docker = False
        # docker_base_image = ""
    run_config.environment = get_or_create_environment(workspace=workspace,
                                                       aml_environment_name=aml_environment_name,
                                                       conda_environment_file=conda_environment_file,
                                                       pip_extra_index_url=pip_extra_index_url,
                                                       environment_variables=environment_variables,
                                                       docker_base_image=docker_base_image,
                                                       register=run_config.docker.use_docker)
    run_config.target = compute_cluster_name
    if max_run_duration:
        run_config.max_run_duration_seconds = run_duration_string_to_seconds(max_run_duration)
    if num_nodes > 1:
        distributed_job_config = MpiConfiguration(node_count=num_nodes)
        run_config.mpi = distributed_job_config
        run_config.framework = "Python"
        run_config.communicator = "IntelMpi"
        run_config.node_count = distributed_job_config.node_count

    inputs, outputs = convert_himl_to_azureml_datasets(cleaned_input_datasets=input_datasets or [],
                                                       cleaned_output_datasets=output_datasets or [],
                                                       workspace=workspace)
    run_config.data = inputs
    run_config.output_data = outputs
    return run_config


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
    run = experiment.submit(script_run_config)
    tags = tags or {"commandline_args": " ".join(script_run_config.arguments)}
    run.set_tags(tags)

    _write_run_recovery_file(run)

    # These need to be 'print' not 'logging.info' so that the calling script sees them outside AzureML
    print("\n==============================================================================")
    print(f"Successfully queued run {run.id} in experiment {run.experiment.name}")
    print(f"Experiment name and run ID are available in file {RUN_RECOVERY_FILE}")
    print(f"Experiment URL: {run.experiment.get_portal_url()}")
    print(f"Run URL: {run.get_portal_url()}")
    print("==============================================================================\n")
    if wait_for_completion:
        print("Waiting for the completion of the AzureML run.")
        run.wait_for_completion(show_output=wait_for_completion_show_output)
        if not is_run_and_child_runs_completed(run):
            raise ValueError(f"Run {run.id} in experiment {run.experiment.name} or one of its child "
                             f"runs failed. Run status: {run.status}.")
    return run


def _str_to_path(s: Optional[PathOrString]) -> Optional[Path]:
    if isinstance(s, str):
        return Path(s)
    return s


def submit_to_azure_if_needed(  # type: ignore
        # ignore missing return statement since we 'exit' instead when submitting to AzureML
        compute_cluster_name: str,
        entry_script: Optional[PathOrString] = None,
        aml_workspace: Optional[Workspace] = None,
        workspace_config_path: Optional[PathOrString] = None,
        snapshot_root_directory: Optional[PathOrString] = None,
        script_params: Optional[List[str]] = None,
        conda_environment_file: Optional[Path] = None,
        aml_environment_name: str = "",
        experiment_name: Optional[str] = None,
        environment_variables: Optional[Dict[str, str]] = None,
        pip_extra_index_url: str = "",
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
        hyperdrive_config: Optional[HyperDriveConfig] = None) -> AzureRunInformation:  # pragma: no cover
    # This function is unit-tested, inside and outside AzureML, in the test_invoking_hello_world* unit tests, but
    # they run the code in a spawned subprocess which is not counted towards coverage analysis; hence the no-cover
    # pragma applied here. Furthermore, submit_to_azure_if_needed is broken into simple small functions which are
    # called with their own unit tests.
    """
    Submit a folder to Azure, if needed and run it.

    Use the flag --azureml to submit to AzureML, and leave it out to run locally.

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
    :param workspace_config_path: The 2nd option is to specify the path to the config.json file downloaded from the
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
    :param conda_environment_file: The file that contains the Conda environment definition.
    :param default_datastore: The data store in your AzureML workspace, that points to your training data in blob
    storage. This is described in more detail in the README.
    :param input_datasets: The script will consume all data in folder in blob storage as the input. The folder must
    exist in blob storage, in the location that you gave when creating the datastore. Once the script has run, it will
    also register the data in this folder as an AzureML dataset.
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
    :return: If the script is submitted to AzureML then we terminate python as the script should be executed in AzureML,
    otherwise we return a AzureRunInformation object.
    """
    _package_setup()
    workspace_config_path = _str_to_path(workspace_config_path)
    snapshot_root_directory = _str_to_path(snapshot_root_directory)
    cleaned_input_datasets = _replace_string_datasets(input_datasets or [],
                                                      default_datastore_name=default_datastore)
    cleaned_output_datasets = _replace_string_datasets(output_datasets or [],
                                                       default_datastore_name=default_datastore)
    # The present function will most likely be called from the script once it is running in AzureML.
    # The '--azureml' flag will not be present anymore, but we don't want to rely on that. From Run.get_context we
    # can infer if the present code is running in AzureML.
    in_azure = is_running_in_azure()
    if in_azure:
        return _generate_azure_datasets(cleaned_input_datasets, cleaned_output_datasets)

    # This codepath is reached when executing outside AzureML. Here we first check if a script submission to AzureML
    # is necessary. If not, return to the caller for local execution.
    if submit_to_azureml is None:
        submit_to_azureml = AZUREML_COMMANDLINE_FLAG in sys.argv[1:]
    if not submit_to_azureml:
        return AzureRunInformation(
            input_datasets=[d.local_folder for d in cleaned_input_datasets],
            output_datasets=[d.local_folder for d in cleaned_output_datasets],
            run=None,
            is_running_in_azure=False,
            output_folder=Path.cwd() / OUTPUT_FOLDER,
            logs_folder=Path.cwd() / LOGS_FOLDER
        )
    if snapshot_root_directory is None:
        logging.info(f"No snapshot root directory given. Uploading all files in the current directory {Path.cwd()}")
        snapshot_root_directory = Path.cwd()

    workspace = _get_workspace(aml_workspace, workspace_config_path)

    logging.info(f"Loaded AzureML workspace {workspace.name}")
    run_config = create_run_configuration(
        workspace=workspace,
        compute_cluster_name=compute_cluster_name,
        aml_environment_name=aml_environment_name,
        conda_environment_file=conda_environment_file,
        environment_variables=environment_variables,
        pip_extra_index_url=pip_extra_index_url,
        docker_base_image=docker_base_image,
        docker_shm_size=docker_shm_size,
        num_nodes=num_nodes,
        max_run_duration=max_run_duration,
        input_datasets=cleaned_input_datasets,
        output_datasets=cleaned_output_datasets,
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
        inputs[consumption.name] = consumption
    outputs = {}
    for index, d in enumerate(cleaned_output_datasets):
        out = d.to_output_dataset(workspace=workspace, dataset_index=index)
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


def _get_workspace(aml_workspace: Optional[Workspace], workspace_config_path: Optional[Path]) -> Workspace:
    """
    Obtain the AzureML workspace from either the passed in value or the passed in path
    :param aml_workspace: If provided this is returned as the AzureML Workspace
    :param workspace_config_path: If not provided with an AzureML Workspace, then load one given the information in this
    config
    :param return: The AzureML Workspace
    """
    if aml_workspace:
        workspace = aml_workspace
    elif workspace_config_path and workspace_config_path.is_file():
        auth = get_authentication()
        workspace = Workspace.from_config(path=str(workspace_config_path), auth=auth)
    else:
        raise ValueError("Cannot glean workspace config from parameters, and so not submitting to AzureML")
    return workspace


def _generate_azure_datasets(
        cleaned_input_datasets: List[DatasetConfig],
        cleaned_output_datasets: List[DatasetConfig]) -> AzureRunInformation:
    """
    Generate returned datasets when running in AzumreML
    :param cleaned_input_datasets: The list of input dataset configs
    :param cleaned_output_datasets: The list of output dataset configs
    :return: The AzureRunInformation containing the AzureML input and output dataset lists etc.
    """
    returned_input_datasets = [Path(RUN_CONTEXT.input_datasets[_input_dataset_key(index)])
                               for index in range(len(cleaned_input_datasets))]
    returned_output_datasets = [Path(RUN_CONTEXT.output_datasets[_output_dataset_key(index)])
                                for index in range(len(cleaned_output_datasets))]
    return AzureRunInformation(
        input_datasets=returned_input_datasets,  # type: ignore
        output_datasets=returned_output_datasets,  # type: ignore
        run=RUN_CONTEXT,
        is_running_in_azure=True,
        output_folder=Path.cwd() / OUTPUT_FOLDER,
        logs_folder=Path.cwd() / LOGS_FOLDER)


@contextmanager
def append_to_amlignore(lines_to_append: List[str], amlignore: Optional[Path] = None) -> Generator:
    """
    Context manager that appends lines to the .amlignore file, and reverts to the previous contents after leaving
    the context.
    If the file does not exist yet, it will be created, the contents written, and deleted when leaving the context.
    :param lines_to_append: The text lines that should be added at the end of the .amlignore file
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
    parser.add_argument("-p", "--workspace_config_path", type=str, required=False, help="AzureML workspace config file")
    parser.add_argument("-c", "--compute_cluster_name", type=str, required=True, help="AzureML cluster name")
    parser.add_argument("-y", "--snapshot_root_directory", type=str, required=True,
                        help="Root of snapshot to upload to AzureML")
    parser.add_argument("-t", "--entry_script", type=str, required=True,
                        help="The script to run in AzureML")
    parser.add_argument("-d", "--conda_environment_file", type=str, required=True, help="The environment to use")

    args = parser.parse_args()

    submit_to_azure_if_needed(
        workspace_config_path=Path(args.workspace_config_path),
        compute_cluster_name=args.compute_cluster_name,
        snapshot_root_directory=Path(args.snapshot_root_directory),
        entry_script=Path(args.entry_script),
        conda_environment_file=Path(args.conda_environment_file))


if __name__ == "__main__":
    main()  # pragma: no cover
