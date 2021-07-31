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
from argparse import ArgumentParser
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List, Optional, Union

from azureml.core import Environment, Experiment, Run, RunConfiguration, ScriptRunConfig, Workspace
from azureml.core.runconfig import DockerConfiguration, MpiConfiguration

from health.azure.azure_util import (DEFAULT_DOCKER_SHM_SIZE, create_run_recovery_id, get_authentication,
                                     get_or_create_python_environment,
                                     get_or_register_environment, run_duration_string_to_seconds,
                                     to_azure_friendly_string)
from health.azure.datasets import StrOrDatasetConfig, _input_dataset_key, _output_dataset_key, _replace_string_datasets

logger = logging.getLogger('health.azure')
logger.setLevel(logging.DEBUG)

RUN_RECOVERY_FILE = "most_recent_run.txt"
WORKSPACE_CONFIG_JSON = "config.json"
AZUREML_COMMANDLINE_FLAG = "--azureml"
RUN_CONTEXT = Run.get_context()
OUTPUT_FOLDER = "outputs"
LOG_FOLDER = "logs"
AML_IGNORE_FILE = ".amlignore"

PathOrString = Union[Path, str]


@dataclass
class AzureRunInformation:
    input_datasets: List[Optional[Path]]
    output_datasets: List[Optional[Path]]
    run: Run
    # If True, the present code is running inside of AzureML.
    is_running_in_azure: bool
    # In Azure, this would be the "outputs" folder. In local runs: "." or create a timestamped folder.
    # The folder that we create here must be added to .amlignore
    output_folder: Path
    log_folder: Path


def is_running_in_azure(aml_run: Run = RUN_CONTEXT) -> bool:
    """
    Returns True if the given run is inside of an AzureML machine, or False if it is a machine outside AzureML.
    :param aml_run: The run to check. If omitted, use the default run in RUN_CONTEXT
    :return: True if the given run is inside of an AzureML machine, or False if it is a machine outside AzureML.
    """
    return hasattr(aml_run, 'experiment')


def _str_to_path(s: Optional[PathOrString]) -> Optional[Path]:
    if isinstance(s, str):
        return Path(s)
    return s


def submit_to_azure_if_needed(  # type: ignore # missing return since we exit
        entry_script: PathOrString,
        compute_cluster_name: str,
        aml_workspace: Optional[Workspace] = None,
        workspace_config_path: Optional[PathOrString] = None,
        snapshot_root_directory: Optional[PathOrString] = None,
        script_params: Optional[List[str]] = None,
        conda_environment_file: Optional[Path] = None,
        aml_environment: str = "",
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
        exit_after_submission: bool = True,
        max_run_duration: str = "",
        submit_to_azureml: Optional[bool] = None) -> AzureRunInformation:
    """
    Submit a folder to Azure, if needed and run it.

    Use the flag --azureml to submit to AzureML, and leave it out to run locally.

    :param exit_after_submission: If True, the function calls exit(0) after a successful submission to AzureML.
    If False, it returns an AzureRunInformation object to the caller, where the submitted run can be read out.
    :param aml_environment: The name of an AzureML environment that should be used to submit the script. If not
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
    :return: If the script is submitted to AzureML then we terminate python as the script should be executed in AzureML,
    otherwise we return a AzureRunInformation object.
    """
    if isinstance(entry_script, str):
        entry_script = Path(entry_script)
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
        returned_input_datasets = [RUN_CONTEXT.input_datasets[_input_dataset_key(index)]
                                   for index in range(len(cleaned_input_datasets))]
        returned_output_datasets = [RUN_CONTEXT.output_datasets[_output_dataset_key(index)]
                                    for index in range(len(cleaned_output_datasets))]
        return AzureRunInformation(
            input_datasets=returned_input_datasets,
            output_datasets=returned_output_datasets,
            run=RUN_CONTEXT,
            is_running_in_azure=True,
            output_folder=Path.cwd() / OUTPUT_FOLDER,
            log_folder=Path.cwd() / LOG_FOLDER
        )

    # This codepath is reached when executing outside AzureML. Here we first check if a script submission to AzureML
    # is necessary. If not, return to the caller for local execution.
    local_run_info = AzureRunInformation(
        input_datasets=[d.local_folder for d in cleaned_input_datasets],
        output_datasets=[d.local_folder for d in cleaned_output_datasets],
        run=RUN_CONTEXT,
        is_running_in_azure=False,
        output_folder=Path.cwd() / OUTPUT_FOLDER,
        log_folder=Path.cwd() / LOG_FOLDER
    )
    if submit_to_azureml is None:
        submit_to_azureml = AZUREML_COMMANDLINE_FLAG in sys.argv[1:]
    if not submit_to_azureml:
        return local_run_info
    if snapshot_root_directory is None:
        logging.info(f"No snapshot root directory given. Uploading all files in the current directory {Path.cwd()}")
        snapshot_root_directory = Path.cwd()

    if aml_workspace:
        workspace = aml_workspace
    elif workspace_config_path and workspace_config_path.is_file():
        auth = get_authentication()
        workspace = Workspace.from_config(path=str(workspace_config_path), auth=auth)
    else:
        raise ValueError("Cannot glean workspace config from parameters. Use 'workspace_config_path' to point to a "
                         "config.json file, or 'aml_workspace' to pass a Workspace object.")

    logging.info(f"Loaded AzureML workspace {workspace.name}")
    if not script_params:
        script_params = [p for p in sys.argv[1:] if p != AZUREML_COMMANDLINE_FLAG]
    entry_script_relative = entry_script.relative_to(snapshot_root_directory)

    if compute_cluster_name not in workspace.compute_targets:
        raise ValueError(f"Could not find the compute target {compute_cluster_name} in the AzureML workspace. ",
                         f"Existing clusters: {list(workspace.compute_targets.keys())}")

    if aml_environment:
        # TODO: Split off version
        environment = Environment.get(workspace, aml_environment)
    elif conda_environment_file:
        environment = get_or_create_python_environment(conda_environment_file=conda_environment_file,
                                                       pip_extra_index_url=pip_extra_index_url,
                                                       docker_base_image=docker_base_image,
                                                       environment_variables=environment_variables)
        environment = get_or_register_environment(workspace, environment)
    else:
        raise ValueError("One of the two arguments 'aml_environment' or 'conda_environment_file' must be given.")

    run_config = RunConfiguration(
        script=entry_script_relative,
        arguments=script_params)
    run_config.environment = environment
    run_config.target = compute_cluster_name
    if max_run_duration:
        run_config.max_run_duration_seconds = run_duration_string_to_seconds(max_run_duration)
    if num_nodes > 1:
        distributed_job_config = MpiConfiguration(node_count=num_nodes)
        run_config.mpi = distributed_job_config
        run_config.framework = "Python"
        run_config.communicator = "IntelMpi"
        run_config.node_count = distributed_job_config.node_count

    inputs = {}
    for index, d in enumerate(cleaned_input_datasets):
        consumption = d.to_input_dataset(workspace=workspace, dataset_index=index)
        inputs[consumption.name] = consumption
    outputs = {}
    for index, d in enumerate(cleaned_output_datasets):
        out = d.to_output_dataset(workspace=workspace, dataset_index=index)
        outputs[out.name] = out
    run_config.data = inputs
    run_config.output_data = outputs
    run_config.docker = DockerConfiguration(use_docker=True,
                                            shm_size=(docker_shm_size or DEFAULT_DOCKER_SHM_SIZE))
    script_run_config = ScriptRunConfig(
        source_directory=str(snapshot_root_directory),
        run_config=run_config)

    cleaned_experiment_name = to_azure_friendly_string(experiment_name)
    if not cleaned_experiment_name:
        cleaned_experiment_name = to_azure_friendly_string(entry_script.stem)
    experiment = Experiment(workspace=workspace, name=cleaned_experiment_name)

    amlignore_path = snapshot_root_directory / AML_IGNORE_FILE
    lines_to_append = [str(path) for path in (ignored_folders or [])]
    with append_to_amlignore(
            amlignore=amlignore_path,
            lines_to_append=lines_to_append):
        run: Run = experiment.submit(script_run_config)
    run.set_tags({"commandline_args": " ".join(script_params)})

    recovery_id = create_run_recovery_id(run)
    recovery_file = Path(RUN_RECOVERY_FILE)
    if recovery_file.exists():
        recovery_file.unlink()
    recovery_file.write_text(recovery_id)

    # These need to be 'print' not 'logging.info' so that the calling script sees them outside AzureML
    print("\n==============================================================================")
    print(f"Successfully queued run {run.id} in experiment {experiment.name}")
    print(f"Experiment name and run ID are available in file {RUN_RECOVERY_FILE}")
    print(f"Experiment URL: {experiment.get_portal_url()}")
    print(f"Run URL: {run.get_portal_url()}")
    print("==============================================================================\n")
    if wait_for_completion:
        print("Waiting for the completion of the AzureML run.")
        run.wait_for_completion(show_output=wait_for_completion_show_output)

    if exit_after_submission:
        exit(0)
    else:
        local_run_info.run = run
        return local_run_info


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
    new_contents = old_contents.splitlines() + lines_to_append
    amlignore.write_text("\n".join(new_contents))
    yield
    if amlignore_exists_already:
        amlignore.write_text(old_contents)
    else:
        amlignore.unlink()


def package_setup_and_hacks() -> None:
    """
    Set up the Python packages where needed. In particular, reduce the logging level for some of the used
    libraries, which are particularly talkative in DEBUG mode. Usually when running in DEBUG mode, we want
    diagnostics about the model building itself, but not for the underlying libraries.
    It also adds workarounds for known issues in some packages.
    """
    # Urllib3 prints out connection information for each call to write metrics, etc
    logging.getLogger('urllib3').setLevel(logging.INFO)
    logging.getLogger('msrest').setLevel(logging.INFO)
    # AzureML prints too many details about logging metrics
    logging.getLogger('azureml').setLevel(logging.INFO)
    # This is working around a spurious error message thrown by MKL, see
    # https://github.com/pytorch/pytorch/issues/37377
    os.environ['MKL_THREADING_LAYER'] = 'GNU'


def main() -> None:
    """
    Handle submit_to_azure if called from the command line.
    """
    parser = ArgumentParser()
    parser.add_argument("-w", "--workspace_name", type=str, required=False, help="Azure ML workspace name")
    parser.add_argument("-s", "--subscription_id", type=str, required=False, help="AzureML subscription id")
    parser.add_argument("-r", "--resource_group", type=str, required=False, help="AzureML resource group")
    parser.add_argument("-p", "--workspace_config_path", type=str, required=False, help="AzureML workspace config file")
    parser.add_argument("-c", "--compute_cluster_name", type=str, required=True, help="AzureML cluster name")
    parser.add_argument("-y", "--snapshot_root_directory", type=str, required=True,
                        help="Root of snapshot to upload to AzureML")
    parser.add_argument("-t", "--entry_script", type=str, required=True,
                        help="The script to run in AzureML")
    parser.add_argument("-d", "--conda_environment_file", type=str, required=True, help="The environment to use")

    args = parser.parse_args()

    submit_to_azure_if_needed(
        workspace_config_path=args.workspace_config_path,
        compute_cluster_name=args.compute_cluster_name,
        snapshot_root_directory=args.snapshot_root_directory,
        entry_script=args.entry_script,
        conda_environment_file=args.conda_environment_file)


if __name__ == "__main__":
    main()
