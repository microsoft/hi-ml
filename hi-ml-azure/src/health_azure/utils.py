#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
Utility functions for interacting with AzureML runs
"""
import hashlib
import logging
import os
import re
from argparse import Namespace
from enum import Enum
from itertools import islice
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import conda_merge
import ruamel.yaml
from azureml._restclient.constants import RunStatus
from azureml.core import Environment, Experiment, Run, Workspace, get_run
from azureml.core.authentication import InteractiveLoginAuthentication, ServicePrincipalAuthentication
from azureml.core.conda_dependencies import CondaDependencies
from azureml.data.azure_storage_datastore import AzureBlobDatastore

EXPERIMENT_RUN_SEPARATOR = ":"
DEFAULT_UPLOAD_TIMEOUT_SECONDS: int = 36_000  # 10 Hours

# The version to use when creating an AzureML Python environment. We create all environments with a unique hashed
# name, hence version will always be fixed
ENVIRONMENT_VERSION = "1"

# Environment variables used for authentication
ENV_SERVICE_PRINCIPAL_ID = "HIML_SERVICE_PRINCIPAL_ID"
ENV_SERVICE_PRINCIPAL_PASSWORD = "HIML_SERVICE_PRINCIPAL_PASSWORD"
ENV_TENANT_ID = "HIML_TENANT_ID"
ENV_RESOURCE_GROUP = "HIML_RESOURCE_GROUP"
ENV_SUBSCRIPTION_ID = "HIML_SUBSCRIPTION_ID"
ENV_WORKSPACE_NAME = "HIML_WORKSPACE_NAME"

# Environment variables used for multi-node training
ENV_AZ_BATCHAI_MPI_MASTER_NODE = "AZ_BATCHAI_MPI_MASTER_NODE"
ENV_MASTER_ADDR = "MASTER_ADDR"
ENV_MASTER_IP = "MASTER_IP"
ENV_MASTER_PORT = "MASTER_PORT"
ENV_OMPI_COMM_WORLD_RANK = "OMPI_COMM_WORLD_RANK"
ENV_NODE_RANK = "NODE_RANK"
ENV_GLOBAL_RANK = "GLOBAL_RANK"
ENV_LOCAL_RANK = "LOCAL_RANK"

RUN_CONTEXT = Run.get_context()
WORKSPACE_CONFIG_JSON = "config.json"


def _find_file(file_name: str, stop_at_pythonpath: bool = True) -> Optional[Path]:
    """
    Recurse up the file system, starting at the current working directory, to find a file. Optionally stop when we hit
    the PYTHONPATH root (defaults to stopping).

    :param file_name: The fine name of the file to find.
    :param stop_at_pythonpath: (Defaults to True.) Whether to stop at the PYTHONPATH root.
    :return: The path to the file, or None if it cannot be found.
    """

    def return_file_or_parent(
            start_at: Path,
            file_name: str,
            stop_at_pythonpath: bool,
            pythonpaths: List[Path]) -> Optional[Path]:
        for child in start_at.iterdir():
            if child.is_file() and child.name == file_name:
                return child
        if start_at.parent == start_at or start_at in pythonpaths:
            return None
        return return_file_or_parent(start_at.parent, file_name, stop_at_pythonpath, pythonpaths)

    pythonpaths: List[Path] = []
    if 'PYTHONPATH' in os.environ:
        pythonpaths = [Path(path_string) for path_string in os.environ['PYTHONPATH'].split(os.pathsep)]
    return return_file_or_parent(
        start_at=Path.cwd(),
        file_name=file_name,
        stop_at_pythonpath=stop_at_pythonpath,
        pythonpaths=pythonpaths)


def get_workspace(aml_workspace: Optional[Workspace], workspace_config_path: Optional[Path]) -> Workspace:
    """
    Retrieve an Azure ML Workspace from one of several places:
      1. If the function has been called during an AML run (i.e. on an Azure agent), returns the associated workspace
      2. If a Workspace object has been provided by the user, return that
      3. If a path to a Workspace config file has been provided, load the workspace according to that.

    If not running inside AML and neither a workspace nor the config file are provided, the code will try to locate a
    config.json file in any of the parent folders of the current working directory. If that succeeds, that config.json
    file will be used to instantiate the workspace.

    :param aml_workspace: If provided this is returned as the AzureML Workspace.
    :param workspace_config_path: If not provided with an AzureML Workspace, then load one given the information in this
        config
    :return: An AzureML workspace.
    """
    if is_running_in_azure_ml(RUN_CONTEXT):
        return RUN_CONTEXT.experiment.workspace

    if aml_workspace:
        return aml_workspace

    if workspace_config_path is None:
        workspace_config_path = _find_file(WORKSPACE_CONFIG_JSON)
        if workspace_config_path:
            logging.info(f"Using the workspace config file {str(workspace_config_path.absolute())}")
        else:
            raise ValueError("No workspace config file given, nor can we find one.")

    if workspace_config_path.is_file():
        auth = get_authentication()
        return Workspace.from_config(path=str(workspace_config_path), auth=auth)
    raise ValueError("Workspace config file does not exist or cannot be read.")


def create_run_recovery_id(run: Run) -> str:
    """
   Creates an recovery id for a run so it's checkpoints could be recovered for training/testing

   :param run: an instantiated run.
   :return: recovery id for a given run in format: [experiment name]:[run id]
   """
    return str(run.experiment.name + EXPERIMENT_RUN_SEPARATOR + run.id)


def split_recovery_id(id: str) -> Tuple[str, str]:
    """
    Splits a run ID into the experiment name and the actual run.
    The argument can be in the format 'experiment_name:run_id',
    or just a run ID like user_branch_abcde12_123. In the latter case, everything before the last
    two alphanumeric parts is assumed to be the experiment name.

    :param id: The string run ID.
    :return: experiment name and run name
    """
    components = id.strip().split(EXPERIMENT_RUN_SEPARATOR)
    if len(components) > 2:
        raise ValueError("recovery_id must be in the format: 'experiment_name:run_id', but got: {}".format(id))
    elif len(components) == 2:
        return components[0], components[1]
    else:
        recovery_id_regex = r"^(\w+)_\d+_[0-9a-f]+$|^(\w+)_\d+$"
        match = re.match(recovery_id_regex, id)
        if not match:
            raise ValueError("The recovery ID was not in the expected format: {}".format(id))
        return (match.group(1) or match.group(2)), id


def fetch_run(workspace: Workspace, run_recovery_id: str) -> Run:
    """
    Finds an existing run in an experiment, based on a recovery ID that contains the experiment ID and the actual RunId.
    The run can be specified either in the experiment_name:run_id format, or just the run_id.

    :param workspace: the configured AzureML workspace to search for the experiment.
    :param run_recovery_id: The Run to find. Either in the full recovery ID format, experiment_name:run_id or
        just the run_id
    :return: The AzureML run.
    """
    experiment, run = split_recovery_id(run_recovery_id)
    try:
        experiment_to_recover = Experiment(workspace, experiment)
    except Exception as ex:
        raise Exception(f"Unable to retrieve run {run} in experiment {experiment}: {str(ex)}")
    run_to_recover = fetch_run_for_experiment(experiment_to_recover, run)
    logging.info("Fetched run #{} {} from experiment {}.".format(run, run_to_recover.number, experiment))
    return run_to_recover


def fetch_run_for_experiment(experiment_to_recover: Experiment, run_id: str) -> Run:
    """
    Gets an AzureML Run object for a given run ID in an experiment.

    :param experiment_to_recover: an experiment
    :param run_id: a string representing the Run ID of one of the runs of the experiment
    :return: the run matching run_id_or_number; raises an exception if not found
    """
    try:
        return get_run(experiment=experiment_to_recover, run_id=run_id, rehydrate=True)
    except Exception:
        available_runs = experiment_to_recover.get_runs()
        available_ids = ", ".join([run.id for run in available_runs])
        raise (Exception(
            "Run {} not found for experiment: {}. Available runs are: {}".format(
                run_id, experiment_to_recover.name, available_ids)))


def get_authentication() -> Union[InteractiveLoginAuthentication, ServicePrincipalAuthentication]:
    """
    Creates a service principal authentication object with the application ID stored in the present object. The
    application key is read from the environment.

    :return: A ServicePrincipalAuthentication object that has the application ID and key or None if the key is not
        present
    """
    service_principal_id = get_secret_from_environment(ENV_SERVICE_PRINCIPAL_ID, allow_missing=True)
    tenant_id = get_secret_from_environment(ENV_TENANT_ID, allow_missing=True)
    service_principal_password = get_secret_from_environment(ENV_SERVICE_PRINCIPAL_PASSWORD, allow_missing=True)
    if service_principal_id and tenant_id and service_principal_password:
        return ServicePrincipalAuthentication(
            tenant_id=tenant_id,
            service_principal_id=service_principal_id,
            service_principal_password=service_principal_password)
    logging.info("Using interactive login to Azure. To use Service Principal authentication, set the environment "
                 f"variables {ENV_SERVICE_PRINCIPAL_ID}, {ENV_SERVICE_PRINCIPAL_PASSWORD}, and {ENV_TENANT_ID}")
    return InteractiveLoginAuthentication()


def get_secret_from_environment(name: str, allow_missing: bool = False) -> Optional[str]:
    """
    Gets a password or key from the secrets file or environment variables.

    :param name: The name of the environment variable to read. It will be converted to uppercase.
    :param allow_missing: If true, the function returns None if there is no entry of the given name in any of the
        places searched. If false, missing entries will raise a ValueError.
    :return: Value of the secret. None, if there is no value and allow_missing is True.
    """
    name = name.upper()
    value = os.environ.get(name, None)
    if not value and not allow_missing:
        raise ValueError(f"There is no value stored for the secret named '{name}'")
    return value


def to_azure_friendly_string(x: Optional[str]) -> Optional[str]:
    """
    Given a string, ensure it can be used in Azure by replacing everything apart from a-z, A-Z, 0-9, or _ with _,
    and replace multiple _ with a single _.

    :param x: Optional string to be converted.
    :return: Converted string, if one supplied. None otherwise.
    """
    if x is None:
        return x
    else:
        return re.sub('_+', '_', re.sub(r'\W+', '_', x))


def _log_conda_dependencies_stats(conda: CondaDependencies, message_prefix: str) -> None:
    """
    Write number of conda and pip packages to logs.

    :param conda: A conda dependencies object
    :param message_prefix: A message to prefix to the log string.
    """
    conda_packages_count = len(list(conda.conda_packages))
    pip_packages_count = len(list(conda.pip_packages))
    logging.info(f"{message_prefix}: {conda_packages_count} conda packages, {pip_packages_count} pip packages")
    logging.debug("  Conda packages:")
    for p in conda.conda_packages:
        logging.debug(f"    {p}")
    logging.debug("  Pip packages:")
    for p in conda.pip_packages:
        logging.debug(f"    {p}")


def merge_conda_files(files: List[Path], result_file: Path) -> None:
    """
    Merges the given Conda environment files using the conda_merge package, and writes the merged file to disk.

    :param files: The Conda environment files to read.
    :param result_file: The location where the merge results should be written.
    """
    for file in files:
        _log_conda_dependencies_stats(CondaDependencies(file), f"Conda environment in {file}")
    # This code is a slightly modified version of conda_merge. That code can't be re-used easily
    # it defaults to writing to stdout
    env_definitions = [conda_merge.read_file(str(f)) for f in files]
    unified_definition = {}
    NAME = "name"
    CHANNELS = "channels"
    DEPENDENCIES = "dependencies"

    name = conda_merge.merge_names(env.get(NAME) for env in env_definitions)
    if name:
        unified_definition[NAME] = name

    try:
        channels = conda_merge.merge_channels(env.get(CHANNELS) for env in env_definitions)
    except conda_merge.MergeError:
        logging.error("Failed to merge channel priorities.")
        raise
    if channels:
        unified_definition[CHANNELS] = channels

    try:
        deps = conda_merge.merge_dependencies(env.get(DEPENDENCIES) for env in env_definitions)
    except conda_merge.MergeError:
        logging.error("Failed to merge dependencies.")
        raise
    if deps:
        unified_definition[DEPENDENCIES] = deps
    else:
        raise ValueError("No dependencies found in any of the conda files.")

    with result_file.open("w") as f:
        ruamel.yaml.dump(unified_definition, f, indent=2, default_flow_style=False)
    _log_conda_dependencies_stats(CondaDependencies(result_file), "Merged Conda environment")


def create_python_environment(conda_environment_file: Path,
                              pip_extra_index_url: str = "",
                              workspace: Optional[Workspace] = None,
                              private_pip_wheel_path: Optional[Path] = None,
                              docker_base_image: str = "",
                              environment_variables: Optional[Dict[str, str]] = None) -> Environment:
    """
    Creates a description for the Python execution environment in AzureML, based on the Conda environment
    definition files that are specified in `source_config`. If such environment with this Conda environment already
    exists, it is retrieved, otherwise created afresh.

    :param environment_variables: The environment variables that should be set when running in AzureML.
    :param docker_base_image: The Docker base image that should be used when creating a new Docker image.
    :param pip_extra_index_url: If provided, use this PIP package index to find additional packages when building
        the Docker image.
    :param workspace: The AzureML workspace to work in, required if private_pip_wheel_path is supplied.
    :param private_pip_wheel_path: If provided, add this wheel as a private package to the AzureML workspace.
    :param conda_environment_file: The file that contains the Conda environment definition.
    """
    conda_dependencies = CondaDependencies(conda_dependencies_file_path=conda_environment_file)
    yaml_contents = conda_environment_file.read_text()
    if pip_extra_index_url:
        # When an extra-index-url is supplied, swap the order in which packages are searched for.
        # This is necessary if we need to consume packages from extra-index that clash with names of packages on
        # pypi
        conda_dependencies.set_pip_option(f"--index-url {pip_extra_index_url}")
        conda_dependencies.set_pip_option("--extra-index-url https://pypi.org/simple")
    # By default, define several environment variables that work around known issues in the software stack
    environment_variables = {
        "AZUREML_OUTPUT_UPLOAD_TIMEOUT_SEC": "3600",
        # Occasionally uploading data during the run takes too long, and makes the job fail. Default is 300.
        "AZUREML_RUN_KILL_SIGNAL_TIMEOUT_SEC": "900",
        "MKL_SERVICE_FORCE_INTEL": "1",
        # Switching to a new software stack in AML for mounting datasets
        "RSLEX_DIRECT_VOLUME_MOUNT": "true",
        "RSLEX_DIRECT_VOLUME_MOUNT_MAX_CACHE_SIZE": "1",
        **(environment_variables or {})
    }
    # See if this package as a whl exists, and if so, register it with AzureML environment.
    if workspace is not None and private_pip_wheel_path is not None:
        if private_pip_wheel_path.is_file():
            whl_url = Environment.add_private_pip_wheel(workspace=workspace,
                                                        file_path=private_pip_wheel_path,
                                                        exist_ok=True)
            conda_dependencies.add_pip_package(whl_url)
            print(f"Added add_private_pip_wheel {private_pip_wheel_path} to AzureML environment.")
        else:
            raise FileNotFoundError(f"Cannot add add_private_pip_wheel: {private_pip_wheel_path}, it is not a file.")
    # Create a name for the environment that will likely uniquely identify it. AzureML does hashing on top of that,
    # and will re-use existing environments even if they don't have the same name.
    # Hashing should include everything that can reasonably change. Rely on hashlib here, because the built-in
    # hash function gives different results for the same string in different python instances.
    hash_string = "\n".join([yaml_contents, docker_base_image, str(environment_variables)])
    sha1 = hashlib.sha1(hash_string.encode("utf8"))
    overall_hash = sha1.hexdigest()[:32]
    unique_env_name = f"HealthML-{overall_hash}"
    env = Environment(name=unique_env_name)
    env.python.conda_dependencies = conda_dependencies
    if docker_base_image:
        env.docker.base_image = docker_base_image
    env.environment_variables = environment_variables
    return env


def register_environment(workspace: Workspace, environment: Environment) -> Environment:
    """
    Try to get the AzureML environment by name and version from the AzureML workspace. If that fails, register the
    environment on the workspace.

    :param workspace: The AzureML workspace to use.
    :param environment: An AzureML execution environment.
    :return: An AzureML execution environment. If the environment did already exist on the workspace, the return value
        is the environment as registered on the workspace, otherwise it is equal to the environment argument.
    """
    try:
        env = Environment.get(workspace, name=environment.name, version=environment.version)
        logging.info(f"Using existing Python environment '{env.name}'.")
    except Exception:
        logging.info(f"Python environment '{environment.name}' does not yet exist, creating and registering it.")
        environment.register(workspace)
    return environment


def run_duration_string_to_seconds(s: str) -> Optional[int]:
    """
    Parse a string that represents a timespan, and returns it converted into seconds. The string is expected to be
    floating point number with a single character suffix s, m, h, d for seconds, minutes, hours, day.
    Examples: '3.5h', '2d'. If the argument is an empty string, None is returned.

    :param s: The string to parse.
    :return: The timespan represented in the string converted to seconds.
    """
    s = s.strip()
    if not s:
        return None
    suffix = s[-1]
    if suffix == "s":
        multiplier = 1
    elif suffix == "m":
        multiplier = 60
    elif suffix == "h":
        multiplier = 60 * 60
    elif suffix == "d":
        multiplier = 24 * 60 * 60
    else:
        raise ValueError("s", f"Invalid suffix: Must be one of 's', 'm', 'h', 'd', but got: {s}")  # type: ignore
    return int(float(s[:-1]) * multiplier)


def set_environment_variables_for_multi_node() -> None:
    """
    Sets the environment variables that PyTorch Lightning needs for multi-node training.
    """
    if ENV_AZ_BATCHAI_MPI_MASTER_NODE in os.environ:
        # For AML BATCHAI
        os.environ[ENV_MASTER_ADDR] = os.environ[ENV_AZ_BATCHAI_MPI_MASTER_NODE]
    elif ENV_MASTER_IP in os.environ:
        # AKS
        os.environ[ENV_MASTER_ADDR] = os.environ[ENV_MASTER_IP]
    else:
        logging.info("No settings for the MPI central node found. Assuming that this is a single node training job.")
        return

    if ENV_MASTER_PORT not in os.environ:
        os.environ[ENV_MASTER_PORT] = "6105"

    if ENV_OMPI_COMM_WORLD_RANK in os.environ:
        os.environ[ENV_NODE_RANK] = os.environ[ENV_OMPI_COMM_WORLD_RANK]  # node rank is the world_rank from mpi run
    env_vars = ", ".join(f"{var} = {os.environ[var]}" for var in [ENV_MASTER_ADDR, ENV_MASTER_PORT, ENV_NODE_RANK])
    print(f"Distributed training: {env_vars}")


def is_run_and_child_runs_completed(run: Run) -> bool:
    """
    Checks if the given run has successfully completed. If the run has child runs, it also checks if the child runs
    completed successfully.

    :param run: The AzureML run to check.
    :return: True if the run and all child runs completed successfully.
    """

    def is_completed(run: Run) -> bool:
        status = run.get_status()
        if run.status == RunStatus.COMPLETED:
            return True
        logging.info(f"Run {run.id} in experiment {run.experiment.name} finished with status {status}.")
        return False

    runs = list(run.get_children())
    runs.append(run)
    return all(is_completed(run) for run in runs)


def get_most_recent_run_id(run_recovery_file: Path) -> str:
    """
    Gets the string name of the most recently executed AzureML run. This is picked up from the `most_recent_run.txt`
    file when running on the cloud.

    :param run_recovery_file: The path of the run recovery file
    :return: The run id
    """
    assert run_recovery_file.is_file(), "When running in cloud builds, this should pick up the ID of a previous \
                                         training run"
    run_id = run_recovery_file.read_text().strip()
    logging.info(f"Read this run ID from file: {run_id}.")
    return run_id


def get_most_recent_run(run_recovery_file: Path, workspace: Workspace) -> Run:
    """
    Gets the name of the most recently executed AzureML run, instantiates that Run object and returns it.
    :param run_recovery_file: The path of the run recovery file
    :param workspace: Azure ML Workspace
    :return: The Run
    """
    run_or_recovery_id = get_most_recent_run_id(run_recovery_file)
    # Check if the id loaded is of run_recovery_id format
    if len(run_or_recovery_id.split(":")) > 1:
        return fetch_run(workspace, run_or_recovery_id)
    # Otherwise treat it as a run_id
    return get_aml_run_from_run_id(run_or_recovery_id, aml_workspace=workspace)


class AzureRunIdSource(Enum):
    LATEST_RUN_FILE = 1
    EXPERIMENT_LATEST = 2
    RUN_ID = 3
    RUN_IDS = 4
    RUN_RECOVERY_ID = 5
    RUN_RECOVERY_IDS = 6


def determine_run_id_source(args: Namespace) -> AzureRunIdSource:
    """
    From the args inputted, determine what is the source of Runs to be downloaded and plotted
    (e.g. extract id from latest run file, or take most recent run of an Experiment etc. )

    :param args: Arguments for determining the source of AML Runs to be retrieved
    :raises ValueError: If none of expected args for retrieving Runs are provided
    :return: The source from which to extract the latest Run id(s)
    """
    if "latest_run_file" in args and args.latest_run_file is not None:
        return AzureRunIdSource.LATEST_RUN_FILE
    if "experiment" in args and args.experiment is not None:
        return AzureRunIdSource.EXPERIMENT_LATEST
    if "run_recovery_ids" in args and args.run_recovery_ids is not None and len(args.run_recovery_ids) > 0:
        return AzureRunIdSource.RUN_RECOVERY_IDS
    if "run_recovery_id" in args and args.run_recovery_id is not None:
        return AzureRunIdSource.RUN_RECOVERY_ID
    if "run_id" in args and args.run_id is not None:
        return AzureRunIdSource.RUN_ID
    if "run_ids" in args and args.run_ids is not None and len(args.run_ids) > 0:
        return AzureRunIdSource.RUN_IDS
    raise ValueError("One of latest_run_file, experiment, run_recovery_id(s) or run_id(s) must be provided")


def get_aml_run_from_latest_run_file(args: Namespace, workspace: Workspace) -> Run:
    """
    Returns the Run object corresponding to the id found in the most recent run file.

    :param args: command line args including latest_run_file
    :param workspace: An Azure ML Workspace object
    :return the Run object corresponding to the id found in the most recent run file.
    """
    latest_run_path = Path(args.latest_run_file)
    return get_most_recent_run(latest_run_path, workspace)


def get_latest_aml_runs_from_experiment(args: Namespace, workspace: Workspace) -> List[Run]:
    """
    Get latest 'num_runs' runs from an AML experiment

    :param args: command line args including experiment name and number of runs to return
    :param workspace: AML Workspace
    :raises ValueError: If Experiment experiment doesn't exist within Workspace
    :return: List of AML Runs
    """
    experiment_name = args.experiment
    tags = args.tags or None
    num_runs = args.num_runs if 'num_runs' in args else 1

    if experiment_name not in workspace.experiments:
        raise ValueError(f"No such experiment {experiment_name} in workspace")

    experiment: Experiment = workspace.experiments[experiment_name]
    return list(islice(experiment.get_runs(tags=tags), num_runs))


def get_aml_runs_from_recovery_ids(args: Namespace, aml_workspace: Optional[Workspace] = None,
                                   workspace_config_path: Optional[Path] = None) -> List[Run]:
    """
    Retrieve multiple Azure ML Runs for each of the run_recovery_ids specified in args.

    :param args: command line arguments
    :param aml_workspace: Optional Azure ML Workspace object
    :param workspace_config_path: Optional path containing AML Workspace settings
    :return: List of AML Runs
    """

    def _get_run_recovery_ids_from_args(args: Namespace) -> List[str]:  # pragma: no cover
        """
        Retrieve a list of run recovery ids from the args as long as more than one is supplied.

        :param args: The command line arguments
        :return: A list of run_recovery_ids as passed in to the command line
        """
        if "run_recovery_ids" not in args or len(args.run_recovery_ids) == 0:
            raise ValueError("Expected to find run_recovery_ids in args but did not")
        else:
            return args.run_recovery_ids

    workspace = get_workspace(aml_workspace=aml_workspace, workspace_config_path=workspace_config_path)

    run_recovery_ids = _get_run_recovery_ids_from_args(args)

    runs = [fetch_run(workspace, run_id) for run_id in run_recovery_ids]
    return [r for r in runs if r is not None]


def get_aml_run_from_recovery_id(args: Namespace, aml_workspace: Optional[Workspace] = None,
                                 workspace_config_path: Optional[Path] = None) -> Run:
    """
    Retrieve a single Azure ML Run for the run_recovery_id specified in args.

    :param args: command line arguments
    :param aml_workspace: Optional Azure ML Workspace object
    :param workspace_config_path: Optional path containing AML Workspace settings
    :return: A single AML Run
    """
    if "run_recovery_id" in args and args.run_recovery_id:
        run_recovery_id = args.run_recovery_id
    else:
        raise ValueError("No run_recovery_id in args")

    workspace = get_workspace(aml_workspace=aml_workspace, workspace_config_path=workspace_config_path)

    return fetch_run(workspace, run_recovery_id)


def get_aml_run_from_run_id(run_id: str, aml_workspace: Optional[Workspace] = None,
                            workspace_config_path: Optional[Path] = None) -> Run:
    """
    Retrieve an Azure ML Run, firstly by retrieving the corresponding Workspace, and then getting the
    run according to the specified run_id. If running in AML, will take the current workspace. Otherwise, if
    neither aml_workspace nor workspace_config_path are provided, will try to locate a config.json file
    in any of the parent folders of the current working directory.

    :param run_id: the parameter corresponding to the 'id' property of the Run
    :param aml_workspace: Optional Azure ML Workspace object
    :param workspace_config_path: Optional path to a Workspace config file
    :return: The Azure ML Run object with the given run_id
    """
    workspace = get_workspace(aml_workspace=aml_workspace, workspace_config_path=workspace_config_path)
    run = workspace.get_run(run_id)
    return run


def get_aml_run_from_run_id_args(args: Namespace, aml_workspace: Optional[Workspace] = None,
                                 workspace_config_path: Optional[Path] = None) -> Run:
    """
    Lookup the run_id arg and then retrieve the Azure ML Run object with this id.

    :param args: Command line args
    :param aml_workspace: Optional Azure ML Workspace object
    :param workspace_config_path: Optional path to a Workspace config file
    :return: The Azure ML Run object with the id as specified by args.run_id
    """
    if "run_id" in args and args.run_id:
        run_id = args.run_id
    else:
        raise ValueError("No run_id in args")
    return get_aml_run_from_run_id(run_id, aml_workspace=aml_workspace, workspace_config_path=workspace_config_path)


def get_aml_runs_from_run_ids(args: Namespace, aml_workspace: Optional[Workspace] = None,
                              workspace_config_path: Optional[Path] = None) -> List[Run]:
    """
    Retrieve AzureML Runs for each of the Run Ids specified in args. If running in AML, will take the
    current workspace. Otherwise, if neither aml_workspace nor workspace_config_path are provided,
    will try to locate a config.json file in any of the parent folders of the current working directory.

    :param args: command line args including experiment name and number of runs to return
    :param aml_workspace: Optional Azure ML Workspace object
    :param workspace_config_path: Optional path containing AML Workspace settings
    :return: List of AML Runs
    """

    def _get_run_ids_from_args(args: Namespace) -> List[str]:  # pragma: no cover
        """
        Retrieve a list of run  ids from the args as long as more than one is supplied.

        :param args: The command line arguments
        :return: A list of run_ids as passed in to the command line
        """
        if len(args.run_ids) == 0:
            raise ValueError("Expected to find run_ids in args but did not")
        else:
            return args.run_ids

    workspace = get_workspace(aml_workspace=aml_workspace, workspace_config_path=workspace_config_path)
    run_ids = _get_run_ids_from_args(args)

    runs = [get_aml_run_from_run_id(r_id, aml_workspace=workspace) for r_id in run_ids]
    return [r for r in runs if r is not None]


def get_aml_runs(args: Namespace, workspace: Workspace, run_id_source: AzureRunIdSource) -> List[Run]:
    """
    Download runs from Azure ML. Runs are specified either in file specified in latest_run_file,
    by run_recovery_ids, or else the latest 'num_runs' runs from experiment 'experiment_name' as
    specified in args.

    :param args: Arguments for determining the source of AML Runs to be retrieved
    :param workspace: Azure ML Workspace
    :param run_id_source: The source from which to download AML Runs
    :raises ValueError: If experiment_name in args does not exist in the Workspace
    :return: List of Azure ML Runs, or an empty list if none are retrieved
    """
    if run_id_source == AzureRunIdSource.LATEST_RUN_FILE:
        runs = [get_aml_run_from_latest_run_file(args, workspace)]
    elif run_id_source == AzureRunIdSource.EXPERIMENT_LATEST:
        runs = get_latest_aml_runs_from_experiment(args, workspace)
    elif run_id_source == AzureRunIdSource.RUN_RECOVERY_ID:
        runs = [get_aml_run_from_recovery_id(args, workspace)]
    elif run_id_source == AzureRunIdSource.RUN_RECOVERY_IDS:
        runs = get_aml_runs_from_recovery_ids(args, workspace)
    elif run_id_source == AzureRunIdSource.RUN_ID:
        runs = [get_aml_run_from_run_id_args(args, workspace)]
    elif run_id_source == AzureRunIdSource.RUN_IDS:
        runs = get_aml_runs_from_run_ids(args, workspace)
    else:
        raise ValueError(f"Unrecognised RunIdSource: {run_id_source}")
    return [run for run in runs if run is not None]


def get_run_file_names(run: Run, prefix: str = "") -> List[str]:
    """
    Get the remote path to all files for a given Run which optionally start with a given prefix

    :param run: The AML Run to look up associated files for
    :param prefix: The optional prefix to filter Run files by
    :return: A list of paths within the Run's container
    """
    all_files = run.get_file_names()
    return [f for f in all_files if f.startswith(prefix)] if prefix else all_files


def _download_files_from_run(run: Run, output_dir: Path, prefix: str = "", validate_checksum: bool = False) -> None:
    """
    Download all files for a given AML run, where the filenames may optionally start with a given
    prefix.

    :param run: The AML Run to download associated files for
    :param output_dir: Local directory to which the Run files should be downloaded.
    :param prefix: Optional prefix to filter Run files by
    :param validate_checksum: Whether to validate the content from HTTP response
    """
    run_paths = get_run_file_names(run, prefix=prefix)
    if len(run_paths) == 0:
        raise ValueError("No such files were found for this Run.")

    for run_path in run_paths:
        output_path = output_dir / run_path
        _download_file_from_run(run, run_path, output_path, validate_checksum=validate_checksum)


def download_files_from_run_id(run_id: str, output_folder: Path, prefix: str = "",
                               workspace: Optional[Workspace] = None,
                               workspace_config_path: Optional[Path] = None,
                               validate_checksum: bool = False) -> None:
    """
    For a given Azure ML run id, first retrieve the Run, and then download all files, which optionally start
    with a given prefix. E.g. if the Run creates a folder called "outputs", which you wish to download all
    files from, specify prefix="outputs". To download all files associated with the run, leave prefix empty.

    If not running inside AML and neither a workspace nor the config file are provided, the code will try to locate a
    config.json file in any of the parent folders of the current working directory. If that succeeds, that config.json
    file will be used to instantiate the workspace.

    If function is called in a distributed PyTorch training script, the files will only be downloaded once per node
    (i.e, all process where is_local_rank_zero() == True). All processes will exit this function once all downloads
    are completed.

    :param run_id: The id of the Azure ML Run
    :param output_folder: Local directory to which the Run files should be downloaded.
    :param prefix: Optional prefix to filter Run files by
    :param workspace: Optional Azure ML Workspace object
    :param workspace_config_path: Optional path to settings for Azure ML Workspace
    :param validate_checksum: Whether to validate the content from HTTP response
    """
    workspace = get_workspace(aml_workspace=workspace, workspace_config_path=workspace_config_path)
    run = get_aml_run_from_run_id(run_id, aml_workspace=workspace)
    _download_files_from_run(run, output_folder, prefix=prefix, validate_checksum=validate_checksum)
    torch_barrier()


def _download_file_from_run(run: Run, filename: str, output_file: Path, validate_checksum: bool = False
                            ) -> Optional[Path]:
    """
    Download a single file from an Azure ML Run, optionally validating the content to ensure the file is not
    corrupted during download. If running inside a distributed setting, will only attempt to download the file
    onto the node with local_rank==0. This prevents multiple processes on the same node from trying to download
    the same file, which can lead to errors.

    :param run: The AML Run to download associated file for
    :param filename: The name of the file as it exists in Azure storage
    :param output_file: Local path to which the file should be downloaded
    :param validate_checksum: Whether to validate the content from HTTP response
    :return: The path to the downloaded file if local rank is zero, else None
    """
    if not is_local_rank_zero():
        return None

    run.download_file(filename, output_file_path=str(output_file), _validate_checksum=validate_checksum)
    return output_file


def is_global_rank_zero() -> bool:
    """
    Tries to guess if the current process is running as DDP rank zero, before the training has actually started,
    by looking at environment variables.

    :return: True if the current process is global rank 0.
    """
    # When doing multi-node training, this indicates which node the present job is on. This is set in
    # set_environment_variables_for_multi_node
    node_rank = os.getenv(ENV_NODE_RANK, "0")
    return is_local_rank_zero() and node_rank == "0"


def is_local_rank_zero() -> bool:
    """
    Tries to guess if the current process is running as DDP local rank zero (i.e., the process that is responsible for
    GPU 0 on each node).

    :return: True if the current process is local rank 0.
    """
    # The per-node jobs for rank zero do not have any of the rank-related environment variables set. PL will
    # set them only once starting its child processes.
    global_rank = os.getenv(ENV_GLOBAL_RANK)
    local_rank = os.getenv(ENV_LOCAL_RANK)
    return global_rank is None and local_rank is None


def download_from_datastore(datastore_name: str, file_prefix: str, output_folder: Path,
                            aml_workspace: Optional[Workspace] = None,
                            workspace_config_path: Optional[Path] = None,
                            overwrite: bool = False,
                            show_progress: bool = False) -> None:
    """
    Download file(s) from an Azure ML Datastore that are registered within a given Workspace. The path
    to the file(s) to be downloaded, relative to the datastore <datastore_name>, is specified by the parameter
    "prefix".  Azure will search for files within the Datastore whose paths begin with this string.
    If you wish to download multiple files from the same folder, set <prefix> equal to that folder's path
    within the Datastore. If you wish to download a single file, include both the path to the folder it
    resides in, as well as the filename itself. If the relevant file(s) are found, they will be downloaded to
    the folder specified by <output_folder>. If this directory does not already exist, it will be created.
    E.g. if your datastore contains the paths ["foo/bar/1.txt", "foo/bar/2.txt"] and you call this
    function with file_prefix="foo/bar" and output_folder="outputs", you would end up with the
    files ["outputs/foo/bar/1.txt", "outputs/foo/bar/2.txt"]

    If not running inside AML and neither a workspace nor the config file are provided, the code will try to locate a
    config.json file in any of the parent folders of the current working directory. If that succeeds, that config.json
    file will be used to instantiate the workspace.

    :param datastore_name: The name of the Datastore containing the blob to be downloaded. This Datastore itself
        must be an instance of an AzureBlobDatastore.
    :param file_prefix: The prefix to the blob to be downloaded
    :param output_folder: The directory into which the blob should be downloaded
    :param aml_workspace: Optional Azure ML Workspace object
    :param workspace_config_path: Optional path to settings for Azure ML Workspace
    :param overwrite: If True, will overwrite any existing file at the same remote path.
        If False, will skip any duplicate file.
    :param show_progress: If True, will show the progress of the file download
    """
    workspace = get_workspace(aml_workspace=aml_workspace, workspace_config_path=workspace_config_path)
    datastore = workspace.datastores[datastore_name]
    assert isinstance(datastore, AzureBlobDatastore), \
        "Invalid datastore type. Can only download from AzureBlobDatastore"  # for mypy
    datastore.download(str(output_folder), prefix=file_prefix, overwrite=overwrite, show_progress=show_progress)
    logging.info(f"Downloaded data to {str(output_folder)}")


def upload_to_datastore(datastore_name: str, local_data_folder: Path, remote_path: Path,
                        aml_workspace: Optional[Workspace] = None,
                        workspace_config_path: Optional[Path] = None,
                        overwrite: bool = False,
                        show_progress: bool = False) -> None:
    """
    Upload a folder to an Azure ML Datastore that is registered within a given Workspace. Note that this will upload
    all files within the folder, but will not copy the folder itself. E.g. if you specify the local_data_dir="foo/bar"
    and that contains the files ["1.txt", "2.txt"], and you specify the remote_path="baz", you would see the
    following paths uploaded to your Datastore: ["baz/1.txt", "baz/2.txt"]

    If not running inside AML and neither a workspace nor the config file are provided, the code will try to locate a
    config.json file in any of the parent folders of the current working directory. If that succeeds, that config.json
    file will be used to instantiate the workspace.

    :param datastore_name: The name of the Datastore to which the blob should be uploaded. This Datastore itself
        must be an instance of an AzureBlobDatastore
    :param local_data_folder: The path to the local directory containing the data to be uploaded
    :param remote_path: The path to which the blob should be uploaded
    :param aml_workspace: Optional Azure ML Workspace object
    :param workspace_config_path: Optional path to settings for Azure ML Workspace
    :param overwrite: If True, will overwrite any existing file at the same remote path.
        If False, will skip any duplicate files and continue to the next.
    :param show_progress: If True, will show the progress of the file download
    """
    if not local_data_folder.is_dir():
        raise TypeError("local_path must be a directory")

    workspace = get_workspace(aml_workspace=aml_workspace, workspace_config_path=workspace_config_path)
    datastore = workspace.datastores[datastore_name]
    assert isinstance(datastore, AzureBlobDatastore), \
        "Invalid datastore type. Can only upload to AzureBlobDatastore"  # for mypy
    datastore.upload(str(local_data_folder), target_path=str(remote_path), overwrite=overwrite,
                     show_progress=show_progress)
    logging.info(f"Uploaded data to {str(remote_path)}")


def download_checkpoints_from_run_id(run_id: str, checkpoint_dir: str, output_folder: Path,
                                     aml_workspace: Optional[Workspace] = None,
                                     workspace_config_path: Optional[Path] = None) -> None:
    """
    Given an Azure ML run id, download all files from a given checkpoint directory within that run, to
    the path specified by output_path.
    If running in AML, will take the current workspace. Otherwise, if neither aml_workspace nor
    workspace_config_path are provided, will try to locate a config.json file in any of the
    parent folders of the current working directory.

    :param run_id: The id of the run to download checkpoints from
    :param checkpoint_dir: The path to the checkpoints directory within the run files
    :param output_folder: The path to which the checkpoints should be stored
    :param aml_workspace: Optional AML workspace object
    :param workspace_config_path: Optional workspace config file
    """
    workspace = get_workspace(aml_workspace=aml_workspace, workspace_config_path=workspace_config_path)
    download_files_from_run_id(run_id, output_folder, prefix=checkpoint_dir, workspace=workspace,
                               validate_checksum=True)


def is_running_in_azure_ml(aml_run: Run = RUN_CONTEXT) -> bool:
    """
    Returns True if the given run is inside of an AzureML machine, or False if it is on a machine outside AzureML.
    When called without arguments, this functions returns True if the present code is running in AzureML.
    Note that in runs with "compute_target='local'" this function will also return True. Such runs execute outside
    of AzureML, but are able to log all their metrics, etc to an AzureML run.

    :param aml_run: The run to check. If omitted, use the default run in RUN_CONTEXT
    :return: True if the given run is inside of an AzureML machine, or False if it is a machine outside AzureML.
    """
    return hasattr(aml_run, 'experiment')


def torch_barrier() -> None:
    """
    This is a barrier to use in distributed jobs. Use it to make all processes that participate in a distributed
    pytorch job to wait for each other. When torch.distributed is not set up or not found, the function exits
    immediately.
    """
    try:
        import torch
    except ModuleNotFoundError:
        logging.info("Skipping the barrier because PyTorch is not available.")
        return
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
