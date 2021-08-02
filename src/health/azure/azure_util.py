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
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import conda_merge
import ruamel.yaml
from azureml.core import Environment, Experiment, Run, Workspace, get_run
from azureml.core.authentication import InteractiveLoginAuthentication, ServicePrincipalAuthentication
from azureml.core.conda_dependencies import CondaDependencies

EXPERIMENT_RUN_SEPARATOR = ":"
DEFAULT_UPLOAD_TIMEOUT_SECONDS: int = 36_000  # 10 Hours
SERVICE_PRINCIPAL_ID = "HIML_SERVICE_PRINCIPAL_ID"
SERVICE_PRINCIPAL_PASSWORD = "HIML_SERVICE_PRINCIPAL_PASSWORD"
TENANT_ID = "HIML_TENANT_ID"
RESOURCE_GROUP = "HIML_RESOURCE_GROUP"
SUBSCRIPTION_ID = "HIML_SUBSCRIPTION_ID"
WORKSPACE_NAME = "HIML_WORKSPACE_NAME"
# The version to use when creating an AzureML Python environment. We create all environments with a unique hashed
# name, hence version will always be fixed
ENVIRONMENT_VERSION = "1"
DEFAULT_DOCKER_BASE_IMAGE = "mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.2-cudnn8-ubuntu18.04"
DEFAULT_DOCKER_SHM_SIZE = "10g"

# Environment variables used for multi-node training
ENV_AZ_BATCHAI_MPI_MASTER_NODE = "AZ_BATCHAI_MPI_MASTER_NODE"
ENV_MASTER_ADDR = "MASTER_ADDR"
ENV_MASTER_IP = "MASTER_IP"
ENV_MASTER_PORT = "MASTER_PORT"
ENV_OMPI_COMM_WORLD_RANK = "OMPI_COMM_WORLD_RANK"
ENV_NODE_RANK = "NODE_RANK"
ENV_GLOBAL_RANK = "GLOBAL_RANK"
ENV_LOCAL_RANK = "LOCAL_RANK"


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
    :param id:
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
    :param run_recovery_id: The Run to find. Either in the full recovery ID format, experiment_name:run_id or just the
    run_id
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


def is_running_on_azure_agent() -> bool:
    """
    Returns True if the code appears to be running on an Azure build agent, and False otherwise.
    """
    # Guess by looking at the AGENT_OS variable, that all Azure hosted agents define.
    return bool(os.environ.get("AGENT_OS", None))


def fetch_run_for_experiment(experiment_to_recover: Experiment, run_id: str) -> Run:
    """
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
    service_principal_id = get_secret_from_environment(SERVICE_PRINCIPAL_ID, allow_missing=True)
    tenant_id = get_secret_from_environment(TENANT_ID, allow_missing=True)
    service_principal_password = get_secret_from_environment(SERVICE_PRINCIPAL_PASSWORD, allow_missing=True)
    if service_principal_id and tenant_id and service_principal_password:
        return ServicePrincipalAuthentication(
            tenant_id=tenant_id,
            service_principal_id=service_principal_id,
            service_principal_password=service_principal_password)
    logging.info("Using interactive login to Azure. To use Service Principal authentication")
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
    secrets = {name: os.environ.get(name, None) for name in [name]}
    if name not in secrets and not allow_missing:
        raise ValueError(f"There is no secret named '{name}' available.")
    value = secrets[name]
    if not value and not allow_missing:
        raise ValueError(f"There is no value stored for the secret named '{name}'")
    return value


def to_azure_friendly_string(x: Optional[str]) -> Optional[str]:
    """
    Given a string, ensure it can be used in Azure by replacing everything apart from a-zA-Z0-9_ with _, and replace
    multiple _ with a single _.
    """
    if x is None:
        return x
    else:
        return re.sub('_+', '_', re.sub(r'\W+', '_', x))


def _log_conda_dependencies_stats(conda: CondaDependencies, message_prefix: str) -> None:
    """
    Writes the number of conda and pip packages to logs.
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


def merge_conda_dependencies(files: List[Path]) -> Tuple[CondaDependencies, str]:
    """
    Creates a CondaDependencies object from the Conda environments specified in one or more files.
    The resulting object contains the union of the Conda and pip packages in the files, where merging
    is done via the conda_merge package.
    :param files: The Conda environment files to read.
    :return: Tuple of (CondaDependencies object that contains packages from all the files,
    string contents of the merge Conda environment)
    """
    for file in files:
        _log_conda_dependencies_stats(CondaDependencies(file), f"Conda environment in {file}")
    temp_merged_file = tempfile.NamedTemporaryFile(delete=False)
    merged_file_path = Path(temp_merged_file.name)
    merge_conda_files(files, result_file=merged_file_path)
    merged_dependencies = CondaDependencies(temp_merged_file.name)
    _log_conda_dependencies_stats(merged_dependencies, "Merged Conda environment")
    merged_file_contents = merged_file_path.read_text()
    temp_merged_file.close()
    return merged_dependencies, merged_file_contents


def merge_conda_files(files: List[Path], result_file: Path) -> None:
    """
    Merges the given Conda environment files using the conda_merge package, and writes the merged file to disk.
    :param files: The Conda environment files to read.
    :param result_file: The location where the merge results should be written.
    """
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
    deps = conda_merge.merge_dependencies(env.get(DEPENDENCIES) for env in env_definitions)
    if deps:
        unified_definition[DEPENDENCIES] = deps
    with result_file.open("w") as f:
        ruamel.yaml.dump(unified_definition, f, indent=2, default_flow_style=False)


def create_python_environment(conda_environment_file: Path,
                              pip_extra_index_url: str,
                              docker_base_image: str,
                              environment_variables: Optional[Dict[str, str]]) -> Environment:
    """
    Creates a description for the Python execution environment in AzureML, based on the Conda environment
    definition files that are specified in `source_config`. If such environment with this Conda environment already
    exists, it is retrieved, otherwise created afresh.
    :param environment_variables: The environment variables that should be set when running in AzureML.
    :param docker_base_image: The Docker base image that should be used when creating a new Docker image.
    :param pip_extra_index_url: If provided, use this PIP package index to find additional packages when building
    the Docker image.
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
    env.docker.base_image = docker_base_image or DEFAULT_DOCKER_BASE_IMAGE
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
