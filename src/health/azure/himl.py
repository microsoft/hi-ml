#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

"""
Wrapper functions for running local Python scripts on Azure ML.

See elevate_this.py for a very simple 'hello world' example of use.
"""
import hashlib
import logging
import re
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import conda_merge
import ruamel.yaml
from azureml.core import (Experiment, Run, RunConfiguration, ScriptRunConfig,
                          Workspace)
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.environment import Environment
from health.azure.himl_configs import (SourceConfig, WorkspaceConfig,
                                       get_service_principal_auth)

logger = logging.getLogger('health.azure')
logger.setLevel(logging.DEBUG)

# The version to use when creating an AzureML Python environment. We create all environments with a unique hashed name,
# hence version will always be fixed
ENVIRONMENT_VERSION = "1"


def submit_to_azure_if_needed(
        workspace_config: Optional[WorkspaceConfig],
        workspace_config_path: Optional[Path],
        compute_cluster_name: str,
        snapshot_root_directory: Path,
        entry_script: Path,
        conda_environment_files: List[Path],
        script_params: List[str] = [],
        environment_variables: Dict[str, str] = {},
        ignored_folders: List[Path] = []) -> Run:
    """
    Submit a folder to Azure, if needed and run it.

    Use the flag --azureml to submit to AzureML, and leave it out to run locally.

    :param workspace_config: Optional workspace config.
    :param workspace_config_file: Optional path to workspace config file.
    :return: Run object for the submitted AzureML run.
    """
    if all(["azureml" not in arg for arg in sys.argv]):
        logging.info("The flag azureml is not set, and so not submitting to AzureML")
        return
    if workspace_config_path and workspace_config_path.is_file():
        auth = get_service_principal_auth()
        workspace = Workspace.from_config(path=workspace_config_path, auth=auth)
    elif workspace_config:
        workspace = workspace_config.get_workspace()
    else:
        raise ValueError("Cannot glean workspace config from parameters, and so not submitting to AzureML")
    logging.info(f"Loaded: {workspace.name}")

    source_config = SourceConfig(
        snapshot_root_directory=snapshot_root_directory,
        conda_environment_files=conda_environment_files,
        entry_script=entry_script,
        script_params=script_params,
        environment_variables=environment_variables)
    environment = get_or_create_python_environment(workspace, source_config)

    # TODO: InnerEye.azure.azure_runner.submit_to_azureml does work here with interupt handlers to kill interupted
    # jobs. We'll do that later if still required.

    run_config = RunConfiguration(
        script=entry_script,
        arguments=source_config.script_params)
    script_run_config = ScriptRunConfig(
        source_directory=str(source_config.snapshot_root_directory),
        run_config=run_config,
        compute_target=workspace.compute_targets[compute_cluster_name],
        environment=environment)

    # replacing everything apart from a-zA-Z0-9_ with _, and replace multiple _ with a single _.
    experiment_name = re.sub('_+', '_', re.sub(r'\W+', '_', entry_script.stem))
    experiment = Experiment(workspace=workspace, name=experiment_name)

    try:
        if ignored_folders:
            lines_to_append = [dir.name for dir in ignored_folders]
            amlignore = snapshot_root_directory / ".amlignore"
            amlignore_was_file = amlignore.is_file()
            if amlignore_was_file:
                old_contents = amlignore.read_text()
                new_contents = old_contents.splitlines() + lines_to_append
                amlignore.write_text("\n".join(new_contents))
            else:
                amlignore.write_text("\n".join(lines_to_append))
        run: Run = experiment.submit(script_run_config)
    finally:
        if ignored_folders:
            if amlignore_was_file:
                amlignore.write_text(old_contents)
            else:
                amlignore.unlink()

    if source_config.script_params:
        run.set_tags({"commandline_args": " ".join(source_config.script_params)})

    print("\n==============================================================================")
    print(f"Successfully queued new run {run.id} in experiment: {experiment.name}")
    print("Experiment URL: {}".format(experiment.get_portal_url()))
    print("Run URL: {}".format(run.get_portal_url()))
    print("==============================================================================\n")
    return run


def get_or_create_python_environment(
        workspace: Workspace,
        source_config: SourceConfig,
        environment_name: str = "",
        register_environment: bool = False,
        docker_shm_size: str = "",
        pip_extra_index_url: str = "") -> Environment:
    """
    Creates a description for the Python execution environment in AzureML, based on the Conda environment definition
    files that are specified in `source_config`. If such environment with this Conda environment already exists, it is
    retrieved, otherwise created afresh.

    :param workspace: The AzureML workspace, used to name, and possibly register, the environment.
    :param azure_config: azure related configurations to use for model scale-out behaviour
    :param source_config: configurations for model execution, such as name and execution mode
    :param environment_name: If specified, try to retrieve the existing Python environment with this name. If that is
    not found, create one from the Conda files provided. This parameter is meant to be used when running inference for
    an existing model.
    :param register_environment: If True, the Python environment will be registered in the AzureML workspace. If False,
    it will only be created, but not registered. Use this for unit testing.
    :param docker_shm_size: The shared memory in the docker image for the AzureML VMs.
    :param pip_extra_index_url: An additional URL where PIP packages should be loaded from.
    """
    # Merge the project-specific dependencies with the packages that InnerEye itself needs. This should not be necessary
    # if the innereye package is installed. It is necessary when working with an outer project and InnerEye as a git
    # submodule and submitting jobs from the local machine.
    # In case of version conflicts, the package version in the outer project is given priority.
    conda_dependencies, merged_yaml = merge_conda_dependencies(source_config.conda_environment_files)  # type: ignore
    if pip_extra_index_url:
        # When an extra-index-url is supplied, swap the order in which packages are searched for.
        # This is necessary if we need to consume packages from extra-index that clash with names of packages on
        # pypi
        conda_dependencies.set_pip_option(f"--index-url {pip_extra_index_url}")
        conda_dependencies.set_pip_option("--extra-index-url https://pypi.org/simple")
    env_variables = {
        "AZUREML_OUTPUT_UPLOAD_TIMEOUT_SEC": str(source_config.upload_timeout_seconds),
        # Occasionally uploading data during the run takes too long, and makes the job fail. Default is 300.
        "AZUREML_RUN_KILL_SIGNAL_TIMEOUT_SEC": "900",
        "MKL_SERVICE_FORCE_INTEL": "1",
        # Switching to a new software stack in AML for mounting datasets
        "RSLEX_DIRECT_VOLUME_MOUNT": "true",
        "RSLEX_DIRECT_VOLUME_MOUNT_MAX_CACHE_SIZE": "1",
        **(source_config.environment_variables or {})
    }
    base_image = "mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.2-cudnn8-ubuntu18.04"
    # Create a name for the environment that will likely uniquely identify it. AzureML does hashing on top of that,
    # and will re-use existing environments even if they don't have the same name.
    # Hashing should include everything that can reasonably change. Rely on hashlib here, because the built-in
    # hash function gives different results for the same string in different python instances.
    hash_string = "\n".join([merged_yaml, docker_shm_size, base_image, str(env_variables)])
    sha1 = hashlib.sha1(hash_string.encode("utf8"))
    overall_hash = sha1.hexdigest()[:32]
    unique_env_name = f"InnerEye-{overall_hash}"
    try:
        env_name_to_find = environment_name or unique_env_name
        env = Environment.get(workspace, name=env_name_to_find, version=ENVIRONMENT_VERSION)
        logging.info(f"Using existing Python environment '{env.name}'.")
        return env
    except Exception:
        logging.info(f"Python environment '{unique_env_name}' does not yet exist, creating and registering it.")
    env = Environment(name=unique_env_name)
    env.docker.enabled = True
    env.docker.shm_size = docker_shm_size
    env.python.conda_dependencies = conda_dependencies
    env.docker.base_image = base_image
    env.environment_variables = env_variables
    if register_environment:
        env.register(workspace)
    return env


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
