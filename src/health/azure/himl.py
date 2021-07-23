#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

"""
Wrapper functions for running local Python scripts on Azure ML.

See elevate_this.py for a very simple 'hello world' example of use.
"""
import logging
import os
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from typing import Generator
from typing import List
from typing import Optional

from azureml.core import Experiment
from azureml.core import Run
from azureml.core import RunConfiguration
from azureml.core import ScriptRunConfig
from azureml.core import Workspace

from health.azure.datasets import StrOrDatasetConfig
from src.health.azure.himl_configs import SourceConfig
from src.health.azure.himl_configs import WorkspaceConfig
from src.health.azure.himl_configs import get_authentication

logger = logging.getLogger('health.azure')
logger.setLevel(logging.DEBUG)

# Re-use the Run object across the package, to avoid repeated and possibly costly calls to create it.
RUN_CONTEXT = Run.get_context()

WORKSPACE_CONFIG_JSON = "config.json"


@dataclass
class AzureRunInformation:
    input_datasets: List[Path]
    output_datasets: List[Path]
    run: Run
    is_running_in_azure: bool
    # In Azure, this would be the "outputs" folder. In local runs: "." or create a timestamped folder.
    # The folder that we create here must be added to .amlignore
    output_folder: Path
    log_folder: Path


def is_running_in_azure(run: Run = RUN_CONTEXT) -> bool:
    """
    Returns True if the given run is inside of an AzureML machine, or False if it is a machine outside AzureML.
    :param run: The run to check. If omitted, use the default run in RUN_CONTEXT
    :return: True if the given run is inside of an AzureML machine, or False if it is a machine outside AzureML.
    """
    return hasattr(run, 'experiment')


def submit_to_azure_if_needed(
        workspace_config: Optional[WorkspaceConfig],
        workspace_config_path: Optional[Path],
        compute_cluster_name: str,
        snapshot_root_directory: Path,
        entry_script: Path,
        conda_environment_file: Path,
        script_params: List[str] = [],
        environment_variables: Dict[str, str] = {},
        ignored_folders: List[Path] = [],
        default_datastore: str = "",
        input_datasets: Optional[List[StrOrDatasetConfig]] = None,
        output_datasets: Optional[List[StrOrDatasetConfig]] = None,
        num_nodes: int = 1,
        ) -> Run:
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
        auth = get_authentication()
        workspace = Workspace.from_config(path=workspace_config_path, auth=auth)
    elif workspace_config:
        workspace = workspace_config.get_workspace()
    else:
        raise ValueError("Cannot glean workspace config from parameters, and so not submitting to AzureML")
    logging.info(f"Loaded: {workspace.name}")

    source_config = SourceConfig(
        snapshot_root_directory=snapshot_root_directory,
        conda_environment_file=conda_environment_file,
        entry_script=entry_script,
        script_params=script_params,
        environment_variables=environment_variables)

    with append_to_amlignore(
            dirs_to_append=ignored_folders,
            snapshot_root_directory=snapshot_root_directory):
        # TODO: InnerEye.azure.azure_runner.submit_to_azureml does work here with interupt handlers to kill interupted
        # jobs. We'll do that later if still required.

        entry_script_relative_path = \
            source_config.entry_script.relative_to(source_config.snapshot_root_directory).as_posix()
        run_config = RunConfiguration(
            script=entry_script_relative_path,
            arguments=source_config.script_params)
        script_run_config = ScriptRunConfig(
            source_directory=str(source_config.snapshot_root_directory),
            run_config=run_config,
            compute_target=workspace.compute_targets[compute_cluster_name])

        # replacing everything apart from a-zA-Z0-9_ with _, and replace multiple _ with a single _.
        experiment_name = re.sub('_+', '_', re.sub(r'\W+', '_', entry_script.stem))
        experiment = Experiment(workspace=workspace, name=experiment_name)

        run: Run = experiment.submit(script_run_config)

        if source_config.script_params:
            run.set_tags({"commandline_args": " ".join(source_config.script_params)})

        logging.info("\n==============================================================================")
        logging.info(f"Successfully queued new run {run.id} in experiment: {experiment.name}")
        logging.info("Experiment URL: {}".format(experiment.get_portal_url()))
        logging.info("Run URL: {}".format(run.get_portal_url()))
        logging.info("==============================================================================\n")
        return run


@contextmanager
def append_to_amlignore(dirs_to_append: List[Path], snapshot_root_directory: Path) -> Generator:
    """
    Context manager that appends lines to the .amlignore file, and reverts to the previous contents after.
    """
    if dirs_to_append:
        lines_to_append = [dir.name for dir in dirs_to_append]
        amlignore = snapshot_root_directory / ".amlignore"
        if amlignore.is_file():
            old_contents = amlignore.read_text()
            new_contents = old_contents.splitlines() + lines_to_append
            amlignore.write_text("\n".join(new_contents))
            yield
            amlignore.write_text(old_contents)
        else:
            amlignore.write_text("\n".join(lines_to_append))
            yield
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
