#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

"""
Wrapper functions for running local Python scripts on Azure ML.

See elevate_this.py for a very simple 'hello world' example of use.
"""
import logging
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Generator, List, Optional

from azureml.core import (Experiment, Run, RunConfiguration, ScriptRunConfig,
                          Workspace)
from azureml.core.environment import Environment

from src.health.azure.himl_configs import (SourceConfig, WorkspaceConfig,
                          get_service_principal_auth)

logger = logging.getLogger('health.azure')
logger.setLevel(logging.DEBUG)


def submit_to_azure_if_needed(
        workspace_config: Optional[WorkspaceConfig],
        workspace_config_path: Optional[Path],
        compute_cluster_name: str,
        snapshot_root_directory: Path,
        entry_script: Path,
        conda_environment_file: Path,
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
        conda_environment_file=conda_environment_file,
        entry_script=entry_script,
        script_params=script_params,
        environment_variables=environment_variables)

    # TODO: InnerEye.azure.azure_runner.submit_to_azureml does work here with interupt handlers to kill interupted
    # jobs. We'll do that later if still required.

    run_config = RunConfiguration(
        script=entry_script,
        arguments=source_config.script_params)
    script_run_config = ScriptRunConfig(
        source_directory=str(source_config.snapshot_root_directory),
        run_config=run_config,
        compute_target=workspace.compute_targets[compute_cluster_name],
        environment=Environment.from_conda_specification('simple-env', conda_environment_file))

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
