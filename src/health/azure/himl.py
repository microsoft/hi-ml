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
from argparse import ArgumentParser
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Generator, List, Optional

from azureml.core import (Experiment, Run, RunConfiguration, ScriptRunConfig,
                          Workspace)
from azureml.core.environment import Environment
from health.azure.himl_configs import (AzureRunInformation, WorkspaceConfig,
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
        # TODO antonsc: Does the snapshot root folder option make sense? Clearly it can't be a folder below the folder
        # where the script lives. But would it ever be a folder further up?
        snapshot_root_directory: Path,
        entry_script: Path,
        conda_environment_file: Path,
        script_params: List[str] = [],
        environment_variables: Dict[str, str] = {},
        ignored_folders: List[Path] = [],
        input_datasets: Optional[List[str]] = None,
        output_datasets: Optional[List[str]] = None) -> Optional[Run]:
    """
    Submit a folder to Azure, if needed and run it.

    Use the flag --azureml to submit to AzureML, and leave it out to run locally.

    :param workspace_config: Optional workspace config.
    :param workspace_config_file: Optional path to workspace config file.
    :return: Run object for the submitted AzureML run.
    """
    if all(["azureml" not in arg for arg in sys.argv]):
        logging.info("The flag azureml is not set, and so not submitting to AzureML")
        return AzureRunInformation(
            input_datasets=[],
            output_datasets=[],
            run=None,
            is_running_in_azure=False,
            output_folder=None,
            log_folder=None
        )
    if workspace_config_path and workspace_config_path.is_file():
        auth = get_service_principal_auth()
        workspace = Workspace.from_config(path=workspace_config_path, auth=auth)
    elif workspace_config:
        workspace = workspace_config.get_workspace()
    else:
        raise ValueError("Cannot glean workspace config from parameters, and so not submitting to AzureML")
    logging.info(f"Loaded: {workspace.name}")

    environment = Environment.from_conda_specification("simple-env", conda_environment_file)

    # TODO: InnerEye.azure.azure_runner.submit_to_azureml does work here with interupt handlers to kill interupted
    # jobs. We'll do that later if still required.

    run_config = RunConfiguration(
        script=entry_script,
        arguments=script_params)
    run_config.environment = environment

    script_run_config = ScriptRunConfig(
        source_directory=str(snapshot_root_directory),
        run_config=run_config,
        compute_target=workspace.compute_targets[compute_cluster_name],
        environment=environment)

    # replacing everything apart from a-zA-Z0-9_ with _, and replace multiple _ with a single _.
    experiment_name = re.sub('_+', '_', re.sub(r'\W+', '_', entry_script.stem))
    experiment = Experiment(workspace=workspace, name=experiment_name)

    with append_to_amlignore(
            amlignore=snapshot_root_directory / ".amlignore",
            lines_to_append=[dir.name for dir in ignored_folders]):
        run: Run = experiment.submit(script_run_config)

    if script_params:
        run.set_tags({"commandline_args": " ".join(script_params)})

    print("\n==============================================================================")
    print(f"Successfully queued new run {run.id} in experiment: {experiment.name}")
    print("Experiment URL: {}".format(experiment.get_portal_url()))
    print("Run URL: {}".format(run.get_portal_url()))
    print("==============================================================================\n")

    return AzureRunInformation(
        input_datasets=[],
        output_datasets=[],
        run=Run.get_context(),
        is_running_in_azure=True,
        output_folder=Path("outputs"),
        log_folder=Path("logs")
    )


@contextmanager
def append_to_amlignore(amlignore: Path, lines_to_append: List[str]) -> Generator:
    """
    Context manager that appends lines to the .amlignore file, and reverts to the previous contents after.
    """
    old_contents = amlignore.read_text() if amlignore.exists() else ""
    new_contents = old_contents.splitlines() + lines_to_append
    amlignore.write_text("\n".join(new_contents))
    yield
    if old_contents:
        amlignore.write_text(old_contents)
    else:
        amlignore.unlink()


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
    if args.workspace_config_path:
        config = None
    else:
        config = WorkspaceConfig(
            workspace_name=args.workspace_name,
            subscription_id=args.subscription_id,
            resource_group=args.resource_group)

    submit_to_azure_if_needed(
        workspace_config=config,
        workspace_config_path=args.workspace_config_path,
        compute_cluster_name=args.compute_cluster_name,
        snapshot_root_directory=args.snapshot_root_directory,
        entry_script=args.entry_script,
        conda_environment_file=args.conda_environment_file)


if __name__ == "__main__":
    main()
