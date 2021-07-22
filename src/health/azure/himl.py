#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

"""
Wrapper functions for running local Python scripts on Azure ML.

See elevate_this.py for a very simple 'hello world' example of use.
"""
from contextlib import contextmanager
import logging
import re
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Generator, List, Optional

from azureml.core import Experiment, Run, RunConfiguration, ScriptRunConfig, Workspace

from himl_configs import get_service_principal_auth, SourceConfig, WorkspaceConfig


logger = logging.getLogger('health.azure')
logger.setLevel(logging.DEBUG)


def submit_to_azure_if_needed(
        workspace_config: Optional[WorkspaceConfig],
        workspace_config_path: Optional[Path],
        compute_target_name: str,
        snapshot_root_directory: Path,
        entry_script: Path,
        conda_environment_file: Path,
        script_params: List[str] = [],
        environment_variables: Optional[Dict[str, str]] = {},
        ignored_folders: List[Path] = []) -> Run:
    """
    Submit a folder to Azure, if needed and run it.

    Use the flag --azureml to submit to AzureML, and leave it out to run locally.

    :param workspace_config: Optional workspace config.
    :param workspace_config_file: Optional path to workspace config file.
    :return: Run object for the submitted AzureML run.
    """
    if "azureml" not in sys.argv:
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

    with append_to_amlignore(ignored_folders):
        # TODO: InnerEye.azure.azure_runner.submit_to_azureml does work here with interupt handlers to kill interupted
        # jobs. We'll do that later if still required.

        # In our existing code:
        # runner.submit_to_azureml calls
        # azure_runner.submit_to_azureml, which calls azure_runner.create_run_config t0 create an 
        # azureml.core.ScriptRunConfig and then calls
        # azure_runner.create_and_submit_experiment, which makes an azureml.core.Experiment and then calls
        # submit on the experiment, passing in the  

        entry_script_relative_path = source_config.entry_script.relative_to(source_config.root_folder).as_posix()
        run_config = RunConfiguration(
            script=entry_script_relative_path,
            arguments=source_config.script_params)
        script_run_config = ScriptRunConfig(
            source_directory=str(source_config.root_folder),
            run_config=run_config,
            compute_target=workspace.compute_targets[compute_target_name])

        # replacing everything apart from a-zA-Z0-9_ with _, and replace multiple _ with a single _.
        experiment_name = re.sub('_+', '_', re.sub(r'\W+', '_', entry_script.stem))
        experiment = Experiment(workspace=workspace, name=experiment_name)

        run: Run = experiment.submit(script_run_config)

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


def main() -> None:
    """
    Handle submit_to_azure if called from the command line.
    """
    parser = ArgumentParser()
    parser.add_argument("-w", "--workspace_name", type=str, required=False,
                        help="Azure ML workspace name")
    parser.add_argument("-s", "--subscription_id", type=str, required=False,
                        help="AzureML subscription id")
    parser.add_argument("-r", "--resource_group", type=str, required=False,
                        help="AzureML resource group")
    parser.add_argument("-p", "--workspace_config_path", type=str, required=False,
                        help="AzureML workspace config file")

    args = parser.parse_args()
    config = WorkspaceConfig(
        workspace_name=args.workspace_name,
        subscription_id=args.subscription_id,
        resource_group=args.resource_group)

    submit_to_azure_if_needed(
        config,
        args.workspace_config_path)


if __name__ == "__main__":
    main()
