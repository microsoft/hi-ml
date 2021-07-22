#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

"""
Wrapper functions for running local Python scripts on Azure ML.
"""
from contextlib import contextmanager
import logging
import re
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Generator, List, Optional

from azureml.core import Workspace

from himl_configs import get_service_principal_auth, SourceConfig, WorkspaceConfig


logger = logging.getLogger('health.azure')
logger.setLevel(logging.DEBUG)


def submit_to_azure_if_needed(
        workspace_config: Optional[WorkspaceConfig],
        workspace_config_path: Optional[Path],
        snapshot_root_directory: Path,
        entry_script: Path,
        conda_environment_file: Path,
        script_params: List[str] = [],
        environment_variables: Optional[Dict[str, str]] = {},
        ignored_folders: List[Path] = []) -> None:
    """
    Submit a folder to Azure, if needed and run it.

    Use the flag --azureml to submit to AzureML, and leave it out to run locally.

    :param workspace_config: Optional workspace config.
    :param workspace_config_file: Optional path to workspace config file.
    :return: None.
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
        # replacing everything apart from a-zA-Z0-9_ with _, and replace multiple _ with a single _.
        experiment_name = re.sub('_+', '_', re.sub(r'\W+', '_', entry_script.stem)) 

        # In our existing code:
        # runner.submit_to_azureml calls
        # azure_runner.submit_to_azureml, which creates an azureml.cor.ScriptRunConfig and calls
        # azure_runner.create_and_submit_experiment, which makes an azureml.core.Experiment and calls
        # submit on the experiment, passing in the  

        azure_run = submit_to_azureml(self.azure_config, source_config,
                                        self.lightning_container.all_azure_dataset_ids(),
                                        self.lightning_container.all_dataset_mountpoints())
        # TODO: Where have we told it which cluster to use? We should use 'lite-testing-ds2' for 'Hello World' test with
        # no model training nor inference


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
