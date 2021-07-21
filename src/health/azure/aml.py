#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

"""
Wrapper functions for running local Python scripts on Azure ML.
"""
import logging
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

from azureml.core import Workspace

from aml_configs import get_service_principal_auth, SourceConfig, WorkspaceConfig

logger = logging.getLogger('health.azure')
logger.setLevel(logging.DEBUG)


def submit_to_azure_if_needed(
        workspace_config: Optional[WorkspaceConfig],
        workspace_config_path: Optional[Path],
        snapshot_directory: Path,
        entry_script: Path,
        conda_environment: Path) -> None:
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
