#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

"""
Wrapper functions for running local Python scripts on Azure ML.
"""
import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

from azureml.core import Workspace

from aml_configs import WorkspaceConfig

logger = logging.getLogger('health.azure')
logger.setLevel(logging.DEBUG)


def submit_to_azure_if_needed(
        workspace_config: Optional[WorkspaceConfig],
        workspace_config_path: Optional[Path]) -> None:
    """
    Submit a folder to Azure, if needed and run it.

    :param workspace_config: Optional workspace config.
    :param workspace_config_file: Optional path to workspace config file.
    :return: None.
    """
    workspace: Workspace = None
    if (workspace_config
            and workspace_config.workspace_name
            and workspace_config.subscription_id
            and workspace_config.resource_group):
        workspace = Workspace.get(
            name=workspace_config.workspace_name,
            subscription_id=workspace_config.subscription_id,
            resource_group=workspace_config.resource_group)
    elif workspace_config_path:
        workspace = Workspace.from_config(path=workspace_config_path)
    else:
        print("Cannot glean workspace config from parameters, and so not submitting to AzureML")
        return

    print(f"Loaded: {workspace.name}")


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
