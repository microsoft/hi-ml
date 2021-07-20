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
from attr import dataclass

from azureml.core import Workspace


logger = logging.getLogger('health.azure')
logger.setLevel(logging.DEBUG)


@dataclass
class WorkspaceConfig:
    workspace_name: str = ""
    subscription_id: str = ""
    resource_group: str = ""


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
    if workspace_config is not None:
        workspace = Workspace.get(
            name=workspace_config.workspace_name,
            subscription_id=workspace_config.subscription_id,
            resource_group=workspace_config.resource_group)
    else:
        workspace = Workspace.from_config(path=workspace_config_path)

    if workspace is None:
        raise ValueError("Unable to get workspace.")

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
