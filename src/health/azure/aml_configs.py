#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

"""
Configs for running local Python scripts on Azure ML.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class WorkspaceConfig:
    """
    Matches the JSON downloaded as config.json from the overview page for the AzureML workspace in the Azure portal.

    The config.json file contains the following JSON (from https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-environment)
    {
        "subscription_id": "<subscription-id>",
        "resource_group": "<resource-group>",
        "workspace_name": "<workspace-name>"
    }
    """
    workspace_name: str = ""
    subscription_id: str = ""
    resource_group: str = ""


@dataclass
class SourceConfig:
    """
    Contains all information that is required to submit a script to AzureML: Entry script, arguments, and information to
    set up the Python environment inside of the AzureML virtual machine.
    """
    root_folder: Path
    entry_script: Path
    conda_dependencies_files: List[Path]
    script_params: List[str] = field(default_factory=list)
    # hyperdrive_config_func: Optional[Callable[[ScriptRunConfig], HyperDriveConfig]] = None  # TODO: Add back hyperdrive support
    upload_timeout_seconds: int = 36000
    environment_variables: Optional[Dict[str, str]] = None

    # Since we are starting with a line, `submit_to_azure_if_needed`, we do not need the infrastructure for consuming or
    # passing on command line arguments yet.
    # def set_script_params_except_submit_flag(self) -> None:
    #     """
    #     Populates the script_param field of the present object from the arguments in sys.argv, with the exception
    #     of the "azureml" flag.
    #     """
    #     self.script_params = remove_arg(AZURECONFIG_SUBMIT_TO_AZUREML, sys.argv[1:])
