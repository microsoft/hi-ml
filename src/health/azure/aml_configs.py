#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

"""
Configs for running local Python scripts on Azure ML.
"""
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from azureml.core import Run, Workspace
from azureml.core.authentication import (InteractiveLoginAuthentication,
                                         ServicePrincipalAuthentication)

DEFAULT_UPLOAD_TIMEOUT_SECONDS: int = 36_000  # 10 Hours
SERVICE_PRINCIPAL_KEY = "HIML_APPLICATION_KEY"

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


class AzureConfig():
    """
    AzureML related configuration settings
    """
    workspace_config: WorkspaceConfig = ""
    service_principal_auth: str = ""
    tenant_id: str = ""
    _workspace: Optional[Workspace] = None

    def get_workspace(self) -> Workspace:
        """
        Return a workspace object for an existing Azure Machine Learning Workspace. When running inside AzureML, the
        workspace that is retrieved is always the one in the current run context. When running outside AzureML, it is
        created or accessed with the service principal. This function will read the workspace only in the first call to
        this method, subsequent calls will return a cached value.

        Throws an exception if the workspace doesn't exist or the required fields don't lead to a uniquely identifiable
        workspace.

        :returns: Azure Machine Learning Workspace
        """
        if self._workspace:
            return self._workspace
        run_context = Run.get_context()
        if not hasattr(run_context, 'experiment'):
            if self.workspace_config.subscription_id and self.workspace_config.resource_group:
                service_principal_auth = self.get_service_principal_auth()
                self._workspace = Workspace.get(
                    name=self.workspace_config.workspace_name,
                    auth=service_principal_auth,
                    subscription_id=self.workspace_config.subscription_id,
                    resource_group=self.workspace_config.resource_group)
            else:
                raise ValueError("The values for 'subscription_id' and 'resource_group' were not found. "
                                 "Was the Azure setup completed?")
        else:
            self._workspace = run_context.experiment.workspace
        return self._workspace

    def get_service_principal_auth(self) -> Optional[Union[
            InteractiveLoginAuthentication,
            ServicePrincipalAuthentication]]:
        """
        Creates a service principal authentication object with the application ID stored in the present object. The
        application key is read from the environment.

        :return: A ServicePrincipalAuthentication object that has the application ID and key or None if the key is not
        present
        """
        application_key = self.get_secret_from_environment(SERVICE_PRINCIPAL_KEY, allow_missing=True)
        if not application_key:
            logging.info("Using interactive login to Azure. To use Service Principal authentication, "
                         f"supply the password in in environment variable '{SERVICE_PRINCIPAL_KEY}'.")
            return InteractiveLoginAuthentication()
        return ServicePrincipalAuthentication(
            tenant_id=self.tenant_id,
            service_principal_id=self.service_principal_auth,  # TODO: This is "", why pretend otherwise?
            service_principal_password=application_key)

    def get_secret_from_environment(self, name: str, allow_missing: bool = False) -> Optional[str]:
        """
        Gets a password or key from the secrets file or environment variables.

        :param name: The name of the environment variable to read. It will be converted to uppercase.
        :param allow_missing: If true, the function returns None if there is no entry of the given name in any of the
        places searched. If false, missing entries will raise a ValueError.
        :return: Value of the secret. None, if there is no value and allow_missing is True.
        """
        name = name.upper()
        secrets = {name: os.environ.get(name, None) for name in [name]}
        if name not in secrets and not allow_missing:
            raise ValueError(f"There is no secret named '{name}' available.")
        value = secrets[name]
        if not value and not allow_missing:
            raise ValueError(f"There is no value stored for the secret named '{name}'")
        return value


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
    upload_timeout_seconds: int = DEFAULT_UPLOAD_TIMEOUT_SECONDS
    environment_variables: Optional[Dict[str, str]] = None

    def __init__(
            self,
            root_folder: Optional[Path] = None,
            entry_script: Optional[Path] = None,
            conda_dependencies_files: List[Path] = [],
            script_params: List[str] = [],
            environment_variables: Optional[Dict[str, str]] = None) -> None:
        """
        """
        if root_folder:
            self.root_folder = root_folder
        else:
            self.root_folder = Path.cwd()
            print(f"Using {self.root_folder} as the snapshoot root to upload to AzureML, "
                    "since no root_folder argument was given.")
        if entry_script:
            self.entry_script = entry_script
        else:
            self.entry_script = Path(__file__)  # TODO: will this be the path to the calling script or to this file in the package?
            print(f"Using {self.entry_script} as the entry script to upload to AzureML, "
                   "since no entry_script argument was given.")
