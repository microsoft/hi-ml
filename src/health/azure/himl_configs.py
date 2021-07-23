#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

"""
Configs for running local Python scripts on Azure ML.
"""
import logging
import os
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from azureml.core import Run
from azureml.core import Workspace
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.authentication import ServicePrincipalAuthentication

DEFAULT_UPLOAD_TIMEOUT_SECONDS: int = 36_000  # 10 Hours
SERVICE_PRINCIPAL_ID = "HIML_APPLICATION_ID"
SERVICE_PRINCIPAL_PASSWORD = "HIML_SERVICE_PRINCIPAL_PASSWORD"
TENANT_ID = "HIML_TENANT_ID"
SUBSCRIPTION_ID = "HIML_SUBSCRIPTION_ID"


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

    def __post_init__(self) -> None:
        if not (self.workspace_name and self.subscription_id and self.resource_group):
            raise ValueError("All three WorkspaceConfig fields must contain values")

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
        run_context = Run.get_context()
        if not hasattr(run_context, 'experiment'):
            service_principal_auth = get_service_principal_auth()
            workspace = Workspace.get(
                name=self.workspace_name,
                auth=service_principal_auth,
                subscription_id=self.subscription_id,
                resource_group=self.resource_group)
        else:
            workspace = run_context.experiment.workspace
        return workspace


def get_service_principal_auth() -> Union[
    InteractiveLoginAuthentication,
    ServicePrincipalAuthentication]:
    """
    Creates a service principal authentication object with the application ID stored in the present object. The
    application key is read from the environment.

    :return: A ServicePrincipalAuthentication object that has the application ID and key or None if the key is not
    present
    """
    service_principal_id = get_secret_from_environment(SERVICE_PRINCIPAL_ID, allow_missing=True)
    tenant_id = get_secret_from_environment(TENANT_ID, allow_missing=True)
    service_principal_password = get_secret_from_environment(SERVICE_PRINCIPAL_PASSWORD, allow_missing=True)
    if service_principal_id and tenant_id and service_principal_password:
        return ServicePrincipalAuthentication(
            tenant_id=tenant_id,
            service_principal_id=service_principal_id,
            service_principal_password=service_principal_password)
    logging.info("Using interactive login to Azure. To use Service Principal authentication")
    return InteractiveLoginAuthentication()


def get_secret_from_environment(name: str, allow_missing: bool = False) -> Optional[str]:
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
    snapshot_root_directory: Path
    entry_script: Path
    conda_environment_file: Path
    script_params: List[str] = field(default_factory=list)
    upload_timeout_seconds: int = DEFAULT_UPLOAD_TIMEOUT_SECONDS
    environment_variables: Optional[Dict[str, str]] = None

    def __post_init__(self) -> None:
        if not self.snapshot_root_directory.is_dir():
            raise ValueError(f"root_folder {self.snapshot_root_directory} is not a directory")
        if not self.entry_script.is_file():
            raise ValueError(f"entry_script {self.entry_script} is not a file")
        if not self.conda_environment_file.is_file():
            raise ValueError(f"conda_environment_file {self.conda_environment_file} is not a file")
