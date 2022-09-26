import logging
from pathlib import Path
from typing import Optional, Union

from health_azure.utils import (ENV_SERVICE_PRINCIPAL_ID, ENV_SERVICE_PRINCIPAL_PASSWORD, ENV_TENANT_ID, ENV_TENANT_ID, find_file_in_parent_to_pythonpath, get_secret_from_environment)

logger = logging.getLogger(__name__)

from azure.ai.ml import MLClient
from azure.ai.ml.entities import AzureBlobDatastore, Environment, Job, Workspace
from azure.identity import ClientSecretCredential, DeviceCodeCredential
from mlflow.entities import Run as MLFlowRun
from mlflow.tracking import MlflowClient


def get_credential() -> Union[ClientSecretCredential, DeviceCodeCredential]:
    service_principal_id = get_secret_from_environment(ENV_SERVICE_PRINCIPAL_ID, allow_missing=True)
    tenant_id = get_secret_from_environment(ENV_TENANT_ID, allow_missing=True)
    service_principal_password = get_secret_from_environment(ENV_SERVICE_PRINCIPAL_PASSWORD, allow_missing=True)
    if service_principal_id and tenant_id and service_principal_password:
        return ClientSecretCredential(
            tenant_id=tenant_id,
            client_id=service_principal_id,
            client_secret=service_principal_password)
    # try:
    #     credential = DefaultAzureCredential()
    #     # Check if given credential can get token successfully.
    #     credential.get_token("https://management.azure.com/.default")
    # except Exception as ex:
    logging.info(
        "Using interactive login to Azure. To use Service Principal authentication, set the environment "
        f"variables {ENV_SERVICE_PRINCIPAL_ID}, {ENV_SERVICE_PRINCIPAL_PASSWORD}, and {ENV_TENANT_ID}"
    )
    return DeviceCodeCredential()


def get_workspace_client(workspace_config_path: str = None, subscription_id: str = None, resource_group: str = None,
                         workspace_name: str = None) -> MLClient:
        credential = get_credential()
        if workspace_config_path:
            workspace = MLClient.from_config(credential=credential, path=str(workspace_config_path))
        else:
            workspace = MLClient(subscription_id=subscription_id, resource_group=resource_group,
            workspace_name=workspace_name, credential=credential)
        logging.info(f"Logged into AzureML workspace {workspace.workspace_name}")
        return workspace


def retrieve_workspace_from_client(client: MLClient) -> Workspace:
    workspace_name = client.workspace_name
    workspace = client.workspaces.get(workspace_name)
    return workspace


def get_workspace(aml_workspace: Optional[Workspace] = None, workspace_config_path: Optional[PathOrString] = None
    ) -> Workspace:
    """
    Retrieve an Azure ML workspace from one of several places:
      1. If the function has been called during an AML run (i.e. on an Azure agent), returns the associated workspace
      2. If a Workspace object has been provided by the user, return that
      3. If a path to a Workspace config file has been provided, load the workspace according to that.
    If not running inside AML and neither a workspace nor the config file are provided, the code will try to locate a
    config.json file in any of the parent folders of the current working directory. If that succeeds, that config.json
    file will be used to instantiate the workspace.
    :param aml_workspace: If provided this is returned as the AzureML Workspace.
    :param workspace_config_path: If not provided with an AzureML Workspace, then load one given the information in this
        config
    :return: An AzureML workspace.
    """
    # if is_running_in_azure_ml(RUN_CONTEXT):
    #     return RUN_CONTEXT.experiment.workspace

    if aml_workspace:
        return aml_workspace

    if workspace_config_path is None:
        workspace_config_path = find_file_in_parent_to_pythonpath(WORKSPACE_CONFIG_JSON)
        if workspace_config_path:
            logging.info(f"Using the workspace config file {str(workspace_config_path.absolute())}")
        else:
            raise ValueError("No workspace config file given, nor can we find one.")

    if isinstance(workspace_config_path, Path):
        workspace_config_path = str(workspace_config_path)
    elif Path(workspace_config_path).is_file():
        workspace_client = get_workspace_client(workspace_config_path)
        workspace = retrieve_workspace_from_client(workspace_client)
        return workspace

    raise ValueError("Workspace config file does not exist or cannot be read.")


def fetch_job(client: MLClient, run_id: str) -> Job:
    job = client.jobs.get(run_id)
    return job


def get_credential() -> Union[ClientSecretCredential, DeviceCodeCredential]:
    service_principal_id = get_secret_from_environment(ENV_SERVICE_PRINCIPAL_ID, allow_missing=True)
    tenant_id = get_secret_from_environment(ENV_TENANT_ID, allow_missing=True)
    service_principal_password = get_secret_from_environment(ENV_SERVICE_PRINCIPAL_PASSWORD, allow_missing=True)
    if service_principal_id and tenant_id and service_principal_password:
        return ClientSecretCredential(
            tenant_id=tenant_id,
            client_id=service_principal_id,
            client_secret=service_principal_password)
    # try:
    #     credential = DefaultAzureCredential()
    #     # Check if given credential can get token successfully.
    #     credential.get_token("https://management.azure.com/.default")
    # except Exception as ex:
    logging.info(
        "Using interactive login to Azure. To use Service Principal authentication, set the environment "
        f"variables {ENV_SERVICE_PRINCIPAL_ID}, {ENV_SERVICE_PRINCIPAL_PASSWORD}, and {ENV_TENANT_ID}"
    )
    return DeviceCodeCredential()
