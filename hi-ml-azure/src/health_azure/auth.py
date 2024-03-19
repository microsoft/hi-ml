import logging
import os
from typing import Union, Optional

from azureml.core.authentication import (
    InteractiveLoginAuthentication,
    ServicePrincipalAuthentication,
    AzureCliAuthentication,
)
from azureml.exceptions import AuthenticationException
from azure.core.exceptions import ClientAuthenticationError
from azure.core.credentials import TokenCredential
from azure.identity import (
    ClientSecretCredential,
    DeviceCodeCredential,
    DefaultAzureCredential,
    InteractiveBrowserCredential,
)


logger = logging.getLogger(__name__)

# Environment variables used for authentication
ENV_SERVICE_PRINCIPAL_ID = "HIML_SERVICE_PRINCIPAL_ID"
ENV_SERVICE_PRINCIPAL_PASSWORD = "HIML_SERVICE_PRINCIPAL_PASSWORD"
ENV_TENANT_ID = "HIML_TENANT_ID"


def get_secret_from_environment(name: str, allow_missing: bool = False) -> Optional[str]:
    """
    Gets a password or key from the secrets file or environment variables.

    :param name: The name of the environment variable to read. It will be converted to uppercase.
    :param allow_missing: If true, the function returns None if there is no entry of the given name in any of the
        places searched. If false, missing entries will raise a ValueError.
    :return: Value of the secret. None, if there is no value and allow_missing is True.
    """
    name = name.upper()
    value = os.environ.get(name, None)
    if not value and not allow_missing:
        raise ValueError(f"There is no value stored for the secret named '{name}'")
    return value


def get_authentication() -> (
    Union[AzureCliAuthentication, InteractiveLoginAuthentication, ServicePrincipalAuthentication]
):
    """
    Creates an authentication object to use with AzureML SDK v1 operation. It first tries to create a service principal
    authentication object, initialized from environment variables. If that is not possible, try Azure CLI
    authentication, and if that is also not possible because the user is not logged in, try interactive authentication.

    :return: An authentication objects to use with AzureML SDK v1 operations.
    """
    service_principal_id = get_secret_from_environment(ENV_SERVICE_PRINCIPAL_ID, allow_missing=True)
    tenant_id = get_secret_from_environment(ENV_TENANT_ID, allow_missing=True)
    service_principal_password = get_secret_from_environment(ENV_SERVICE_PRINCIPAL_PASSWORD, allow_missing=True)
    # Check if all 3 environment variables are set
    if service_principal_id and tenant_id and service_principal_password:
        logger.info(
            "Found environment variables for Service Principal authentication: First characters of App ID "
            f"are {service_principal_id[:8]}... in tenant {tenant_id[:8]}..."
        )
        return ServicePrincipalAuthentication(
            tenant_id=tenant_id,
            service_principal_id=service_principal_id,
            service_principal_password=service_principal_password,
        )
    try:
        logger.debug("Trying to authenticate using Azure CLI")
        auth = AzureCliAuthentication()
        _ = auth.get_token()
        logger.info("Successfully started AzureCLI authentication.")
        return auth
    except AuthenticationException:
        pass

    logger.info(
        "Using interactive login to Azure. To use Service Principal authentication, set the environment "
        f"variables {ENV_SERVICE_PRINCIPAL_ID}, {ENV_SERVICE_PRINCIPAL_PASSWORD}, and {ENV_TENANT_ID}. "
        "For Azure CLI authentication, log in using 'az login' and ensure that the subscription is set."
    )
    return InteractiveLoginAuthentication()


def _validate_credential(credential: TokenCredential) -> None:
    """
    Validate credential by attempting to get token. If authentication has been successful, get_token
    will succeed. Otherwise an exception will be raised

    :param credential: The credential object to validate.
    """
    credential.get_token("https://management.azure.com/.default")


def _get_legitimate_service_principal_credential(
    tenant_id: str, service_principal_id: str, service_principal_password: str
) -> TokenCredential:
    """
    Create a ClientSecretCredential given a tenant id, service principal id and password

    :param tenant_id: The Azure tenant id.
    :param service_principal_id: The id of an existing Service Principal.
    :param service_principal_password: The password of an existing Service Principal.
    :raises ValueError: If the credential cannot be validated (i.e. authentication was unsucessful).
    :return: The validated credential.
    """
    cred = ClientSecretCredential(
        tenant_id=tenant_id, client_id=service_principal_id, client_secret=service_principal_password
    )
    try:
        _validate_credential(cred)
        return cred
    except ClientAuthenticationError as e:
        raise ValueError(
            f"Found environment variables for {ENV_SERVICE_PRINCIPAL_ID}, "
            f"{ENV_SERVICE_PRINCIPAL_PASSWORD}, and {ENV_TENANT_ID} but was "
            f"not able to authenticate"
        ) from e


def _get_legitimate_device_code_credential() -> Optional[TokenCredential]:
    """
    Create a DeviceCodeCredential for interacting with Azure resources. If the credential can't be
    validated, return None.

    :return: A valid Azure credential.
    """
    cred = DeviceCodeCredential(timeout=60)
    try:
        _validate_credential(cred)
        return cred
    except ClientAuthenticationError:
        return None


def _get_legitimate_default_credential() -> Optional[TokenCredential]:
    """
    Create a DefaultAzure credential for interacting with Azure resources and validates it.

    :return: A valid Azure credential.
    """
    cred = DefaultAzureCredential(timeout=60)
    _validate_credential(cred)
    return cred


def _get_legitimate_interactive_browser_credential() -> Optional[TokenCredential]:
    """
    Create an InteractiveBrowser credential for interacting with Azure resources. If the credential can't be
    validated, return None.

    :return: A valid Azure credential.
    """
    cred = InteractiveBrowserCredential(timeout=60)
    try:
        _validate_credential(cred)
        return cred
    except ClientAuthenticationError:
        return None


def get_credential() -> Optional[TokenCredential]:
    """
    Get a credential for authenticating with Azure. There are multiple ways to retrieve a credential.
    If environment variables pertaining to details of a Service Principal are available, those will be used
    to authenticate. If no environment variables exist, and the script is not currently
    running inside of Azure ML or another Azure agent, will attempt to retrieve a credential via a
    device code (which requires the user to visit a link and enter a provided code). If this fails, or if running in
    Azure, DefaultAzureCredential will be used which iterates through a number of possible authentication methods
    including identifying an Azure managed identity, cached credentials from VS code, Azure CLI, Powershell etc.
    Otherwise returns None.

    :return: Any of the aforementioned credentials if available, else None.
    """
    service_principal_id = get_secret_from_environment(ENV_SERVICE_PRINCIPAL_ID, allow_missing=True)
    tenant_id = get_secret_from_environment(ENV_TENANT_ID, allow_missing=True)
    service_principal_password = get_secret_from_environment(ENV_SERVICE_PRINCIPAL_PASSWORD, allow_missing=True)
    if service_principal_id and tenant_id and service_principal_password:
        logger.info(
            "Found environment variables for Service Principal authentication: First characters of App ID "
            f"are {service_principal_id[:8]}... in tenant {tenant_id[:8]}..."
        )
        return _get_legitimate_service_principal_credential(tenant_id, service_principal_id, service_principal_password)

    try:
        cred = _get_legitimate_default_credential()
        if cred is not None:
            return cred
    except ClientAuthenticationError:
        cred = _get_legitimate_device_code_credential()
        if cred is not None:
            return cred

        cred = _get_legitimate_interactive_browser_credential()
        if cred is not None:
            return cred

    raise ValueError(
        "Unable to generate and validate a credential. Please see Azure ML documentation"
        "for instructions on different options to get a credential"
    )
