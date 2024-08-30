import logging
import os
import base64
import json
from typing import Optional, Union

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
    AzureCliCredential,
    InteractiveBrowserCredential,
)


logger = logging.getLogger(__name__)

# Environment variables used for authentication
ENV_SERVICE_PRINCIPAL_ID = "HIML_SERVICE_PRINCIPAL_ID"
ENV_SERVICE_PRINCIPAL_PASSWORD = "HIML_SERVICE_PRINCIPAL_PASSWORD"
ENV_TENANT_ID = "HIML_TENANT_ID"

# This is an environment variable that is set by GitHub Actions, for checking if the code is running in GitHub
ENV_GITHUB_ACTIONS = "GITHUB_ACTIONS"

# The scope for the access tokens that are requested from Azure
ACCESS_TOKEN_SCOPE = "https://management.azure.com/.default"


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
        logger.warning(
            "The use of Service Principal credentials is discouraged because of the risk of password "
            "compromise. Consider switching to OpenID Connect authentication."
        )
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
        _ = auth.get_token(ACCESS_TOKEN_SCOPE)
        logger.info("Successfully started AzureCLI authentication.")
        return auth
    except AuthenticationException as ex:
        # If the code is running in GitHub, there is no point in even trying to authenticate interactively.
        # Raise the exception to get some information about the authentication problem.
        # Otherwise, try to authenticate interactively.
        # The GITHUB_ACTIONS environment variable is meant to be used exactly for this check
        # https://docs.github.com/en/actions/learn-github-actions/variables
        if os.getenv(ENV_GITHUB_ACTIONS, "") == "true":
            raise AuthenticationException("AzureCLI authentication must be set up when running in GitHub") from ex

    logger.info(
        "Using interactive login to Azure. To use Service Principal authentication, set the environment "
        f"variables {ENV_SERVICE_PRINCIPAL_ID}, {ENV_SERVICE_PRINCIPAL_PASSWORD}, and {ENV_TENANT_ID}. "
        "For Azure CLI authentication, log in using 'az login' and ensure that the subscription is set."
    )
    return InteractiveLoginAuthentication()


def _validate_credential(credential: TokenCredential) -> str:
    """
    Validate credential by attempting to get token. If authentication has been successful, get_token
    will succeed. Otherwise an exception will be raised

    :param credential: The credential object to validate.
    :return: The access token (read from AccessToken.token)
    """
    return credential.get_token(ACCESS_TOKEN_SCOPE).token


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


def _get_legitimate_cli_credential() -> TokenCredential:
    """
    Create an AzureCli credential for interacting with Azure resources and validates it.

    :return: A valid Azure credential.
    """
    cred = AzureCliCredential()
    token = _validate_credential(cred)
    object_id = extract_object_id_from_token(token)
    message = object_id[:8] if object_id else "No object ID found"
    logger.info(f"Successfully authenticated with Azure CLI. Object ID (first characters): {message}")
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
        logger.warning(
            "The use of Service Principal credentials is discouraged because of the risk of password "
            "compromise. Consider switching to OpenID Connect authentication."
        )
        logger.info(
            "Found environment variables for Service Principal authentication: First characters of App ID "
            f"are {service_principal_id[:8]}... in tenant {tenant_id[:8]}..."
        )
        return _get_legitimate_service_principal_credential(tenant_id, service_principal_id, service_principal_password)

    try:
        return _get_legitimate_cli_credential()
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


def extract_object_id_from_token(token: str) -> str:
    """Extracts the object ID from an access token.
    The object ID is the unique identifier for the user or service principal in Azure Active Directory.

    :param token: The access token.
    :return: The object ID of the token.
    """
    # This is magic code to disect the token, taken from
    # https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/identity/azure-identity/azure/identity/_internal/decorators.py#L38
    try:
        encoding = "utf-8"
        base64_meta_data = token.split(".")[1].encode(encoding) + b"=="
        json_bytes = base64.decodebytes(base64_meta_data)
        json_string = json_bytes.decode(encoding)
        json_dict = json.loads(json_string)
        return json_dict["oid"]  # type: ignore
    except Exception:
        return ""
