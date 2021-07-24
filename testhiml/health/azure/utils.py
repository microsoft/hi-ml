from pathlib import Path

from azureml.core import Workspace

from health.azure.himl import WORKSPACE_CONFIG_JSON
from health.azure.himl_configs import SUBSCRIPTION_ID, get_authentication, get_secret_from_environment


def repository_root() -> Path:
    """
    Gets the root folder of the git repository.
    """
    return Path(__file__).parent.parent.parent.parent


def default_aml_workspace() -> Workspace:
    """
    Gets the default AzureML workspace that is used for testing.
    """
    config_json = repository_root() / WORKSPACE_CONFIG_JSON
    if config_json.is_file():
        return Workspace.from_config()
    else:
        subscription_id = get_secret_from_environment(SUBSCRIPTION_ID, allow_missing=False)
        auth = get_authentication()
        return Workspace.get(name="InnerEye-DeepLearning",
                             auth=auth,
                             subscription_id=subscription_id,
                             resource_group="InnerEye-DeepLearning")


DEFAULT_DATASTORE = "innereyedatasets"
