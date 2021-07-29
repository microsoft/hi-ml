#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
Test utility functions for tests in the package.
"""
from pathlib import Path

from azureml.core import Run, Workspace

from health.azure.azure_util import SUBSCRIPTION_ID, fetch_run, get_authentication, get_secret_from_environment
from health.azure.himl import RUN_RECOVERY_FILE

DEFAULT_WORKSPACE_CONFIG_JSON = "config.json"
DEFAULT_DATASTORE = "innereyedatasets"
FALLBACK_SINGLE_RUN = "refs_pull_545_merge:refs_pull_545_merge_1626538212_d2b07afd"

TEST_OUTPUTS_PATH = "outputs"


def repository_root() -> Path:
    """
    Gets the root folder of the git repository.
    """
    return Path(__file__).parent.parent.parent.parent


def default_aml_workspace() -> Workspace:
    """
    Gets the default AzureML workspace that is used for testing.
    """
    config_json = repository_root() / DEFAULT_WORKSPACE_CONFIG_JSON
    if config_json.is_file():
        return Workspace.from_config()
    else:
        subscription_id = get_secret_from_environment(SUBSCRIPTION_ID, allow_missing=False)
        auth = get_authentication()
        return Workspace.get(name="InnerEye-DeepLearning",
                             auth=auth,
                             subscription_id=subscription_id,
                             resource_group="InnerEye-DeepLearning")


DEFAULT_WORKSPACE = default_aml_workspace()


def get_most_recent_run_id(run_recovery_file: Path = Path(RUN_RECOVERY_FILE)) -> str:
    """
    Gets the string name of the most recently executed AzureML run. This is picked up from the `most_recent_run.txt`
    file when running on the cloud.
    :param run_recovery_file: The path of the run recovery file
    :return: The run id
    """
    assert run_recovery_file.is_file(), "When running in cloud builds, this should pick up the ID of a previous \
                                         training run"
    run_id = run_recovery_file.read_text().strip()
    print(f"Read this run ID from file: {run_id}")
    return run_id


def get_most_recent_run(run_recovery_file: Path = Path(RUN_RECOVERY_FILE)) -> Run:
    """
    Gets the name of the most recently executed AzureML run, instantiates that Run object and returns it.
    :param run_recovery_file: The path of the run recovery file
    :return: The run
    """
    run_recovery_id = get_most_recent_run_id(
        run_recovery_file=run_recovery_file)
    return fetch_run(workspace=DEFAULT_WORKSPACE, run_recovery_id=run_recovery_id)
