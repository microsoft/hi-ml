#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
Utility functions for interacting with AzureML runs
"""
import logging
import os
import re
from typing import Optional, Tuple, Union

from azureml.core import Experiment, Run, Workspace, get_run
from azureml.core.authentication import InteractiveLoginAuthentication, ServicePrincipalAuthentication

EXPERIMENT_RUN_SEPARATOR = ":"
DEFAULT_UPLOAD_TIMEOUT_SECONDS: int = 36_000  # 10 Hours
SERVICE_PRINCIPAL_ID = "HIML_SERVICE_PRINCIPAL_ID"
SERVICE_PRINCIPAL_PASSWORD = "HIML_SERVICE_PRINCIPAL_PASSWORD"
TENANT_ID = "HIML_TENANT_ID"
SUBSCRIPTION_ID = "HIML_SUBSCRIPTION_ID"


def create_run_recovery_id(run: Run) -> str:
    """
   Creates an recovery id for a run so it's checkpoints could be recovered for training/testing

   :param run: an instantiated run.
   :return: recovery id for a given run in format: [experiment name]:[run id]
   """
    return str(run.experiment.name + EXPERIMENT_RUN_SEPARATOR + run.id)


def split_recovery_id(id: str) -> Tuple[str, str]:
    """
    Splits a run ID into the experiment name and the actual run.
    The argument can be in the format 'experiment_name:run_id',
    or just a run ID like user_branch_abcde12_123. In the latter case, everything before the last
    two alphanumeric parts is assumed to be the experiment name.
    :param id:
    :return: experiment name and run name
    """
    components = id.strip().split(EXPERIMENT_RUN_SEPARATOR)
    if len(components) > 2:
        raise ValueError("recovery_id must be in the format: 'experiment_name:run_id', but got: {}".format(id))
    elif len(components) == 2:
        return components[0], components[1]
    else:
        recovery_id_regex = r"^(\w+)_\d+_[0-9a-f]+$|^(\w+)_\d+$"
        match = re.match(recovery_id_regex, id)
        if not match:
            raise ValueError("The recovery ID was not in the expected format: {}".format(id))
        return (match.group(1) or match.group(2)), id


def fetch_run(workspace: Workspace, run_recovery_id: str) -> Run:
    """
    Finds an existing run in an experiment, based on a recovery ID that contains the experiment ID and the actual RunId.
    The run can be specified either in the experiment_name:run_id format, or just the run_id.
    :param workspace: the configured AzureML workspace to search for the experiment.
    :param run_recovery_id: The Run to find. Either in the full recovery ID format, experiment_name:run_id or just the
    run_id
    :return: The AzureML run.
    """
    experiment, run = split_recovery_id(run_recovery_id)
    try:
        experiment_to_recover = Experiment(workspace, experiment)
    except Exception as ex:
        raise Exception(f"Unable to retrieve run {run} in experiment {experiment}: {str(ex)}")
    run_to_recover = fetch_run_for_experiment(experiment_to_recover, run)
    logging.info("Fetched run #{} {} from experiment {}.".format(run, run_to_recover.number, experiment))
    return run_to_recover


def is_running_on_azure_agent() -> bool:
    """
    Returns True if the code appears to be running on an Azure build agent, and False otherwise.
    """
    # Guess by looking at the AGENT_OS variable, that all Azure hosted agents define.
    return bool(os.environ.get("AGENT_OS", None))


def fetch_run_for_experiment(experiment_to_recover: Experiment, run_id: str) -> Run:
    """
    :param experiment_to_recover: an experiment
    :param run_id: a string representing the Run ID of one of the runs of the experiment
    :return: the run matching run_id_or_number; raises an exception if not found
    """
    try:
        return get_run(experiment=experiment_to_recover, run_id=run_id, rehydrate=True)
    except Exception:
        available_runs = experiment_to_recover.get_runs()
        available_ids = ", ".join([run.id for run in available_runs])
        raise (Exception(
            "Run {} not found for experiment: {}. Available runs are: {}".format(
                run_id, experiment_to_recover.name, available_ids)))


def get_authentication() -> Union[InteractiveLoginAuthentication, ServicePrincipalAuthentication]:
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


def to_azure_friendly_string(x: Optional[str]) -> Optional[str]:
    """
    Given a string, ensure it can be used in Azure by replacing everything apart from a-zA-Z0-9_ with _, and replace
    multiple _ with a single _.
    """
    if x is None:
        return x
    else:
        return re.sub('_+', '_', re.sub(r'\W+', '_', x))
