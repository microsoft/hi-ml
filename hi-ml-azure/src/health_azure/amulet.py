#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

"""This module contains functions for identifying and dealing with cloud jobs that are submitted via the Amulet toolbox.
"""

import logging
import os

from pathlib import Path
from typing import Optional, Set

from .utils import ENV_RANK

# Environment variables used by Amulet
ENV_AMLT_PROJECT_NAME = "AZUREML_ARM_PROJECT_NAME"
ENV_AMLT_INPUT_OUTPUT = "AZURE_ML_INPUT_OUTPUT"
ENV_AMLT_OUTPUT_DIR = "AMLT_OUTPUT_DIR"
ENV_AMLT_SNAPSHOT_DIR = "SNAPSHOT_DIR"
ENV_AMLT_AZ_BATCHAI_DIR = "AZ_BATCHAI_JOB_WORK_DIR"
ENV_AMLT_DATAREFERENCE_DATA = 'AZUREML_DATAREFERENCE_data'
ENV_AMLT_DATAREFERENCE_OUTPUT = "AZUREML_DATAREFERENCE_output"


def _path_from_env(env_name: str) -> Optional[Path]:
    """Reads a path from an environment variable, and returns it as a Path object
    if the variable is set. Returns None if the variable is not set or empty.

    :param env_name: The name of the environment variable to read.
    :return: The path read from the environment variable, or None if the variable is not set."""
    path = os.getenv(env_name, None)
    if path is None or path == "":
        return None
    return Path(path)


def get_amulet_output_dir() -> Optional[Path]:
    return _path_from_env(ENV_AMLT_OUTPUT_DIR)


def get_amulet_data_dir() -> Optional[Path]:
    """Gets the directory where the data for the current Amulet job is stored.

    :return: The directory where the data for the current Amulet job is stored.
        Returns None if the current job is not an Amulet job, or no data container is set.
    """
    return _path_from_env(ENV_AMLT_DATAREFERENCE_DATA)


def get_amulet_aml_working_dir() -> Optional[Path]:
    """
    For a job submitted by Amulet, return the path to the Azure ML working directory (i.e. where Azure ML is storing the
    code for the Run). The environment variable that contains this value differs depending on whether the job is
    distributed or not.

    :return: A string representing the path to the Azure ML working directory if the job is submited by Amulet,
        otherwise an empty string
    """
    snapshot_dir = _path_from_env(ENV_AMLT_SNAPSHOT_DIR)
    if snapshot_dir is not None:
        # A non-distributed job submitted by Amulet
        logging.debug(f"Found {ENV_AMLT_SNAPSHOT_DIR} in env vars: {snapshot_dir}")
        return snapshot_dir
    batchai_dir = _path_from_env(ENV_AMLT_AZ_BATCHAI_DIR)
    if batchai_dir is not None:
        # A distributed job submitted by Amulet
        logging.debug(f"Found {ENV_AMLT_AZ_BATCHAI_DIR} in env vars: {batchai_dir}")
        return batchai_dir
    return None


def get_amulet_keys_not_set() -> Set[str]:
    """
    Check environment variables for a given set associated with Amulet jobs. Returns a set containing any keys
    that are not available
    """
    amulet_keys = {ENV_AMLT_PROJECT_NAME, ENV_AMLT_INPUT_OUTPUT, ENV_AMLT_OUTPUT_DIR}
    return amulet_keys.difference(os.environ)


def is_amulet_job() -> bool:
    """
    Checks whether a given set of environment variables associated with Amulet are available. If not, this is not
    an Amulet job so returns False. Otherwise returns True
    """
    missing_amlt_env_vars = get_amulet_keys_not_set()
    amlt_aml_working_dir = get_amulet_aml_working_dir()
    if len(missing_amlt_env_vars) == 0 and amlt_aml_working_dir is not None:
        return True
    return False


def prepare_amulet_job() -> None:
    """Make all necessary preparations to run the present training script as an Amulet job."""
    # The RANK environment is set by Amulet, but not by AzureML. If set, PyTorch Lightning will think that all
    # processes are running at rank 0 in its `rank_zero_only` decorator, which will cause the logging to fail.
    if ENV_RANK in os.environ:
        logging.info("Removing RANK environment variable.")
        del os.environ[ENV_RANK]
