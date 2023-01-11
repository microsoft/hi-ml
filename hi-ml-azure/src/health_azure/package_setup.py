#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import warnings
from typing import Dict


def set_logging_levels(levels: Dict[str, int]) -> None:
    """Sets the logging levels for the given module-level loggers.

    :param levels: A mapping from module name to desired logging level.
    """
    for module, level in levels.items():
        logging.getLogger(module).setLevel(level)


def health_azure_package_setup() -> None:
    """
    Set up the Python packages where needed. In particular, reduce the logging level for some of the used
    libraries, which are particularly talkative in DEBUG mode. Usually when running in DEBUG mode, we want
    diagnostics about the model building itself, but not for the underlying libraries.
    """
    module_levels = {
        # The adal package creates a logging.info line each time it gets an authentication token, avoid that.
        "adal-python": logging.WARNING,
        # Azure core prints full HTTP requests even in INFO mode
        "azure": logging.WARNING,
        # AzureML prints too many details about logging metrics.
        "azureml": logging.INFO,
        # AzureML prints too many details about uploading and downloading files even at INFO level.
        "azureml.data": logging.WARNING,
        # Microsoft Authentication Library msal
        "msal": logging.INFO,
        "msrest": logging.INFO,
        # Urllib3 prints out connection information for each call to write metrics, etc
        "urllib3": logging.INFO,
    }
    set_logging_levels(module_levels)
    # PyJWT prints out warnings that are beyond our control
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="jwt")
