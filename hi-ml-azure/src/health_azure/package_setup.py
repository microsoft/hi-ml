#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import warnings


def package_setup() -> None:
    """
    Set up the Python packages where needed. In particular, reduce the logging level for some of the used
    libraries, which are particularly talkative in DEBUG mode. Usually when running in DEBUG mode, we want
    diagnostics about the model building itself, but not for the underlying libraries.
    """
    # The adal package creates a logging.info line each time it gets an authentication token, avoid that.
    logging.getLogger('adal-python').setLevel(logging.WARNING)
    # Azure core prints full HTTP requests even in INFO mode
    logging.getLogger('azure').setLevel(logging.WARNING)
    # Microsoft Authentication Library msal
    logging.getLogger('msal').setLevel(logging.INFO)
    # PyJWT prints out warnings that are beyond our control
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="jwt")
    # Urllib3 prints out connection information for each call to write metrics, etc
    logging.getLogger('urllib3').setLevel(logging.INFO)
    logging.getLogger('msrest').setLevel(logging.INFO)
    # AzureML prints too many details about logging metrics.
    logging.getLogger('azureml').setLevel(logging.INFO)
    # AzureML prints too many details about uploading and downloading files even at INFO level.
    logging.getLogger('azureml.data').setLevel(logging.WARNING)
