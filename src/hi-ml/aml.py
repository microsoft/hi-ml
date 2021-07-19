#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

"""
Wrapper functions for running local Python scripts on Azure ML.
"""

import logging

logger = logging.getLogger('hi-ml')
logger.setLevel(logging.DEBUG)


def submit_to_azure_if_needed() -> None:
    """
    Submit a folder to Azure, if needed and run it.
    """
