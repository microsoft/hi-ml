#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

"""
Tests for hi-ml.
"""

import logging

try:
    from hi_ml.aml import submit_to_azure_if_needed  # type: ignore
except ImportError:
    logging.info("using local src")
    from src.hi_ml.aml import submit_to_azure_if_needed  # type: ignore

logger = logging.getLogger('test_hi_ml')
logger.setLevel(logging.DEBUG)


def test_submit_to_azure_if_needed() -> None:
    """
    Test that submit_to_azure_if_needed can be called.
    """
    submit_to_azure_if_needed()

