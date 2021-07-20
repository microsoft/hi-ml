#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

"""
Tests for hi-ml.
"""

import logging
import pytest


try:
    from health.azure.aml import submit_to_azure_if_needed  # type: ignore
except ImportError:
    logging.info("using local src")
    from src.health.azure.aml import submit_to_azure_if_needed  # type: ignore

logger = logging.getLogger('test.health.azure')
logger.setLevel(logging.DEBUG)


def test_submit_to_azure_if_needed() -> None:
    """
    Test that submit_to_azure_if_needed can be called.
    """
    with pytest.raises(Exception) as ex:
        submit_to_azure_if_needed(
            workspace_config=None,
            workspace_config_path=None)
    assert "We could not find config.json in:" in str(ex)
