#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from pathlib import Path
from unittest import mock

import pytest

from health.azure.himl import AzureRunInformation
from health.azure.himl import submit_to_azure_if_needed

logger = logging.getLogger('test.health.azure')
logger.setLevel(logging.DEBUG)


@pytest.mark.fast
def test_submit_to_azure_if_needed() -> None:
    """
    Test that submit_to_azure_if_needed can be called, and returns immediately.
    """
    with mock.patch("sys.argv", [""]):
        result = submit_to_azure_if_needed(entry_script=Path(__file__),
                                           compute_cluster_name="foo",
                                           conda_environment_file=Path("env.yml"),
                                           )
        assert isinstance(result, AzureRunInformation)
        assert not result.is_running_in_azure

