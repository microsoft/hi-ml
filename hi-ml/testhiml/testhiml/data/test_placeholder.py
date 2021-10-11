#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import pytest

from health_azure.utils import run_duration_string_to_seconds
from health.data.placeholder import placeholder


@pytest.mark.fast
def test_placeholder_fast() -> None:
    """A test placeholder test, fast version"""
    assert placeholder() is True


def test_placeholder() -> None:
    """A test placeholder test"""
    assert placeholder() is True


@pytest.mark.fast
def test_run_duration() -> None:
    """Test the health_azure has been imported correctly."""
    actual = run_duration_string_to_seconds("0.5m")
    assert actual == 30
    assert isinstance(actual, int)
