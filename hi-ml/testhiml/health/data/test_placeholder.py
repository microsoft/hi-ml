#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import pytest


@pytest.mark.fast
def test_placeholder_fast() -> None:
    assert True


def test_placeholder() -> None:
    assert True
