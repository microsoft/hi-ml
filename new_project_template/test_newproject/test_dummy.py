#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------
import pytest

from health_newproject.dummy import dummy


@pytest.mark.gpu
def test_dummy() -> None:
    assert dummy() == 1
