from typing import Generator

import pytest

from health.azure.himl import package_setup_and_hacks


@pytest.fixture(autouse=True, scope='session')
def test_suite_setup() -> Generator:
    package_setup_and_hacks()
    yield
