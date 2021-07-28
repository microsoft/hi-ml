#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import json
import logging
import os
from typing import Generator

import pytest

from health.azure.azure_util import RESOURCE_GROUP, SUBSCRIPTION_ID, WORKSPACE_NAME
from health.azure.himl import package_setup_and_hacks
from testhiml.health.azure.util import DEFAULT_WORKSPACE_CONFIG_JSON, repository_root


@pytest.fixture(autouse=True, scope='session')
def test_suite_setup() -> Generator:
    package_setup_and_hacks()
    yield


@pytest.fixture(autouse=True, scope='session')
def check_config_json() -> Generator:
    """
    Check config.json exists. If so, do nothing, otherwise,
    create one using environment variables.
    """
    config_json = repository_root() / DEFAULT_WORKSPACE_CONFIG_JSON
    if config_json.exists():
        yield
    else:
        try:
            logging.info(f"creating {str(config_json)}")

            with open(str(config_json), 'a', encoding="utf-8") as file:
                config = {
                    "subscription_id": os.getenv(SUBSCRIPTION_ID, ""),
                    "resource_group": os.getenv(RESOURCE_GROUP, ""),
                    "workspace_name": os.getenv(WORKSPACE_NAME, "")
                }
                json.dump(config, file)

            yield
        finally:
            if config_json.exists():
                logging.info(f"deleting {str(config_json)}")
                config_json.unlink()
