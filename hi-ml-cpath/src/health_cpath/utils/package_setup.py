#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging

from health_ml.utils import package_setup as health_ml_setup


def package_setup() -> None:
    """
    Set up the Python packages where needed. In particular, reduce the logging level for some of the used
    libraries, which are particularly talkative in DEBUG mode. Usually when running in DEBUG mode, we want
    diagnostics about the model building itself, but not for the underlying libraries.

    It also adds workarounds for known issues in some packages.
    """
    health_ml_setup()
    # PIL prints out byte-level information in DEBUG mode
    logging.getLogger('PIL').setLevel(logging.INFO)
