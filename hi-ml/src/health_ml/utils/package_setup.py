#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import matplotlib
import os

from health_azure import health_azure_package_setup, set_logging_levels


def health_ml_package_setup() -> None:
    """
    Set up the Python packages where needed. In particular, reduce the logging level for some of the used
    libraries, which are particularly talkative in DEBUG mode. Usually when running in DEBUG mode, we want
    diagnostics about the model building itself, but not for the underlying libraries.

    It also adds workarounds for known issues in some packages.
    """
    health_azure_package_setup()
    module_levels = {
        # DEBUG level info when opening checkpoint and other files
        'fsspec': logging.INFO,
        # Matplotlib is also very talkative in DEBUG mode, filling half of the log file in a PR build.
        'matplotlib': logging.INFO,
        # Jupyter notebook report generation
        'nbconvert': logging.INFO,
        # Numba code generation is extremely talkative in DEBUG mode, disable that.
        'numba': logging.WARNING,
        # PIL prints out byte-level information when loading PNG files in DEBUG mode
        "PIL": logging.INFO,
        # Jupyter notebook report generation
        'papermill': logging.INFO,
    }
    set_logging_levels(module_levels)
    # This is working around a spurious error message thrown by MKL, see
    # https://github.com/pytorch/pytorch/issues/37377
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    # Workaround for issues with matplotlib on some X servers, see
    # https://stackoverflow.com/questions/45993879/matplot-lib-fatal-io-error-25-inappropriate-ioctl-for-device-on-x
    # -server-loc
    matplotlib.use('Agg')
