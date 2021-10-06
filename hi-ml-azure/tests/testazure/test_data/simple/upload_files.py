#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
Testing run_upload_files.
"""

from pathlib import Path

from azureml.core.run import Run


def init_test(tmp_path: Path) -> None:
    """
    Create test files.

    :param tmp_path: Folder to create test files in.
    """
    pass


def run_test(run: Run) -> None:
    """
    Run a set of tests against run.upload_folder and run_upload_folder.

    :param run: AzureML run.
    """
    pass
