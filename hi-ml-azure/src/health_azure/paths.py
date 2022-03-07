#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging
from pathlib import Path

ENVIRONMENT_YAML_FILE_NAME = "environment.yml"

REPO_HIML_FOLDER = "hi-ml"
REPO_HIML_AZURE_FOLDER = "hi-ml-azure"


def is_himl_used_from_git_repo() -> bool:
    """Returns False if HI-ML was installed as a package into site-packages. Returns True if the HI-ML codebase is
    used from a clone of the full git repository.

    :return: False if HI-ML is installed as a package, True if used via source from git.
    """
    health_ml_root = Path(__file__).parent.parent
    logging.debug(f"health_ml root: {health_ml_root}")
    if health_ml_root.parent.stem == "site-packages":
        return False
    himl_root = health_ml_root.parent.parent
    # These two folder are present in the top-level folder of the git repo
    expected_folders = [REPO_HIML_FOLDER, REPO_HIML_AZURE_FOLDER]
    all_folders_exist = all((himl_root / folder).is_dir() for folder in expected_folders)
    if all_folders_exist:
        return True
    raise ValueError(
        "Unable to determine the installation status: Code is not used from site-packages, but the "
        "expected top-level folders are not present?"
    )


def git_repo_root_folder() -> Path:
    """
    Attempts to return the path to the top-level hi-ml repo that contains the hi-ml and hi-ml-azure packages.
    This top level repo will only be present if hi-ml has been installed as a git submodule, or the repo has
    been directly downloaded. Otherwise (e.g.if hi-ml has been installed as a pip package) returns None

    return: Path to the himl root dir if it exists, else None
    """
    if not is_himl_used_from_git_repo():
        raise ValueError("This function can only be used if the HI-ML package is used directly from the git repo.")
    return Path(__file__).parent.parent.parent.parent
