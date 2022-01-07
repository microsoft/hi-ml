#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from typing import Optional

from health_azure.utils import PathOrString

ENVIRONMENT_YAML_FILE_NAME = "environment.yml"


def get_environment_yaml_file() -> Path:
    """
    Returns the path where the environment.yml file is located. This can be inside of the hi-ml package, or in
    the repository root when working with the code as a submodule.
    The function throws an exception if the file is not found at either of the two possible locations.

    :return: The full path to the environment files.
    """
    # The environment file is copied into the package folder in setup.py.
    env = HI_ML_PACKAGE_ROOT / ENVIRONMENT_YAML_FILE_NAME
    if not env.exists():
        env = repository_root_directory(ENVIRONMENT_YAML_FILE_NAME)
        if not env.exists():
            raise ValueError(f"File {ENVIRONMENT_YAML_FILE_NAME} was not found not found in the package folder "
                             f"{HI_ML_PACKAGE_ROOT}, and not in the repository root {repository_root_directory()}.")
    return env


def repository_root_directory(path: Optional[PathOrString] = None) -> Path:
    """
    Gets the full path to the root directory that holds the present repository.

    :param path: if provided, a relative path to append to the absolute path to the repository root.
    :return: The full path to the repository's root directory, with symlinks resolved if any.
    """
    current = Path(__file__)
    root = current.parent.parent.parent.parent.parent
    if path:
        return root / path
    else:
        return root


HI_ML_PACKAGE_NAME = "health_ml"
HI_ML_PACKAGE_ROOT = repository_root_directory(HI_ML_PACKAGE_NAME)

# The property in the model registry that holds the name of the Python environment
PYTHON_ENVIRONMENT_NAME = "python_environment_name"

SETTINGS_YAML_FILE_NAME = "settings.yml"
SETTINGS_YAML_FILE = HI_ML_PACKAGE_ROOT / SETTINGS_YAML_FILE_NAME
