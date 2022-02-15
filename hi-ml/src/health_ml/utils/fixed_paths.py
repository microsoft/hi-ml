#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from health_azure.utils import PathOrString

ENVIRONMENT_YAML_FILE_NAME = "environment.yml"


def get_environment_yaml_file() -> Path:
    """
    Returns the path where the environment.yml file is located, in the repository root directory.
    The function throws an exception if the file is not found

    :return: The full path to the environment files.
    """
    # The environment file is copied into the package folder in setup.py.
    root_dir = repository_root_directory()
    env = root_dir / ENVIRONMENT_YAML_FILE_NAME
    if not env.exists():
        raise ValueError(f"File {ENVIRONMENT_YAML_FILE_NAME} was not found not found in in the repository root"
                         f"{root_dir}.")
    return env


def repository_root_directory(path: Optional[PathOrString] = None) -> Path:
    """
    Gets the full path to the root directory that holds the present repository.

    :param path: if provided, a relative path to append to the absolute path to the repository root.
    :return: The full path to the repository's root directory, with symlinks resolved if any.
    """
    root = Path.cwd()
    if path:
        full_path = root / path
        assert full_path.exists(), f"Path {full_path} doesn't exist"
        return root / path
    else:
        return root


def himl_root_dir() -> Optional[Path]:
    """
    Attempts to return the path to the top-level hi-ml repo that contains the hi-ml and hi-ml-azure packages.
    This top level repo will only be present if hi-ml has been installed as a git submodule, or the repo has
    been directly downlaoded. Otherwise (e.g.if hi-ml has been installed as a pip package) returns None

    return: Path to the himl root dir if it exists, else None
    """
    health_ml_root = Path(__file__).parent.parent
    print(f"health ml root: {health_ml_root}")
    if health_ml_root.parent.stem == "site-packages":
        return None
    himl_root = health_ml_root.parent.parent.parent
    assert (himl_root / "hi-ml").is_dir() , f"no such dir {himl_root / 'hi-ml'}"
    assert (himl_root / "hi-ml-azure").is_dir(), f"no such dir {himl_root / 'hi-ml-azure'}"
    return himl_root


@dataclass(frozen=True)
class OutputFolderForTests:
    """
    Data class for the output directories for a given test
    """
    root_dir: Path

    def create_file_or_folder_path(self, file_or_folder_name: str) -> Path:
        """
        Creates a full path for the given file or folder name relative to the root directory stored in the present
        object.
        :param file_or_folder_name: Name of file or folder to be created under root_dir
        """
        return self.root_dir / file_or_folder_name

    def make_sub_dir(self, dir_name: str) -> Path:
        """
        Makes a sub directory under root_dir
        :param dir_name: Name of subdirectory to be created.
        """
        sub_dir_path = self.create_file_or_folder_path(dir_name)
        sub_dir_path.mkdir()
        return sub_dir_path
