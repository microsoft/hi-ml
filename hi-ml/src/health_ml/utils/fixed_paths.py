#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from dataclasses import dataclass
from pathlib import Path
from health_azure.paths import git_repo_root_folder, is_himl_used_from_git_repo


def repository_root_directory() -> Path:
    """
    Gets the full path to the root directory that holds the present repository.
    This function should only be called if the repository is available (for example, in unit tests).
    It will raise a ValueError if the repo is not available.

    :param path: if provided, a relative path to append to the absolute path to the repository root.
    :return: The full path to the repository's root directory, with symlinks resolved if any.
    """
    if is_himl_used_from_git_repo():
        return git_repo_root_folder()
    raise ValueError("This function should not be used if hi-ml is used as an installed package.")


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
