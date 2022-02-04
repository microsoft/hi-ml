#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from health_azure.utils import PathOrString


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


def assert_file_exists(file_path: Path) -> None:
    """
    Checks if the given file exists.
    """
    assert file_path.exists(), f"File does not exist: {file_path}"


def assert_binary_files_match(actual_file: Path, expected_file: Path) -> None:
    """
    Checks if two files contain exactly the same bytes. If PNG files mismatch, additional diagnostics is printed.
    """
    # Uncomment this line to batch-update all result files that use this assert function
    # expected_file.write_bytes(actual_file.read_bytes())
    assert_file_exists(actual_file)
    assert_file_exists(expected_file)
    actual = actual_file.read_bytes()
    expected = expected_file.read_bytes()
    if actual == expected:
        return
    if actual_file.suffix == ".png" and expected_file.suffix == ".png":
        actual_image = Image.open(actual_file)
        expected_image = Image.open(expected_file)
        actual_size = actual_image.size
        expected_size = expected_image.size
        assert actual_size == expected_size, f"Image sizes don't match: actual {actual_size}, expected {expected_size}"
        assert np.allclose(np.array(actual_image), np.array(expected_image)), "Image pixel data does not match."
    assert False, f"File contents does not match: len(actual)={len(actual)}, len(expected)={len(expected)}"


def full_ml_test_data_path(suffix: str = "") -> Path:
    """
    Returns the path to a folder named "test_data" / <suffix>  in testhisto

    :param suffix: The name of the folder to create in "test_data". If not provided,
    the path to test_data will be returned
    :return: The full absolute path of the directory
    """
    root = Path(os.path.realpath(__file__)).parent.parent.parent
    test_data_dir = root  / "test_data"
    return test_data_dir / suffix
