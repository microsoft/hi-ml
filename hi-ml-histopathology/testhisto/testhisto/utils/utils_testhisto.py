#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os
from pathlib import Path
from typing import Any, Collection, Mapping, Optional

import numpy as np
import torch
from PIL import Image

from health_azure.utils import PathOrString


def tests_root_directory(path: Optional[PathOrString] = None) -> Path:
    """
    Gets the full path to the root directory that holds the tests.
    If a relative path is provided then concatenate it with the absolute path
    to the repository root.

    :return: The full path to the repository's root directory, with symlinks resolved if any.
    """
    root = Path(os.path.realpath(__file__)).parent.parent.parent
    return root / path if path else root


def assert_dicts_equal(d1: Mapping, d2: Mapping, exclude_keys: Collection[Any] = (),
                       rtol: float = 1e-5, atol: float = 1e-8) -> None:
    assert isinstance(d1, Mapping)
    assert isinstance(d2, Mapping)
    keys1 = [key for key in d1 if key not in exclude_keys]
    keys2 = [key for key in d2 if key not in exclude_keys]
    assert keys1 == keys2
    for key in keys1:
        msg = f"Dictionaries differ for key '{key}': {d1[key]} vs {d2[key]}"
        if isinstance(d1[key], torch.Tensor):
            assert torch.allclose(d1[key], d2[key], rtol=rtol, atol=atol, equal_nan=True), msg
        elif isinstance(d1[key], np.ndarray):
            assert np.allclose(d1[key], d2[key], rtol=rtol, atol=atol, equal_nan=True), msg
        else:
            assert d1[key] == d2[key], msg


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
        return
    assert False, f"File contents does not match: len(actual)={len(actual)}, len(expected)={len(expected)}"


def full_ml_test_data_path(suffix: str = "") -> Path:
    """
    Returns the path to a folder named "test_data" / <suffix>  in testhisto

    :param suffix: The name of the folder to create in "test_data". If not provided,
    the path to test_data will be returned
    :return: The full absolute path of the directory
    """
    root = Path(os.path.realpath(__file__)).parent.parent.parent
    test_data_dir = root / "test_data"
    return test_data_dir / suffix
