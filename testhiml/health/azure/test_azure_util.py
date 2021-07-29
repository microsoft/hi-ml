#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
Tests for the functions in health.azure.azure_util
"""
import time
from pathlib import Path
from typing import Optional

import pytest

from health.azure.azure_util import (merge_conda_dependencies, merge_conda_files, run_duration_string_to_seconds,
                                     to_azure_friendly_string)
from testhiml.health.azure.util import repository_root


def test_to_azure_friendly_string() -> None:
    """
    Tests the to_azure_friendly_string function which should replace everything apart from a-zA-Z0-9_ with _, and
    replace multiple _ with a single _
    """
    bad_string = "full__0f_r*bb%sh"
    good_version = to_azure_friendly_string(bad_string)
    assert good_version == "full_0f_r_bb_sh"
    good_string = "Not_Full_0f_Rubbish"
    good_version = to_azure_friendly_string(good_string)
    assert good_version == good_string


def test_merge_conda(random_folder: Path) -> None:
    """
    Tests the logic for merging Conda environment files.
    """
    env1 = """
channels:
  - defaults
  - pytorch
dependencies:
  - conda1=1.0
  - conda2=2.0
  - conda_both=3.0
  - pip:
      - azureml-sdk==1.7.0
      - foo==1.0
"""
    env2 = """
channels:
  - defaults
dependencies:
  - conda1=1.1
  - conda_both=3.0
  - pip:
      - azureml-sdk==1.6.0
      - bar==2.0
"""
    # Spurious test failures on Linux build agents, saying that they can't write the file. Wait a bit.
    time.sleep(0.5)
    file1 = random_folder / "env1.yml"
    file1.write_text(env1)
    file2 = random_folder / "env2.yml"
    file2.write_text(env2)
    # Spurious test failures on Linux build agents, saying that they can't read the file. Wait a bit.
    time.sleep(0.5)
    files = [file1, file2]
    merged_file = random_folder / "merged.yml"
    merge_conda_files(files, merged_file)
    assert merged_file.read_text().splitlines() == """channels:
- defaults
- pytorch
dependencies:
- conda1=1.0
- conda1=1.1
- conda2=2.0
- conda_both=3.0
- pip:
  - azureml-sdk==1.6.0
  - azureml-sdk==1.7.0
  - bar==2.0
  - foo==1.0
""".splitlines()
    conda_dep, _ = merge_conda_dependencies(files)
    # We expect to see the union of channels.
    assert list(conda_dep.conda_channels) == ["defaults", "pytorch"]
    # Package version conflicts are not resolved, both versions are retained.
    assert list(conda_dep.conda_packages) == ["conda1=1.0", "conda1=1.1", "conda2=2.0", "conda_both=3.0"]
    assert list(conda_dep.pip_packages) == ["azureml-sdk==1.6.0", "azureml-sdk==1.7.0", "bar==2.0", "foo==1.0"]


@pytest.mark.parametrize(["s", "expected"],
                         [
                             ("1s", 1),
                             ("0.5m", 30),
                             ("1.5h", 90 * 60),
                             ("1.0d", 24 * 3600),
                             ("", None),
                         ])
def test_run_duration(s: str, expected: Optional[float]) -> None:
    actual = run_duration_string_to_seconds(s)
    assert actual == expected
    if expected:
        assert isinstance(actual, int)


def test_run_duration_fails() -> None:
    with pytest.raises(Exception):
        run_duration_string_to_seconds("17b")


def test_repository_root() -> None:
    root = repository_root()
    assert (root / "SECURITY.md").is_file()
