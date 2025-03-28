#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
Tests for the functions in health_azure.azure_util
"""
import json
import logging
import os
import sys
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union
from unittest import mock
from unittest.mock import MagicMock, mock_open, patch
from uuid import uuid4
from xmlrpc.client import Boolean

import conda_merge
import numpy as np
import pandas as pd
import param
import pytest
from _pytest.logging import LogCaptureFixture
from azure.storage.blob import ContainerClient
from azureml._restclient.constants import RunStatus
from azureml.core import Experiment, Run, ScriptRunConfig, Workspace
from azureml.core.run import _OfflineRun
from azureml.core.environment import CondaDependencies
from azure.core.exceptions import ResourceNotFoundError
from azureml.data.azure_storage_datastore import AzureBlobDatastore

import health_azure.utils as util
from health_azure.argparsing import EXPERIMENT_RUN_SEPARATOR, parse_args_and_update_config
from health_azure.himl import AML_IGNORE_FILE, append_to_amlignore, effective_experiment_name
from health_azure.utils import (
    ENV_MASTER_ADDR,
    ENV_MASTER_PORT,
    MASTER_PORT_DEFAULT,
    PackageDependency,
    download_files_by_suffix,
    download_file_if_necessary,
    resolve_workspace_config_path,
    sanitize_snapshoot_directory,
    sanitize_entry_script,
    generate_unique_environment_name,
)
from testazure.test_himl import RunTarget, render_and_run_test_script
from testazure.utils_testazure import (
    DEFAULT_IGNORE_FOLDERS,
    DEFAULT_WORKSPACE,
    MockRun,
    change_working_directory,
    create_unittest_run_object,
    experiment_for_unittests,
    repository_root,
)

RUN_ID = uuid4().hex
RUN_NUMBER = 42
EXPERIMENT_NAME = "fancy-experiment"


def oh_no() -> None:
    """
    Raise a simple exception. To be used as a side_effect for mocks.
    """
    raise ValueError("Throwing an exception")


def test_is_running_in_azureml() -> None:
    """
    Test if the code correctly recognizes that it is executed in AzureML
    """
    # These tests would always run outside of AzureML, on local box or Azure build agents. Function should return
    # False in all those cases
    assert not util.is_running_in_azure_ml()
    assert not util.is_running_in_azure_ml(util.RUN_CONTEXT)
    # When in AzureML, the Run has a field "experiment"
    mock_workspace = "foo"
    with patch("health_azure.utils.RUN_CONTEXT") as mock_run_context:
        mock_run_context.experiment = MagicMock(workspace=mock_workspace)
        # We can't try that with the default argument because of Python's handling of mutable default arguments
        # (default argument value has been assigned already before mocking)
        assert util.is_running_in_azure_ml(util.RUN_CONTEXT)


@patch("health_azure.utils.Run")
def test_create_run_recovery_id(mock_run: MagicMock) -> None:
    """
    The recovery id created for a run
    """
    mock_run.id = RUN_ID
    mock_run.experiment.name = EXPERIMENT_NAME
    recovery_id = util.create_run_recovery_id(mock_run)
    assert recovery_id == EXPERIMENT_NAME + EXPERIMENT_RUN_SEPARATOR + RUN_ID


@patch("health_azure.utils.Workspace")
@patch("health_azure.utils.Experiment")
@patch("health_azure.utils.Run")
def test_fetch_run(mock_run: MagicMock, mock_experiment: MagicMock, mock_workspace: MagicMock) -> None:
    mock_run.id = RUN_ID
    mock_run.experiment = mock_experiment
    mock_experiment.name = EXPERIMENT_NAME
    recovery_id = EXPERIMENT_NAME + EXPERIMENT_RUN_SEPARATOR + RUN_ID
    mock_run.number = RUN_NUMBER
    with mock.patch("health_azure.utils.get_run", return_value=mock_run):
        run_to_recover = util.fetch_run(mock_workspace, recovery_id)
        assert run_to_recover.number == RUN_NUMBER
    mock_experiment.side_effect = oh_no
    with pytest.raises(Exception) as e:
        util.fetch_run(mock_workspace, recovery_id)
    assert str(e.value).startswith(f"Unable to retrieve run {RUN_ID}")


@patch("health_azure.utils.Run")
@patch("health_azure.utils.Experiment")
@patch("health_azure.utils.get_run")
def test_fetch_run_for_experiment(get_run: MagicMock, mock_experiment: MagicMock, mock_run: MagicMock) -> None:
    get_run.side_effect = oh_no
    mock_run.id = RUN_ID
    mock_experiment.get_runs = lambda: [mock_run, mock_run, mock_run]
    mock_experiment.name = EXPERIMENT_NAME
    with pytest.raises(Exception) as e:
        util.fetch_run_for_experiment(mock_experiment, RUN_ID)
    exp = f"Run {RUN_ID} not found for experiment: {EXPERIMENT_NAME}. Available runs are: {RUN_ID}, {RUN_ID}, {RUN_ID}"
    assert str(e.value) == exp


def test_to_azure_friendly_string() -> None:
    """
    Tests the to_azure_friendly_string function which should replace everything apart from a-zA-Z0-9_ with _, and
    replace multiple _ with a single _
    """
    bad_string = "full__0f-r*bb%sh"
    good_version = util.to_azure_friendly_string(bad_string)
    assert good_version == "full_0f-r_bb_sh"
    good_string = "Not_Full_0f-Rubbish"
    good_version = util.to_azure_friendly_string(good_string)
    assert good_version == good_string
    optional_string = None
    assert optional_string == util.to_azure_friendly_string(optional_string)


def test_split_recovery_id_fails() -> None:
    """
    Other tests test the main branch of split_recovery_id, but they do not test the exceptions
    """
    with pytest.raises(ValueError) as e:
        id = EXPERIMENT_RUN_SEPARATOR.join([str(i) for i in range(3)])
        util.split_recovery_id(id)
        assert str(e.value) == f"recovery_id must be in the format: 'experiment_name:run_id', but got: {id}"
    with pytest.raises(ValueError) as e:
        id = "foo_bar"
        util.split_recovery_id(id)
        assert str(e.value) == f"The recovery ID was not in the expected format: {id}"


@pytest.mark.parametrize(
    ["id", "expected1", "expected2"],
    [
        ("foo:bar", "foo", "bar"),
        ("foo:bar_ab_cd", "foo", "bar_ab_cd"),
        ("a_b_c_00_123", "a_b_c", "a_b_c_00_123"),
        ("baz_00_123", "baz", "baz_00_123"),
        ("foo_bar_abc_123_456", "foo_bar_abc", "foo_bar_abc_123_456"),
        # This is the run ID of a hyperdrive parent run. It only has one numeric part at the end
        ("foo_bar_123", "foo_bar", "foo_bar_123"),
        # This is a hyperdrive child run
        ("foo_bar_123_3", "foo_bar", "foo_bar_123_3"),
    ],
)
def test_split_recovery_id(id: str, expected1: str, expected2: str) -> None:
    """
    Check that run recovery ids are correctly parsed into experiment and run id.
    """
    assert util.split_recovery_id(id) == (expected1, expected2)


@pytest.mark.fast
def test_split_dependency() -> None:
    assert util._split_dependency("foo.bar") == ("foo.bar", "", "")
    assert util._split_dependency(" foo.bar == 1.0 ") == ("foo.bar", "==", "1.0")
    assert util._split_dependency("foo.bar>=1.0") == ("foo.bar", ">=", "1.0")
    assert util._split_dependency("foo.bar<=1.0") == ("foo.bar", "<=", "1.0")
    assert util._split_dependency("foo.bar=1.0") == ("foo.bar", "=", "1.0")
    assert util._split_dependency("foo=1.0; platform_system=='Linux'") == (
        "foo",
        "=",
        "1.0",
        ";",
        "platform_system",
        "==",
        "'Linux'",
    )


@pytest.fixture
def dummy_pip_dep_list_one_pinned() -> List[PackageDependency]:
    return [
        PackageDependency("a==0.1"),
        PackageDependency("a>=0.2"),
        PackageDependency("a=0.3"),
        PackageDependency("a=0.4; platform_system='Linux'"),
        PackageDependency("a"),
    ]


@pytest.fixture
def dummy_pip_dep_list_two_pinned() -> List[PackageDependency]:
    return [PackageDependency("b==0.1"), PackageDependency("b==0.2")]


@pytest.fixture
def dummy_pip_dep_list_none_pinned() -> List[PackageDependency]:
    return [PackageDependency("c>=0.1"), PackageDependency("c=0.2"), PackageDependency("c")]


@pytest.mark.fast
def test_resolve_pip_package_clash(
    dummy_pip_dep_list_one_pinned: List[PackageDependency],
    dummy_pip_dep_list_two_pinned: List[PackageDependency],
    dummy_pip_dep_list_none_pinned: List[PackageDependency],
) -> None:
    pin_pip_operator = util.PinnedOperator.PIP
    # if only one pinned version, that should be returned
    expected_keep_dep = PackageDependency("a==0.1")

    keep_dep = util._resolve_package_clash(dummy_pip_dep_list_one_pinned, pin_pip_operator)
    assert keep_dep.package_name == expected_keep_dep.package_name
    assert keep_dep.operator == expected_keep_dep.operator
    assert keep_dep.version == expected_keep_dep.version

    # if two pinned versions are found, a ValueError should be raised
    with pytest.raises(ValueError) as e:
        util._resolve_package_clash(dummy_pip_dep_list_two_pinned, pin_pip_operator)
        assert "Found more than one pinned dependency for package" in str(e)

    # if no pinned package versions are found, a ValueError should be raised
    with pytest.raises(ValueError) as e:
        util._resolve_package_clash(dummy_pip_dep_list_none_pinned, pin_pip_operator)
        assert "Encountered 3 requirements for c, none of which specify a pinned version" in str(e)


@pytest.mark.fast
def test_resolve_pip_dependencies(
    dummy_pip_dep_list_one_pinned: List[PackageDependency],
    dummy_pip_dep_list_two_pinned: List[PackageDependency],
    dummy_pip_dep_list_none_pinned: List[PackageDependency],
) -> None:
    pin_pip_operator = util.PinnedOperator.PIP

    # if only one pinned version, a list containing only that should be returned
    dummy_dependency_dict_one_pinned = {"a": dummy_pip_dep_list_one_pinned}
    resolved_list = util._resolve_dependencies(dummy_dependency_dict_one_pinned, pin_pip_operator)
    assert len(resolved_list) == 1
    resolved_package = resolved_list[0]
    assert isinstance(resolved_package, PackageDependency)
    assert resolved_package.package_name == "a"
    assert resolved_package.operator == "=="
    assert resolved_package.version == "0.1"

    # if two pinned versions are found, a ValueError should be raised
    dummy_dependency_dict_two_pinned = {"b": dummy_pip_dep_list_two_pinned}
    with pytest.raises(ValueError) as e:
        util._resolve_dependencies(dummy_dependency_dict_two_pinned, pin_pip_operator)
        assert "Found more than one pinned dependency for package" in str(e)

    # if no pinned package versions are found, a ValueError should be raised
    dummy_dependency_dict_none_pinned = {"c": dummy_pip_dep_list_none_pinned}
    with pytest.raises(ValueError) as e:
        util._resolve_dependencies(dummy_dependency_dict_none_pinned, pin_pip_operator)
        assert "Encountered 3 requirements for c, none of which specify a pinned version" in str(e)

    # even if one package has exactly one pinned version, if other packages don't, a
    # ValueError should be raised
    dummy_dependency_dict = {"a": dummy_pip_dep_list_one_pinned, "b": dummy_pip_dep_list_two_pinned}
    with pytest.raises(ValueError) as e:
        util._resolve_dependencies(dummy_dependency_dict, pin_pip_operator)
        assert "Found more than one pinned dependency for package" in str(e)


@pytest.mark.fast
def test_retrieve_unique_pip_deps() -> None:
    pin_pip_operator = util.PinnedOperator.PIP
    # if one pinned package is found, that should be retained
    deps_with_one_pinned = [
        "package==1.0",
        "git+https:www.github.com/something.git",
        "foo=1.0; platform_system='Linux'",
    ]
    dedup_deps = util._retrieve_unique_deps(deps_with_one_pinned, pin_pip_operator)  # type: ignore
    assert dedup_deps == [d.replace(" ", "") for d in deps_with_one_pinned]

    # if duplicates are found with more than one pinned, a ValueError should be raised
    deps_with_duplicates = ["package==1.0", "package==1.1", "git+https:www.github.com/something.git"]
    with pytest.raises(ValueError) as e:
        util._retrieve_unique_deps(deps_with_duplicates, pin_pip_operator)
        assert "Found more than one pinned dependency for package" in str(e)

    # if duplicates are found with none pinned, a ValueErorr should be raised
    deps_with_duplicates = ["package>=1.0", "package>1.1", "git+https:www.github.com/something.git"]
    with pytest.raises(ValueError) as e:
        util._retrieve_unique_deps(deps_with_duplicates, pin_pip_operator)
        assert "Encountered 2 requirements for package, none of which specify a pinned version" in str(e)

    # A more complex case
    complex_deps_with_duplicates = ["a==0.1", "b>=0.2", "c", "a>=1.1", "b==1.2", "c>=1.3", "c==2.3"]
    expected_dedup_deps = ["a==0.1", "b==1.2", "c==2.3"]
    dedup_deps = util._retrieve_unique_deps(complex_deps_with_duplicates, pin_pip_operator)  # type: ignore
    assert dedup_deps == expected_dedup_deps


@pytest.fixture
def dummy_conda_dep_list_one_pinned() -> List[PackageDependency]:
    return [PackageDependency("a=0.1"), PackageDependency("a>=0.2"), PackageDependency("a")]


@pytest.fixture
def dummy_conda_dep_list_two_pinned() -> List[PackageDependency]:
    return [PackageDependency("b=0.1"), PackageDependency("b=0.2")]


@pytest.fixture
def dummy_conda_dep_list_none_pinned() -> List[PackageDependency]:
    return [PackageDependency("c>=0.1"), PackageDependency("c")]


@pytest.mark.fast
def test_resolve_conda_package_clash(
    dummy_conda_dep_list_one_pinned: List[PackageDependency],
    dummy_conda_dep_list_two_pinned: List[PackageDependency],
    dummy_conda_dep_list_none_pinned: List[PackageDependency],
) -> None:
    pin_conda_operator = util.PinnedOperator.CONDA
    # if only one pinned version, that should be returned
    expected_keep_dep = PackageDependency("a=0.1")

    keep_dep = util._resolve_package_clash(dummy_conda_dep_list_one_pinned, pin_conda_operator)
    assert keep_dep.package_name == expected_keep_dep.package_name
    assert keep_dep.operator == expected_keep_dep.operator
    assert keep_dep.version == expected_keep_dep.version

    # if two pinned versions are found, a ValueError should be raised
    with pytest.raises(ValueError) as e:
        util._resolve_package_clash(dummy_conda_dep_list_two_pinned, pin_conda_operator)
        assert "Found more than one pinned dependency for package" in str(e)

    # if no pinned package versions are found, a ValueError should be raised
    with pytest.raises(ValueError) as e:
        util._resolve_package_clash(dummy_conda_dep_list_none_pinned, pin_conda_operator)
        assert "Encountered 3 requirements for c, none of which specify a pinned version" in str(e)


@pytest.mark.fast
def test_resolve_conda_dependencies(
    dummy_conda_dep_list_one_pinned: List[PackageDependency],
    dummy_conda_dep_list_two_pinned: List[PackageDependency],
    dummy_conda_dep_list_none_pinned: List[PackageDependency],
) -> None:
    pin_conda_operator = util.PinnedOperator.CONDA

    # if only one pinned version, a list containing only that should be returned
    dummy_dependency_dict_one_pinned = {"a": dummy_conda_dep_list_one_pinned}
    resolved_list = util._resolve_dependencies(dummy_dependency_dict_one_pinned, pin_conda_operator)
    assert len(resolved_list) == 1
    resolved_package = resolved_list[0]
    assert isinstance(resolved_package, PackageDependency)
    assert resolved_package.package_name == "a"
    assert resolved_package.operator == "="
    assert resolved_package.version == "0.1"

    # if two pinned versions are found, a ValueError should be raised
    dummy_dependency_dict_two_pinned = {"b": dummy_conda_dep_list_two_pinned}
    with pytest.raises(ValueError) as e:
        util._resolve_dependencies(dummy_dependency_dict_two_pinned, pin_conda_operator)
        assert "Found more than one pinned dependency for package" in str(e)

    # if no pinned package versions are found, a ValueError should be raised
    dummy_dependency_dict_none_pinned = {"c": dummy_conda_dep_list_none_pinned}
    with pytest.raises(ValueError) as e:
        util._resolve_dependencies(dummy_dependency_dict_none_pinned, pin_conda_operator)
        assert "Encountered 3 requirements for c, none of which specify a pinned version" in str(e)

    # even if one package has exactly one pinned version, if other packages don't, a
    # ValueError should be raised
    dummy_dependency_dict = {"a": dummy_conda_dep_list_one_pinned, "b": dummy_conda_dep_list_two_pinned}
    with pytest.raises(ValueError) as e:
        util._resolve_dependencies(dummy_dependency_dict, pin_conda_operator)
        assert "Found more than one pinned dependency for package" in str(e)


@pytest.mark.fast
def test_retrieve_unique_conda_deps() -> None:
    pin_conda_operator = util.PinnedOperator.CONDA
    # if one pinned package is found, that should be retained
    deps_with_one_pinned = ["package=1.0", "git+https:www.github.com/something.git"]
    dedup_deps = util._retrieve_unique_deps(deps_with_one_pinned, pin_conda_operator)  # type: ignore
    assert dedup_deps == deps_with_one_pinned

    # if duplicates are found with more than one pinned, a ValueError should be raised
    deps_with_duplicates = ["package=1.0", "package=1.1", "git+https:www.github.com/something.git"]
    with pytest.raises(ValueError) as e:
        util._retrieve_unique_deps(deps_with_duplicates, pin_conda_operator)
        assert "Found more than one pinned dependency for package" in str(e)

    # if duplicates are found with none pinned, a ValueErorr should be raised
    deps_with_duplicates = ["package>=1.0", "package>1.1", "git+https:www.github.com/something.git"]
    with pytest.raises(ValueError) as e:
        util._retrieve_unique_deps(deps_with_duplicates, pin_conda_operator)
        assert "Encountered 2 requirements for package, none of which specify a pinned version" in str(e)

    # A more complex case
    complex_deps_with_duplicates = ["a=0.1", "b>=0.2", "c", "a>=1.1", "b=1.2", "c>=1.3", "c=2.3"]
    expected_dedup_deps = ["a=0.1", "b=1.2", "c=2.3"]
    dedup_deps = util._retrieve_unique_deps(complex_deps_with_duplicates, pin_conda_operator)  # type: ignore
    assert dedup_deps == expected_dedup_deps


def _generate_conda_env_lines(channels: List[str], conda_packages: List[str], pip_packages: List[str]) -> List[str]:
    return (
        ["channels:"]
        + [f"- {ch}" for ch in channels]
        + ["dependencies:"]
        + [f"- {cp}" for cp in conda_packages]
        + ["- pip:"]
        + [f"  - {pp}" for pp in pip_packages]
    )


def _generate_conda_env_str(channels: List[str], conda_packages: List[str], pip_packages: List[str]) -> str:
    conda_env_lines = _generate_conda_env_lines(channels, conda_packages, pip_packages)
    return "\n".join(conda_env_lines)


def _create_and_write_env_file(env_definition: str, temp_folder: Path, file_name: str) -> Path:
    """
    Given an environment definition (e.g. Conda or pip), create a file in this location and
    write the definition to it
    """
    file_path = temp_folder / file_name
    file_path.write_text(env_definition)
    return file_path


def assert_pip_length(yaml: Any, expected_length: int) -> None:
    """Checks if the pip dependencies section of a Conda YAML file has the expected number of entries"""
    pip = util._get_pip_dependencies(yaml)
    assert pip is not None
    assert len(pip[1]) == expected_length


@pytest.mark.fast
def test_pip_include_1(tmp_path: Path) -> None:
    """Test if Conda files that use PIP include are handled correctly. This uses the top-level environment.yml
    file in the repository.
    """
    yaml_contents = """name: himl
channels:
  - defaults
dependencies:
  - pip=23.3
  - pip:
      - -r run_requirements.txt
      - some_other_pip_package
"""
    env_file = tmp_path / "environment.yml"
    env_file.write_text(yaml_contents)
    assert env_file.is_file()
    original_yaml = conda_merge.read_file(env_file)
    # The pip section has 2 entries, one that is a reference to a file, and one that is a package.
    assert_pip_length(original_yaml, 2)
    uses_pip_include, modified_yaml = util.is_conda_file_with_pip_include(env_file)
    assert uses_pip_include
    # After filtering out the pip include, the pip section should have only one entry
    pip = util._get_pip_dependencies(modified_yaml)
    assert pip == (1, ["some_other_pip_package"])


@pytest.mark.fast
def test_pip_include_2(tmp_path: Path) -> None:
    """Test if Conda files that use PIP include are recognized."""
    # Environment file without a "-r" include statement
    conda_str = """name: simple-envpip
dependencies:
  - pip:
    - azureml-sdk==1.23.0
  - more_conda
"""
    tmp_conda = tmp_path / "env.yml"
    tmp_conda.write_text(conda_str)
    uses_pip_include, modified_yaml = util.is_conda_file_with_pip_include(tmp_conda)
    assert not uses_pip_include
    assert_pip_length(modified_yaml, 1)

    # Environment file that has a "-r" include statement
    conda_str = """name: simple-env
dependencies:
  - pip:
    - -r foo.txt
    - any_package
"""
    tmp_conda.write_text(conda_str)
    uses_pip_include, modified_yaml = util.is_conda_file_with_pip_include(tmp_conda)
    assert uses_pip_include
    assert util._get_pip_dependencies(modified_yaml) == (0, ["any_package"])


@pytest.mark.parametrize(
    ["s", "expected"],
    [
        ("1s", 1),
        ("0.5m", 30),
        ("1.5h", 90 * 60),
        ("1.0d", 24 * 3600),
        ("", None),
    ],
)  # NOQA
@pytest.mark.fast
def test_run_duration(s: str, expected: Optional[float]) -> None:
    actual = util.run_duration_string_to_seconds(s)
    assert actual == expected
    if expected:
        assert isinstance(actual, int)


@pytest.mark.fast
def test_run_duration_fails() -> None:
    with pytest.raises(Exception):
        util.run_duration_string_to_seconds("17b")


@pytest.mark.fast
def test_repository_root() -> None:
    root = repository_root()
    assert (root / "SECURITY.md").is_file()


def test_nonexisting_amlignore(random_folder: Path) -> None:
    """
    Test that we can create an .AMLignore file, and it gets deleted after use.
    """
    folder1 = "Added1"
    added_folders = [folder1]
    cwd = Path.cwd()
    amlignore = random_folder / AML_IGNORE_FILE
    assert not amlignore.is_file()
    os.chdir(random_folder)
    with append_to_amlignore(added_folders):
        new_contents = amlignore.read_text()
        for f in added_folders:
            assert f in new_contents
    assert not amlignore.is_file()
    os.chdir(cwd)


def test_generate_unique_environment_name_from_description() -> None:
    dummy_env_description_string_1 = "A pretend environment description\ncontaining information about pip "
    "packages\netc etc"
    env_name_1 = util.generate_unique_environment_name(dummy_env_description_string_1)
    assert env_name_1.startswith("HealthML-")

    dummy_env_description_string_2 = "A slightly differetpretend environment description\ncontaining "
    "information about pip packages\netc etc"
    env_name_2 = util.generate_unique_environment_name(dummy_env_description_string_2)
    assert env_name_2.startswith("HealthML-")
    assert env_name_1 != env_name_2


@patch("health_azure.utils.Workspace")
def test_create_python_environment(
    mock_workspace: mock.MagicMock,
    random_folder: Path,
) -> None:
    conda_str = """name: simple-env
dependencies:
  - pip=23.3
  - python=3.9.18
  - pip:
    - azureml-sdk==1.23.0
    - something-else==0.1.5
  - pip:
    - --index-url https://test.pypi.org/simple/
    - --extra-index-url https://pypi.org/simple
    - hi-ml-azure
"""
    conda_environment_file = random_folder / "environment.yml"
    conda_environment_file.write_text(conda_str)
    conda_dependencies = CondaDependencies(conda_dependencies_file_path=conda_environment_file)
    env = util.create_python_environment(conda_environment_file=conda_environment_file)
    assert list(env.python.conda_dependencies.conda_channels) == list(conda_dependencies.conda_channels)
    assert list(env.python.conda_dependencies.conda_packages) == list(conda_dependencies.conda_packages)
    assert list(env.python.conda_dependencies.pip_options) == list(conda_dependencies.pip_options)
    assert list(env.python.conda_dependencies.pip_packages) == list(conda_dependencies.pip_packages)
    # Just check that the environment has a reasonable name. Detailed checks for uniqueness of the name follow below.
    assert env.name.startswith("HealthML")

    pip_extra_index_url = "https://where.great.packages.live/"
    docker_base_image = "viennaglobal.azurecr.io/azureml/azureml_a187a87cc7c31ac4d9f67496bc9c8239"
    env = util.create_python_environment(
        conda_environment_file=conda_environment_file,
        pip_extra_index_url=pip_extra_index_url,
        docker_base_image=docker_base_image,
    )
    assert env.docker.base_image == docker_base_image

    private_pip_wheel_url = "https://some.blob/private/wheel"
    with mock.patch("health_azure.utils.Environment") as mock_environment:
        mock_environment.add_private_pip_wheel.return_value = private_pip_wheel_url
        env = util.create_python_environment(
            conda_environment_file=conda_environment_file,
            workspace=mock_workspace,
            private_pip_wheel_path=Path(__file__),
        )
    envs_pip_packages = list(env.python.conda_dependencies.pip_packages)
    assert "hi-ml-azure" in envs_pip_packages
    assert private_pip_wheel_url in envs_pip_packages


@patch("health_azure.utils.Workspace")
def test_create_python_environment_v2(
    mock_workspace: mock.MagicMock,
    random_folder: Path,
) -> None:
    conda_str = """name: simple-env
dependencies:
  - pip=23.3
  - python=3.9.18
  - pip:
    - azureml-sdk==1.23.0
    - something-else==0.1.5
  - pip:
    - --index-url https://test.pypi.org/simple/
    - --extra-index-url https://pypi.org/simple
    - hi-ml-azure
"""
    conda_environment_file = random_folder / "environment.yml"
    conda_environment_file.write_text(conda_str)
    env = util.create_python_environment_v2(conda_environment_file=conda_environment_file)

    # Check that the environment has a reasonable name. Detailed checks for uniqueness of the name follow below.
    assert env.name.startswith("HealthML")
    assert env.name.endswith("-v2")
    assert env._conda_file_path == conda_environment_file

    pip_extra_index_url = "https://where.great.packages.live/"
    docker_base_image = "viennaglobal.azurecr.io/azureml/azureml_a187a87cc7c31ac4d9f67496bc9c8239"
    env = util.create_python_environment_v2(
        conda_environment_file=conda_environment_file,
        pip_extra_index_url=pip_extra_index_url,
        docker_base_image=docker_base_image,
    )

    assert env.image == docker_base_image


def test_create_environment_unique_name(random_folder: Path) -> None:
    """
    Test if the name of the conda environment changes with each of the components
    """
    conda_str1 = """name: simple-env
dependencies:
  - pip=23.3
  - python=3.9.18
"""
    conda_environment_file = random_folder / "environment.yml"
    conda_environment_file.write_text(conda_str1)
    env1 = util.create_python_environment(conda_environment_file=conda_environment_file)

    # Changing the contents of the conda file should create a new environment names
    conda_str2 = """name: simple-env
dependencies:
  - pip=23.3
"""
    assert conda_str1 != conda_str2
    conda_environment_file.write_text(conda_str2)
    env2 = util.create_python_environment(conda_environment_file=conda_environment_file)
    assert env1.name != env2.name

    # Using a different PIP index URL can lead to different package resolution, so this should change name too
    env3 = util.create_python_environment(conda_environment_file=conda_environment_file, pip_extra_index_url="foo")
    assert env3.name != env2.name

    # Docker base image
    env5 = util.create_python_environment(conda_environment_file=conda_environment_file, docker_base_image="docker")
    assert env5.name != env2.name

    # PIP wheel
    with mock.patch("health_azure.utils.Environment") as mock_environment:
        mock_environment.add_private_pip_wheel.return_value = "private_pip_wheel_url"
        env6 = util.create_python_environment(
            conda_environment_file=conda_environment_file,
            workspace=DEFAULT_WORKSPACE.workspace,
            private_pip_wheel_path=Path(__file__),
        )
        assert env6.name != env2.name

    all_names = [env1.name, env2.name, env3.name, env5.name, env6.name]
    all_names_set = {*all_names}
    assert len(all_names) == len(all_names_set), "Environment names are not unique"


def test_create_environment_wheel_fails(random_folder: Path) -> None:
    """
    Test if all necessary checks are carried out when adding private wheels to an environment.
    """
    conda_str = """name: simple-env
dependencies:
  - pip=23.3
  - python=3.9.18
"""
    conda_environment_file = random_folder / "environment.yml"
    conda_environment_file.write_text(conda_str)
    # Wheel file does not exist at all:
    with pytest.raises(FileNotFoundError) as ex1:
        util.create_python_environment(
            conda_environment_file=conda_environment_file, private_pip_wheel_path=Path("does_not_exist")
        )
        assert "Cannot add private wheel" in str(ex1)
    # Wheel exists, but no workspace provided:
    with pytest.raises(ValueError) as ex2:
        util.create_python_environment(
            conda_environment_file=conda_environment_file, private_pip_wheel_path=Path(__file__)
        )
        assert "AzureML workspace must be provided" in str(ex2)


class MockEnvironment:
    def __init__(self, name: str, version: str = "autosave") -> None:
        self.name = name
        self.version = version


@patch("health_azure.utils.Environment")
@patch("health_azure.utils.Workspace")
def test_register_environment(
    mock_workspace: mock.MagicMock,
    mock_environment: mock.MagicMock,
    caplog: LogCaptureFixture,
) -> None:
    def _mock_env_get(workspace: Workspace, name: str = "", version: Optional[str] = None) -> MockEnvironment:
        if version is None:
            raise Exception("not found")
        return MockEnvironment(name, version=version)

    env_name = "an environment"
    env_version = "environment version"
    mock_environment.get.return_value = mock_environment
    mock_environment.name = env_name
    mock_environment.version = env_version
    with caplog.at_level(logging.INFO):  # type: ignore
        _ = util.register_environment(mock_workspace, mock_environment)
        caplog_text: str = caplog.text  # for mypy
        assert f"Using existing Python environment '{env_name}' with version '{env_version}'" in caplog_text

        # test that log is correct when exception is triggered
        mock_environment.get.side_effect = oh_no
        _ = util.register_environment(mock_workspace, mock_environment)
        caplog_text = caplog.text  # for mypy
        assert (
            f"environment '{env_name}' does not yet exist, creating and registering it with version"
            f" '{env_version}'" in caplog_text
        )

        # test that environment version equals ENVIRONMENT_VERSION when exception is triggered
        # rather than default value of "autosave"
        mock_environment.version = None
        with patch.object(mock_environment, "get", _mock_env_get):
            with patch.object(mock_environment, "register") as mock_register:
                mock_register.return_value = mock_environment
                env = util.register_environment(mock_workspace, mock_environment)
                assert env.version == util.ENVIRONMENT_VERSION


@patch("azure.ai.ml.entities.Environment")
@patch("azure.ai.ml.MLClient")
def test_register_environment_v2(
    mock_ml_client: MagicMock,
    mock_environment_v2: MagicMock,
    caplog: LogCaptureFixture,
) -> None:
    def _mock_cant_find_env(env_name: str, label_or_version: str) -> None:
        raise ResourceNotFoundError("Does not exist")

    env_name = "an environment"
    env_version = "environment version"
    mock_ml_client.environments.get.return_value = mock_environment_v2
    mock_environment_v2.name = env_name
    mock_environment_v2.version = env_version
    with caplog.at_level(logging.INFO):  # type: ignore
        _ = util.register_environment_v2(mock_environment_v2, mock_ml_client)
        caplog_text = caplog.text
        assert f"Found a registered environment with name {env_name}, returning that." in caplog_text

        # test that log is correct when exception is triggered
        mock_ml_client.environments.get.side_effect = _mock_cant_find_env
        _ = util.register_environment_v2(mock_environment_v2, mock_ml_client)
        caplog_text = caplog.text
        assert "Didn't find existing environment. Registering a new one." in caplog_text


def test_set_environment_variables_for_multi_node(caplog: LogCaptureFixture) -> None:
    # If none of AZ_BATCHAI_MPI_MASTER_NODE, AZ_BATCH_MASTER_NODE or ENV_MASTER_IP are set, should assume
    # single node training job
    with caplog.at_level(logging.INFO):  # type: ignore
        util.set_environment_variables_for_multi_node()
        assert "No settings for the MPI central node found" in caplog.messages[-1]  # type: ignore
        assert "Assuming that this is a single node training job" in caplog.text  # type: ignore

    # If all of ENV_MASTER_IP, AZ_BATCHAI_MPI_MASTER_NODE and AZ_BATCH_MASTER_NODE are set, the latter should
    # take precedence. NODE_RANK should get updated to the value of ENV_OMPI_COMM_WORLD_RANK
    port_mock = str(MASTER_PORT_DEFAULT - 1)  # Avoid setting to the default value
    node_rank_mock = "8"

    master_addr_mock = "1234.0.0.0"
    master_node_mock = f"{master_addr_mock}:{port_mock}"

    mpi_master_addr_mock = "5678.9.9.9"
    mpi_master_node_mock = f"{mpi_master_addr_mock}"

    master_ip_mock = "9012.3.3.3"

    env_dict_with_master_var = {
        # mpi_master vars
        util.ENV_AZ_BATCHAI_MPI_MASTER_NODE: mpi_master_node_mock,
        util.ENV_MASTER_ADDR: mpi_master_addr_mock,
        # master vars
        util.ENV_AZ_BATCH_MASTER_NODE: master_node_mock,
        # AKS vars
        util.ENV_MASTER_IP: master_ip_mock,
        # other vars
        util.ENV_OMPI_COMM_WORLD_RANK: node_rank_mock,
        util.ENV_MASTER_PORT: port_mock,
    }

    with caplog.at_level(logging.INFO):  # type: ignore
        with mock.patch.dict(os.environ, env_dict_with_master_var, clear=True):
            util.set_environment_variables_for_multi_node()
            out = caplog.messages[-1]
            assert (
                f"Distributed training: MASTER_ADDR = {master_addr_mock}, MASTER_PORT = "
                f"{port_mock}, NODE_RANK = {node_rank_mock}"
            ) in out
            assert os.environ[ENV_MASTER_ADDR] == master_addr_mock
            assert os.environ[ENV_MASTER_PORT] == port_mock

    # If AZ_BATCH_MASTER_NODE is not set, but AZ_BATCHAI_MPI_MASTER_NODE is, address should be taken from that
    # In this case we expect master address to equal the mpi version, but port and rank will be the same as before
    env_dict_with_mpi_master_var = {
        # mpi_master vars
        util.ENV_AZ_BATCHAI_MPI_MASTER_NODE: mpi_master_node_mock,
        util.ENV_MASTER_ADDR: mpi_master_addr_mock,
        # AKS vars
        util.ENV_MASTER_IP: master_ip_mock,
        # other vars
        util.ENV_OMPI_COMM_WORLD_RANK: node_rank_mock,
        util.ENV_MASTER_PORT: port_mock,
    }

    with caplog.at_level(logging.INFO):  # type: ignore
        with mock.patch.dict(os.environ, env_dict_with_mpi_master_var, clear=True):
            util.set_environment_variables_for_multi_node()
            out = caplog.messages[-1]
            assert (
                f"Distributed training: MASTER_ADDR = {mpi_master_addr_mock}, MASTER_PORT = "
                f"{port_mock}, NODE_RANK = {node_rank_mock}"
            ) in out
            assert os.environ[ENV_MASTER_ADDR] == mpi_master_addr_mock
            assert os.environ[ENV_MASTER_PORT] == port_mock

    # If neither AZ_BATCH_MASTER_NODE nor AZ_BATCHAI_MPI_MASTER_NODE is set, but ENV_MASTER_IP is, the address
    # should be updated with its value
    env_dict_with_env_master_ip_var = {
        # mpi_master vars
        util.ENV_MASTER_ADDR: mpi_master_addr_mock,
        # AKS vars
        util.ENV_MASTER_IP: master_ip_mock,
        # other vars
        util.ENV_OMPI_COMM_WORLD_RANK: node_rank_mock,
        util.ENV_MASTER_PORT: port_mock,
    }

    with caplog.at_level(logging.INFO):  # type: ignore
        with mock.patch.dict(os.environ, env_dict_with_env_master_ip_var, clear=True):
            util.set_environment_variables_for_multi_node()
            out = caplog.messages[-1]
            assert (
                f"Distributed training: MASTER_ADDR = {master_ip_mock}, MASTER_PORT = "
                f"{port_mock}, NODE_RANK = {node_rank_mock}"
            ) in out
            assert os.environ[ENV_MASTER_ADDR] == master_ip_mock
            assert os.environ[ENV_MASTER_PORT] == port_mock

    # If ENV_MASTER_PORT is not set, the default port value should be used
    env_dict_with_mpi_master_var_no_master_port = {
        # mpi_master vars
        util.ENV_AZ_BATCHAI_MPI_MASTER_NODE: mpi_master_node_mock,
        util.ENV_MASTER_ADDR: mpi_master_addr_mock,
        # AKS vars
        util.ENV_MASTER_IP: master_ip_mock,
        # other vars
        util.ENV_OMPI_COMM_WORLD_RANK: node_rank_mock,
    }

    with caplog.at_level(logging.INFO):  # type: ignore
        with mock.patch.dict(os.environ, env_dict_with_mpi_master_var_no_master_port, clear=True):
            util.set_environment_variables_for_multi_node()
            out = caplog.messages[-1]
            assert (
                f"Distributed training: MASTER_ADDR = {mpi_master_addr_mock}, MASTER_PORT = "
                f"{MASTER_PORT_DEFAULT}, NODE_RANK = {node_rank_mock}"
            ) in out
            assert os.environ[ENV_MASTER_ADDR] == mpi_master_addr_mock
            assert os.environ[ENV_MASTER_PORT] == str(MASTER_PORT_DEFAULT)

    # If OMPI_COMM_WORLD_RANK is not set, but one of AZ_BATCHAI_MPI_MASTER_NODE, AZ_BATCH_MASTER_NODE or
    # ENV_MASTER_IP is set, a KeyError should be raised
    env_dict_with_mpi_master_var_no_world_rank = {
        # mpi_master vars
        util.ENV_AZ_BATCHAI_MPI_MASTER_NODE: mpi_master_node_mock,
        util.ENV_MASTER_ADDR: mpi_master_addr_mock,
        # AKS vars
        util.ENV_MASTER_IP: master_ip_mock,
    }

    with pytest.raises(KeyError) as ex:
        with mock.patch.dict(os.environ, env_dict_with_mpi_master_var_no_world_rank, clear=True):
            util.set_environment_variables_for_multi_node()
            out = caplog.messages[-1]
            assert "NODE_RANK" in str(ex)

    # # If ENV_AZ_BATCHAI_MPI_MASTER_NODE is set to localhost, it should be assumed to be a single node job
    caplog.clear()

    env_dict_with_mpi_master_localhost = {
        # mpi_master vars
        util.ENV_AZ_BATCHAI_MPI_MASTER_NODE: "localhost",
        util.ENV_MASTER_ADDR: mpi_master_addr_mock,
        # other vars
        util.ENV_OMPI_COMM_WORLD_RANK: node_rank_mock,
    }

    with mock.patch.dict(os.environ, env_dict_with_mpi_master_localhost, clear=True):
        with caplog.at_level(logging.INFO):  # type: ignore
            util.set_environment_variables_for_multi_node()
            assert "No settings for the MPI central node found" in caplog.messages[-1]  # type: ignore
            assert "Assuming that this is a single node training job" in caplog.text  # type: ignore


@pytest.mark.parametrize(
    "master_node_mock, addr, port, should_pass",
    [
        ("1234.0.0.0", "1234.0.0.0", "6105", True),
        ("1234.0.0.0:4444", "1234.0.0.0", "4444", True),
        ("1234.0.0.0:4444:1", "1234.0.0.0", "4444", False),
    ],
)
def test_set_env_vars_multi_node_split_master_addr(
    master_node_mock: str, addr: str, port: str, should_pass: Boolean, caplog: LogCaptureFixture
) -> None:
    # Accepted formats of AZ_BATCH_MASTER_NODE are "ip:port" and "ip". Check these are parsed correctly
    node_rank_mock = "1"
    env_dict_with_master_var = {
        util.ENV_AZ_BATCH_MASTER_NODE: master_node_mock,
        # need to set OMPI_GLOBAL_WORLD_RANK to avoid a KeyError when finding NODE_RANK
        util.ENV_OMPI_COMM_WORLD_RANK: node_rank_mock,
    }
    if should_pass:
        with caplog.at_level(logging.INFO):  # type: ignore
            with mock.patch.dict(os.environ, env_dict_with_master_var, clear=True):
                util.set_environment_variables_for_multi_node()
                out = caplog.messages[-1]
                assert (
                    f"Distributed training: MASTER_ADDR = {addr}, MASTER_PORT = "
                    f"{port}, NODE_RANK = {node_rank_mock}"
                ) in out
    else:
        with pytest.raises(ValueError) as ex:
            with mock.patch.dict(os.environ, env_dict_with_master_var, clear=True):
                util.set_environment_variables_for_multi_node()
        assert "Format not recognized" in str(ex.value)


@pytest.mark.fast
@patch("health_azure.utils.fetch_run")
@patch("azureml.core.Workspace")
def test_get_most_recent_run(mock_workspace: MagicMock, mock_fetch_run: MagicMock, tmp_path: Path) -> None:
    mock_run_id = "run_abc_123"
    mock_run = MockRun(mock_run_id)
    mock_workspace.get_run.return_value = mock_run
    mock_fetch_run.return_value = mock_run

    latest_path = tmp_path / "most_recent_run.txt"
    latest_path.write_text(mock_run_id)

    run = util.get_most_recent_run(latest_path, mock_workspace)
    assert run.id == mock_run_id


def _get_experiment_runs(tags: Dict[str, str]) -> List[MockRun]:
    mock_run_no_tags = MockRun()
    mock_run_tags = MockRun(tags={"completed": "True"})
    all_runs = [mock_run_no_tags for _ in range(5)] + [mock_run_tags for _ in range(5)]
    return [r for r in all_runs if r.tags == tags] if tags else all_runs


@pytest.mark.fast
@pytest.mark.parametrize(
    "num_runs, tags, expected_num_returned", [(1, {"completed": "True"}, 1), (3, {}, 3), (2, {"Completed: False"}, 0)]
)
def test_get_latest_aml_run_from_experiment(num_runs: int, tags: Dict[str, str], expected_num_returned: int) -> None:
    mock_experiment_name = "MockExperiment"

    with mock.patch("health_azure.utils.Experiment") as mock_experiment:
        with mock.patch(
            "health_azure.utils.Workspace", experiments={mock_experiment_name: mock_experiment}
        ) as mock_workspace:
            mock_experiment.get_runs.return_value = _get_experiment_runs(tags)
            aml_runs = util.get_latest_aml_runs_from_experiment(
                mock_experiment_name, num_runs=num_runs, tags=tags, aml_workspace=mock_workspace
            )
            assert len(aml_runs) == expected_num_returned


def test_get_latest_aml_run_from_experiment_remote() -> None:
    """
    Test that a remote run with particular tags can be correctly retrieved, ignoring any more recent
    experiments which do not have the correct tags. Note: this test will instantiate 2 new Runs in the
    workspace described in your config.json file, under an experiment defined by experiment_for_unittests()
    """
    ws = DEFAULT_WORKSPACE.workspace
    assert True

    experiment = Experiment(ws, experiment_for_unittests())
    config = ScriptRunConfig(
        source_directory=".", command=["cd ."], compute_target="local"  # command that does nothing
    )
    # Create first run and tag
    with append_to_amlignore(amlignore=Path("") / AML_IGNORE_FILE, lines_to_append=DEFAULT_IGNORE_FOLDERS):
        first_run = experiment.submit(config)
        first_run.diplay_name = "test_get_latest_aml_run_from_experiment_remote"
    tags = {"experiment_type": "great_experiment"}
    first_run.set_tags(tags)
    first_run.wait_for_completion()

    # Create second run and ensure no tags
    with append_to_amlignore(amlignore=Path("") / AML_IGNORE_FILE, lines_to_append=DEFAULT_IGNORE_FOLDERS):
        second_run = experiment.submit(config)
    if any(second_run.get_tags()):
        second_run.remove_tags(tags)

    # Retrieve latest run with given tags (expect first_run to be returned)
    retrieved_runs = util.get_latest_aml_runs_from_experiment(experiment_for_unittests(), tags=tags, aml_workspace=ws)
    assert len(retrieved_runs) == 1
    assert retrieved_runs[0].id == first_run.id
    assert retrieved_runs[0].get_tags() == tags


@pytest.mark.fast
@patch("health_azure.utils.Workspace")
@pytest.mark.parametrize("mock_run_id", ["run_abc_123", "experiment1:run_bcd_456"])
def test_get_aml_run_from_run_id(mock_workspace: MagicMock, mock_run_id: str) -> None:
    def _mock_get_run(run_id: str) -> MockRun:
        if len(mock_run_id.split(EXPERIMENT_RUN_SEPARATOR)) > 1:
            return MockRun(mock_run_id.split(EXPERIMENT_RUN_SEPARATOR)[1])
        return MockRun(mock_run_id)

    mock_workspace.get_run = _mock_get_run

    aml_run = util.get_aml_run_from_run_id(mock_run_id, aml_workspace=mock_workspace)
    if len(mock_run_id.split(EXPERIMENT_RUN_SEPARATOR)) > 1:
        mock_run_id = mock_run_id.split(EXPERIMENT_RUN_SEPARATOR)[1]

    assert aml_run.id == mock_run_id


def _get_file_names(pref: str = "") -> List[str]:
    file_names = ["somepath.txt", "abc/someotherpath.txt", "abc/def/anotherpath.txt"]
    if len(pref) > 0:
        return [u for u in file_names if u.startswith(pref)]
    else:
        return file_names


def test_get_run_file_names() -> None:
    with patch("azureml.core.Run") as mock_run:
        expected_file_names = _get_file_names()
        mock_run.get_file_names.return_value = expected_file_names
        # check that we get the expected run paths if no filter is applied
        run_paths = util.get_run_file_names(mock_run)  # type: ignore
        assert len(run_paths) == len(expected_file_names)
        assert sorted(run_paths) == sorted(expected_file_names)

        # Now check we get the expected run paths if a filter is applied
        prefix = "abc"
        run_paths = util.get_run_file_names(mock_run, prefix=prefix)
        assert all([f.startswith(prefix) for f in run_paths])


def _mock_download_file(
    filename: str, output_file_path: Optional[Path] = None, _validate_checksum: bool = False
) -> None:
    """
    Creates an empty file at the given output_file_path
    """
    output_file_path = Path('test_output') if output_file_path is None else output_file_path
    output_file_path = Path(output_file_path) if not isinstance(output_file_path, Path) else output_file_path  # mypy
    output_file_path.parent.mkdir(exist_ok=True, parents=True)
    output_file_path.touch(exist_ok=True)


@pytest.mark.parametrize("dummy_env_vars", [{}, {util.ENV_LOCAL_RANK: "1"}])
@pytest.mark.parametrize("prefix", ["", "abc"])
def test_download_run_files(tmp_path: Path, dummy_env_vars: Dict[Optional[str], Optional[str]], prefix: str) -> None:
    # Assert that 'downloaded' paths don't exist to begin with
    dummy_paths = [Path(x) for x in _get_file_names(pref=prefix)]
    expected_paths = [tmp_path / dummy_path for dummy_path in dummy_paths]
    # Ensure that paths don't already exist
    [p.unlink() for p in expected_paths if p.exists()]  # type: ignore
    assert not any([p.exists() for p in expected_paths])

    mock_run = MockRun(run_id="id123")
    with mock.patch.dict(os.environ, dummy_env_vars):
        with patch("health_azure.utils.get_run_file_names") as mock_get_run_paths:
            mock_get_run_paths.return_value = dummy_paths  # type: ignore
            mock_run.download_file = MagicMock()  # type: ignore
            mock_run.download_file.side_effect = _mock_download_file

            util._download_files_from_run(mock_run, output_dir=tmp_path)
            # First test the case where is_local_rank_zero returns True
            if not any(dummy_env_vars):
                # Check that our mocked _download_file_from_run has been called once for each file
                assert sum([p.exists() for p in expected_paths]) == len(expected_paths)
            # Now test the case where is_local_rank_zero returns False - in this case nothing should be created
            else:
                assert not any([p.exists() for p in expected_paths])


@patch("health_azure.utils.get_workspace")
@patch("health_azure.utils.get_aml_run_from_run_id")
@patch("health_azure.utils._download_files_from_run")
def test_download_files_from_run_id(
    mock_download_run_files: MagicMock, mock_get_aml_run_from_run_id: MagicMock, mock_workspace: MagicMock
) -> None:
    mock_run = {"id": "run123"}
    mock_get_aml_run_from_run_id.return_value = mock_run
    util.download_files_from_run_id("run123", Path(__file__))
    mock_download_run_files.assert_called_with(mock_run, Path(__file__), prefix="", validate_checksum=False)


@pytest.mark.parametrize("dummy_env_vars, expect_file_downloaded", [({}, True), ({util.ENV_LOCAL_RANK: "1"}, False)])
@patch("azureml.core.Run", MockRun)
def test_download_file_from_run(tmp_path: Path, dummy_env_vars: Dict[str, str], expect_file_downloaded: bool) -> None:
    dummy_filename = "filetodownload.txt"
    expected_file_path = tmp_path / dummy_filename

    # mock the method 'download_file' on the AML Run class and assert it gets called with the expected params
    mock_run = MockRun(run_id="id123")
    mock_run.download_file = MagicMock(return_value=None)  # type: ignore
    mock_run.download_file.side_effect = _mock_download_file

    with mock.patch.dict(os.environ, dummy_env_vars):
        _ = util._download_file_from_run(mock_run, dummy_filename, expected_file_path)

        if expect_file_downloaded:
            mock_run.download_file.assert_called_with(
                dummy_filename, output_file_path=str(expected_file_path), _validate_checksum=False
            )
            assert expected_file_path.exists()
        else:
            assert not expected_file_path.exists()


def test_download_file_from_run_remote(tmp_path: Path) -> None:
    # This test will create a Run in your workspace (using only local compute)
    ws = DEFAULT_WORKSPACE.workspace
    experiment = Experiment(ws, experiment_for_unittests())
    config = ScriptRunConfig(
        source_directory=".", command=["cd ."], compute_target="local"  # command that does nothing
    )
    with append_to_amlignore(amlignore=Path("") / AML_IGNORE_FILE, lines_to_append=DEFAULT_IGNORE_FOLDERS):
        run = experiment.submit(config)
        run.diplay_name = "test_download_file_from_run_remote"

    file_to_upload = tmp_path / "dummy_file.txt"
    file_contents = "Hello world"
    file_to_upload.write_text(file_contents)

    # This should store the file in outputs
    run.upload_file("dummy_file", str(file_to_upload))

    output_file_path = tmp_path / "downloaded_file.txt"
    assert not output_file_path.exists()

    start_time = time.perf_counter()
    _ = util._download_file_from_run(run, "dummy_file", output_file_path)
    end_time = time.perf_counter()
    time_dont_validate_checksum = end_time - start_time

    assert output_file_path.exists()
    assert output_file_path.read_text() == file_contents

    # Now delete the file and try again with _validate_checksum == True
    output_file_path.unlink()
    assert not output_file_path.exists()
    start_time = time.perf_counter()
    _ = util._download_file_from_run(run, "dummy_file", output_file_path, validate_checksum=True)
    end_time = time.perf_counter()
    time_validate_checksum = end_time - start_time

    assert output_file_path.exists()
    assert output_file_path.read_text() == file_contents

    logging.info(
        f"Time to download file without checksum: {time_dont_validate_checksum} vs time with"
        f"validation {time_validate_checksum}."
    )


def test_download_run_file_during_run(tmp_path: Path) -> None:
    """
    Test if we can download files from a run, when executing inside AzureML. This should not require any additional
    information about the workspace to use, but pick up the current workspace.
    """
    # Create a run that contains a simple txt file
    experiment_name = effective_experiment_name("himl-tests")
    run_to_download_from = util.create_aml_run_object(
        experiment_name=experiment_name, workspace=DEFAULT_WORKSPACE.workspace
    )
    file_contents = "Hello World!"
    file_name = "hello.txt"
    full_file_path = tmp_path / file_name
    full_file_path.write_text(file_contents)
    run_to_download_from.upload_file(file_name, str(full_file_path))
    run_to_download_from.complete()
    run_id = run_to_download_from.id

    # Test if we can retrieve the run directly from the workspace. This tests for a bug in an earlier version
    # of the code where run IDs as those created from runs outside AML were not recognized
    run_2 = util.get_aml_run_from_run_id(run_id, aml_workspace=DEFAULT_WORKSPACE.workspace)
    assert run_2.id == run_id

    # Now create an AzureML run with a simple script that uses that file. The script will download the file,
    # where the download is should pick up the workspace from the current AML run.
    script_body = ""
    script_body += f"run_id = '{run_id}'\n"
    script_body += f"    file_name = '{file_name}'\n"
    script_body += f"    file_contents = '{file_contents}'\n"
    script_body += """
    output_path = Path("outputs")
    output_path.mkdir(exist_ok=True)

    download_files_from_run_id(run_id, output_path, prefix=file_name)
    full_file_path = output_path / file_name
    actual_contents = full_file_path.read_text().strip()
    print(f"{actual_contents}")
    assert actual_contents == file_contents
"""
    extra_options = {
        "imports": """
import sys
from pathlib import Path
from azureml.core import Run
from health_azure.utils import download_files_from_run_id""",
        "body": script_body,
    }
    # Run the script locally first, then in the cloud. In local runs, the workspace should be picked up from the
    # config.json file, in AzureML runs it should be read off the run context.
    render_and_run_test_script(tmp_path, RunTarget.LOCAL, extra_options, extra_args=[], expected_pass=True)
    print("Local run finished")
    render_and_run_test_script(tmp_path / "foo", RunTarget.AZUREML, extra_options, extra_args=[], expected_pass=True)


def test_replace_directory(tmp_path: Path) -> None:
    extra_options = {
        "imports": """
import sys
import shutil
from pathlib import Path
from health_azure.utils import replace_directory
""",
        "body": """
    output_dir = Path("outputs/test_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    file_name = "hello.txt"
    (output_dir / file_name).write_text("Hello World!")
    assert (output_dir / file_name).exists()
    new_output_dir = output_dir.parent / "more_test_outputs"

    replace_directory(output_dir, new_output_dir)

    assert not output_dir.exists()
    assert (new_output_dir / file_name).exists()
""",
    }

    render_and_run_test_script(tmp_path, RunTarget.LOCAL, extra_options, extra_args=[], expected_pass=True)
    print("Local run finished")

    render_and_run_test_script(tmp_path / "foo", RunTarget.AZUREML, extra_options, extra_args=[], expected_pass=True)


def test_is_global_rank_zero() -> None:
    with mock.patch.dict(os.environ, {util.ENV_NODE_RANK: "0", util.ENV_GLOBAL_RANK: "0", util.ENV_LOCAL_RANK: "0"}):
        assert not util.is_global_rank_zero()

    with mock.patch.dict(os.environ, {util.ENV_GLOBAL_RANK: "0", util.ENV_LOCAL_RANK: "0"}):
        assert not util.is_global_rank_zero()

    with mock.patch.dict(os.environ, {util.ENV_NODE_RANK: "0"}):
        assert util.is_global_rank_zero()


def test_is_local_rank_zero() -> None:
    # mock the environment variables
    with mock.patch.dict(os.environ, {}):
        assert util.is_local_rank_zero()

    with mock.patch.dict(os.environ, {util.ENV_GLOBAL_RANK: "1", util.ENV_LOCAL_RANK: "1"}):
        assert not util.is_local_rank_zero()


@pytest.mark.parametrize(
    "dummy_recovery_id",
    ["expt:run_abc_1234", "['expt:abc_432','expt2:def_111']", "run_ghi_1234", "['run_jkl_1234','run_mno_7654']"],
)
def test_get_run_source(
    dummy_recovery_id: str,
) -> None:
    arguments = ["", "--run", dummy_recovery_id]
    with patch.object(sys, "argv", arguments):
        script_config = util.AmlRunScriptConfig()
        script_config = parse_args_and_update_config(script_config, arguments)

        if isinstance(script_config.run, List):
            assert isinstance(script_config.run[0], str)
        else:
            assert isinstance(script_config.run, str)


def get_container_client(datastore: AzureBlobDatastore) -> ContainerClient:
    """Gets a ContainerClient to interact with the blobs in the given datastore.

    param datastore: The datastore from which the files should be read.
    """
    return datastore.blob_service.get_container_client(datastore.container_name)


def get_blobs_in_datastore(datastore: AzureBlobDatastore, prefix: str) -> List[Any]:
    """Gets all blobs in the datastore where the name starts with the given prefix.

    param datastore: The datastore from which the files should be read.
    param prefix: The prefix string for the files that should be returned.
    """
    return list(get_container_client(datastore).list_blobs(name_starts_with=prefix))


def delete_blobs_in_datastore(datastore: AzureBlobDatastore, prefix: str) -> None:
    """Deletes all existing files in blob storage at the location that the test uses.

    param datastore: The datastore from which the files should be deleted.
    param prefix: The prefix string for the files that should be deleted.
    """
    container_client = get_container_client(datastore)
    for existing_blob in get_blobs_in_datastore(datastore, prefix):
        container_client.delete_blob(existing_blob.name)


@pytest.mark.parametrize("overwrite", [True, False])
@pytest.mark.skip(reason="This test no longer works because we removed access keys from the datastores")
def test_download_from_datastore(tmp_path: Path, overwrite: bool) -> None:
    """
    Test that download_from_datastore successfully downloads file from Blob Storage.
    Note that this will temporarily upload a file to the default datastore of the default workspace -
    (determined by either a config.json file, or by specifying workspace settings in the environment variables).
    After the test has completed, the blob will be deleted.
    """
    ws = DEFAULT_WORKSPACE.workspace
    default_datastore: AzureBlobDatastore = ws.get_default_datastore()
    dummy_file_content = "Hello world"
    local_data_path = tmp_path / "local_data"
    local_data_path.mkdir()
    test_data_path_remote = "test_data/abc"

    delete_blobs_in_datastore(datastore=default_datastore, prefix=test_data_path_remote)
    try:
        # Create dummy data files and upload to datastore (checking they are uploaded)
        dummy_filenames = []
        num_dummy_files = 2
        for i in range(num_dummy_files):
            dummy_filename = f"dummy_data_{i}.txt"
            dummy_filenames.append(dummy_filename)
            data_to_upload_path = local_data_path / dummy_filename
            data_to_upload_path.write_text(dummy_file_content)
        default_datastore.upload(str(local_data_path), test_data_path_remote, overwrite=False)
        # Wait a bit because there seem to be spurious errors with files not yet existing at this point
        time.sleep(0.1)
        existing = get_blobs_in_datastore(default_datastore, prefix=test_data_path_remote)
        assert len(existing) == num_dummy_files

        # Check that the file doesn't currently exist at download location
        downloaded_data_path = tmp_path / "downloads"
        assert not downloaded_data_path.exists()

        # Now attempt to download
        util.download_from_datastore(
            default_datastore.name,
            test_data_path_remote,
            downloaded_data_path,
            aml_workspace=ws,
            overwrite=overwrite,
            show_progress=True,
        )
        expected_local_download_dir = downloaded_data_path / test_data_path_remote
        assert expected_local_download_dir.exists()
        expected_download_paths = [expected_local_download_dir / dummy_filename for dummy_filename in dummy_filenames]
        assert all([p.exists() for p in expected_download_paths])
    finally:
        delete_blobs_in_datastore(datastore=default_datastore, prefix=test_data_path_remote)


@pytest.mark.parametrize("overwrite", [True, False])
@pytest.mark.skip(reason="This test no longer works because we removed access keys from the datastores")
def test_upload_to_datastore(tmp_path: Path, overwrite: bool) -> None:
    """
    Test that upload_to_datastore successfully uploads a file to Blob Storage.
    Note that this will temporarily upload a file to the default datastore of the default workspace -
    (determined by either a config.json file, or by specifying workspace settings in the environment variables).
    After the test has completed, the blob will be deleted.
    """
    ws = DEFAULT_WORKSPACE.workspace
    default_datastore: AzureBlobDatastore = ws.get_default_datastore()
    dummy_file_content = "Hello world"

    remote_data_dir = "test_data"
    dummy_file_name = Path("abc/uploaded_file.txt")
    expected_remote_path = Path(remote_data_dir) / dummy_file_name.name

    delete_blobs_in_datastore(datastore=default_datastore, prefix=str(expected_remote_path.as_posix()))

    try:
        # Create a dummy data file and upload to datastore
        data_to_upload_path = tmp_path / dummy_file_name
        data_to_upload_path.parent.mkdir(exist_ok=True, parents=True)
        data_to_upload_path.write_text(dummy_file_content)

        util.upload_to_datastore(
            default_datastore.name,
            data_to_upload_path.parent,
            Path(remote_data_dir),
            aml_workspace=ws,
            overwrite=overwrite,
            show_progress=True,
        )
        # Wait a bit because there seem to be spurious errors with files not yet existing at this point
        time.sleep(0.1)

        existing_blobs = get_blobs_in_datastore(default_datastore, prefix=str(expected_remote_path.as_posix()))
        assert len(existing_blobs) == 1
    finally:
        delete_blobs_in_datastore(datastore=default_datastore, prefix=str(expected_remote_path.as_posix()))


@pytest.mark.parametrize(
    "arguments, run_id",
    [
        (["", "--run", "run_abc_123"], "run_abc_123"),
        (["", "--run", "run_abc_123,run_def_456"], ["run_abc_123", "run_def_456"]),
        (["", "--run", "expt_name:run_abc_123"], "expt_name:run_abc_123"),
    ],
)
def test_script_config_run_src(arguments: List[str], run_id: Union[str, List[str]]) -> None:
    with patch.object(sys, "argv", arguments):
        script_config = util.AmlRunScriptConfig()
        script_config = parse_args_and_update_config(script_config, arguments)

        if isinstance(run_id, list):
            for script_config_run, expected_run_id in zip(script_config.run, run_id):
                assert script_config_run == expected_run_id
        else:
            if len(run_id.split(EXPERIMENT_RUN_SEPARATOR)) > 1:
                assert script_config.run == [run_id.split(EXPERIMENT_RUN_SEPARATOR)[1]]
            else:
                assert script_config.run == [run_id]


@patch("health_azure.utils.download_files_from_run_id")
@patch("health_azure.utils.get_workspace")
def test_checkpoint_download(mock_get_workspace: MagicMock, mock_download_files: MagicMock) -> None:
    mock_workspace = MagicMock()
    mock_get_workspace.return_value = mock_workspace
    dummy_run_id = "run_def_456"
    prefix = "path/to/file"
    output_file_dir = Path("my_ouputs")
    util.download_checkpoints_from_run_id(dummy_run_id, prefix, output_file_dir, aml_workspace=mock_workspace)
    mock_download_files.assert_called_once_with(
        dummy_run_id, output_file_dir, prefix=prefix, workspace=mock_workspace, validate_checksum=True
    )


@pytest.mark.slow
def test_checkpoint_download_remote(tmp_path: Path) -> None:
    """
    Creates a large dummy file (around 250 MB) and ensures we can upload it to a Run and subsequently download
    with no issues, thus replicating the behaviour of downloading a large checkpoint file.
    """
    num_dummy_files = 1
    prefix = "outputs/checkpoints/"

    ws = DEFAULT_WORKSPACE.workspace
    experiment = Experiment(ws, experiment_for_unittests())
    config = ScriptRunConfig(
        source_directory=".", command=["cd ."], compute_target="local"  # command that does nothing
    )
    with append_to_amlignore(amlignore=Path("") / AML_IGNORE_FILE, lines_to_append=DEFAULT_IGNORE_FOLDERS):
        run = experiment.submit(config)
        run.display_name = "test_checkpoint_download_remote"

    file_contents = "Hello world"
    file_name = ""  # for pyright
    for i in range(num_dummy_files):
        file_name = f"dummy_checkpoint_{i}.txt"
        large_file_path = tmp_path / file_name
        with open(str(large_file_path), "wb") as f_path:
            f_path.seek((1024 * 1024 * 240) - 1)
            f_path.write(bytearray(file_contents, encoding="UTF-8"))

        file_size = large_file_path.stat().st_size
        logging.info(f"File {i} size: {file_size}")

        local_path = str(large_file_path)
        run.upload_file(prefix + file_name, local_path)

    # Check the local dir is empty to begin with
    output_file_dir = tmp_path
    assert not (output_file_dir / prefix).exists()

    whole_file_path = prefix + file_name
    start_time = time.perf_counter()
    util.download_checkpoints_from_run_id(run.id, whole_file_path, output_file_dir, aml_workspace=ws)
    end_time = time.perf_counter()
    time_taken = end_time - start_time
    logging.info(f"Time taken to download file: {time_taken}")

    download_file_path = output_file_dir / prefix / "dummy_checkpoint_0.txt"
    assert (output_file_dir / prefix).is_dir()
    assert len(list((output_file_dir / prefix).iterdir())) == num_dummy_files
    found_file_contents = ""  # for pyright
    with open(str(download_file_path), "rb") as f_path:
        for line in f_path:
            chunk = line.strip(b'\x00')
            if chunk:
                found_file_contents = chunk.decode("utf-8")
                break

    assert found_file_contents == file_contents

    # Delete the file downloaded file and check that download_checkpoints also works on a single checkpoint file
    download_file_path.unlink()
    assert not download_file_path.exists()

    util.download_checkpoints_from_run_id(run.id, whole_file_path, output_file_dir, aml_workspace=ws)
    assert download_file_path.exists()
    with open(str(download_file_path), "rb") as f_path:
        for line in f_path:
            chunk = line.strip(b'\x00')
            if chunk:
                found_file_contents = chunk.decode("utf-8")
                break
    assert found_file_contents == file_contents


@pytest.mark.parametrize(
    ("available", "initialized", "expected_barrier_called"),
    [(False, True, False), (True, False, False), (False, False, False), (True, True, True)],
)
@pytest.mark.fast
def test_torch_barrier(available: bool, initialized: bool, expected_barrier_called: bool) -> None:
    distributed = mock.MagicMock()
    distributed.is_available.return_value = available
    distributed.is_initialized.return_value = initialized
    distributed.barrier = mock.MagicMock()
    with mock.patch.dict("sys.modules", {"torch": mock.MagicMock(distributed=distributed)}):
        util.torch_barrier()
        if expected_barrier_called:
            distributed.barrier.assert_called_once()
        else:
            assert distributed.barrier.call_count == 0


class DummyConfig(param.Parameterized):
    string_param = param.String()
    int_param = param.Integer()

    def validate(self) -> None:
        assert isinstance(self.string_param, str)
        assert isinstance(self.int_param, int)


class ClassFrom(param.Parameterized):
    foo = param.String("foo")
    bar = param.Integer(1)
    baz = param.String("baz")
    _private = param.String("private")
    constant = param.String("constant", constant=True)


class ClassTo(param.Parameterized):
    foo = param.String("foo2")
    bar = param.Integer(2)
    _private = param.String("private2")
    constant = param.String("constant2", constant=True)


class NotParameterized:
    foo = 1


@pytest.mark.fast
def test_create_from_matching_params() -> None:
    """
    Test if Parameterized objects can be cloned by looking at matching fields.
    """
    class_from = ClassFrom()
    class_to = util.create_from_matching_params(class_from, cls_=ClassTo)
    assert isinstance(class_to, ClassTo)
    assert class_to.foo == "foo"
    assert class_to.bar == 1
    # Constant fields should not be touched
    assert class_to.constant == "constant2"
    # Private fields must be copied over.
    assert class_to._private == "private"
    # Baz is only present in the "from" object, and should not be copied to the new object
    assert not hasattr(class_to, "baz")

    with pytest.raises(ValueError) as ex:
        util.create_from_matching_params(class_from, NotParameterized)
    assert "subclass of param.Parameterized" in str(ex)
    assert "NotParameterized" in str(ex)


@pytest.mark.fast
def test_create_v2_job_command_line_args_from_params() -> None:
    test_params = ["--azureml"]
    expected_command_line_arg_str = f"{test_params[0]}"
    v2_command_line_args = util.create_v2_job_command_line_args_from_params(test_params)
    assert v2_command_line_args == expected_command_line_arg_str

    test_params = ["--azureml", "--test_arg_1=['test_arg_1_value_1', 'test_arg_1_value_2']"]
    expected_command_line_arg_str = f'{test_params[0]} "{test_params[1]}"'
    v2_command_line_args = util.create_v2_job_command_line_args_from_params(test_params)
    assert v2_command_line_args == expected_command_line_arg_str

    with pytest.raises(ValueError, match="cannot contain both single and double quotes"):
        test_params = ["--azureml", "'--test_arg_1=[\"test_arg_1_value_1\", \'test_arg_1_value_2\']'"]
        util.create_v2_job_command_line_args_from_params(test_params)


class MockChildRun:
    def __init__(self, run_id: str, cross_val_index: int):
        self.id = run_id
        self.tags = {"hyperparameters": json.dumps({"child_run_index": cross_val_index})}
        self.status = RunStatus.COMPLETED

    def get_metrics(self) -> Dict[str, Union[float, List[Union[int, float]]]]:
        num_epochs = 5
        return {
            "epoch": list(range(num_epochs)),
            "train/loss": [np.random.rand() for _ in range(num_epochs)],
            "train/auroc": [np.random.rand() for _ in range(num_epochs)],
            "val/loss": [np.random.rand() for _ in range(num_epochs)],
            "val/recall": [np.random.rand() for _ in range(num_epochs)],
            "test/f1score": np.random.rand(),
            "test/accuracy": np.random.rand(),
        }


class MockHyperDriveRun:
    def __init__(self, num_children: int) -> None:
        self.num_children = num_children

    def get_children(self) -> List[MockChildRun]:
        return [MockChildRun(f"run_abc_{i}456", i) for i in range(self.num_children)]


class MockRunWithMetrics:
    def __init__(self, run_id: str = 'run1234', tags: Optional[Dict[str, str]] = None) -> None:
        self.id = run_id
        self.status = RunStatus.COMPLETED
        self.metrics: Dict[str, Union[float, List[float]]] = {
            "test/accuracy": 0.8,
            "test/auroc": 0.7,
            "val/loss": [1.0, 0.8, 0.75],
        }

    def get_metrics(self) -> Dict[str, Union[List[float], float]]:
        """
        Return dummy metrics which can either be a float - i.e. if the metric is calculated in the
        test phase, or a list of floats, if calculated during the validation phase
        """
        return self.metrics


def test_download_files_from_hyperdrive_children(tmp_path: Path) -> None:
    def _mock_get_tags(run: Any, arg_name: Any) -> Dict[str, str]:
        return run.id

    def _mock_download_file(child_run_id: str, local_folder_child_run: Path, prefix: Optional[str] = None) -> None:
        prefix = prefix or ""  # for pyright
        expected_path = local_folder_child_run / prefix
        expected_path.touch()

    num_child_runs = 2
    hyperparam_name = "crossval_index"
    remote_file_path = "dummy_file.csv"
    local_download_folder = tmp_path / "downloaded_hyperdrive"
    local_download_folder.mkdir(exist_ok=False)
    assert len(list(local_download_folder.iterdir())) == 0

    mock_run = MagicMock()
    mock_run_1, mock_run_2 = MagicMock(id=1), MagicMock(id=2)

    with patch("health_azure.utils.download_files_from_run_id", new=_mock_download_file):
        with patch("health_azure.utils.get_tags_from_hyperdrive_run", new=_mock_get_tags):
            mock_run.get_children.return_value = [mock_run_1, mock_run_2]
            util.download_files_from_hyperdrive_children(
                mock_run, remote_file_path, local_download_folder, hyperparam_name=hyperparam_name
            )

    assert len(list(local_download_folder.iterdir())) == num_child_runs
    assert (local_download_folder / str(mock_run_1.id)).is_dir()
    assert (local_download_folder / str(mock_run_1.id) / remote_file_path).exists()


@pytest.mark.fast
@pytest.mark.parametrize("use_run_id", [True, False])
def test_get_metrics_for_hyperdrive_run(use_run_id: bool) -> None:
    """Test the case where we get metrics, providing the run ID"""
    num_crossval_splits = 2
    run_id = "run_id_123"
    child_run_arg_name = "child_run_index"
    workspace_config_path = Path("foo")
    mock_run = MockHyperDriveRun(num_children=num_crossval_splits)
    if use_run_id:
        with patch("health_azure.utils.get_aml_run_from_run_id") as mock_get_run:
            mock_get_run.return_value = mock_run
            metrics_dict = util.get_metrics_for_hyperdrive_run(
                run_id=run_id,
                child_run_arg_name=child_run_arg_name,
                aml_workspace=DEFAULT_WORKSPACE.workspace,
                workspace_config_path=workspace_config_path,
            )
            mock_get_run.assert_called_once_with(
                run_id, aml_workspace=DEFAULT_WORKSPACE.workspace, workspace_config_path=workspace_config_path
            )
    else:
        with patch("health_azure.utils.get_aml_run_from_run_id") as mock_get_run:
            metrics_dict = util.get_metrics_for_hyperdrive_run(
                run=mock_run,
                child_run_arg_name=child_run_arg_name,
                aml_workspace=DEFAULT_WORKSPACE.workspace,
            )
    assert isinstance(metrics_dict, Dict)
    assert len(metrics_dict) == num_crossval_splits
    expected_metrics = MockChildRun(run_id="", cross_val_index=1).get_metrics()
    for _, value in metrics_dict.items():
        assert isinstance(value, Dict)
        assert len(value) == len(expected_metrics)


@pytest.mark.fast
def test_get_metrics_for_hyperdrive_run_from_run() -> None:
    """Test getting metrics when neither run nor run ID is provided"""
    with pytest.raises(ValueError, match="Either run or run_id must be provided"):
        util.get_metrics_for_hyperdrive_run(child_run_arg_name="child_run_index")


@pytest.mark.fast
def test_get_metrics_for_hyperdrive_run_offline() -> None:
    """Test getting metrics when the run is offline"""
    offline_run = Run.get_context()
    assert isinstance(offline_run, _OfflineRun)
    assert util.get_metrics_for_hyperdrive_run(child_run_arg_name="child_run_index", run=offline_run) == {}


@pytest.mark.fast
def test_get_metrics_for_run_offline() -> None:
    """Test getting metrics when the run is offline"""
    offline_run = Run.get_context()
    assert isinstance(offline_run, _OfflineRun)
    assert util.get_metrics_for_run(run=offline_run) == {}


@pytest.mark.fast
def test_aggregate_hyperdrive_metrics() -> None:
    """Test aggregating hyperdrive metrics and passing of all run-related settings"""
    child_run_tag = "foo"
    metric_name = "bar"
    metrics = {child_run_tag: {metric_name: 1.0}}
    with patch("health_azure.utils.get_metrics_for_hyperdrive_run") as mock_get_metrics:
        mock_get_metrics.return_value = metrics
        run = MockHyperDriveRun(num_children=2)
        run_id = "run_id"
        child_run_arg_name = "crossval_index"
        aml_workspace = (DEFAULT_WORKSPACE.workspace,)
        workspace_config_path = Path("config")
        keep_metrics = [metric_name]
        df = util.aggregate_hyperdrive_metrics(
            child_run_arg_name=child_run_arg_name,
            run=run,
            run_id=run_id,
            keep_metrics=keep_metrics,
            aml_workspace=aml_workspace,
            workspace_config_path=workspace_config_path,
        )
        mock_get_metrics.assert_called_once_with(
            child_run_arg_name=child_run_arg_name,
            run_id=run_id,
            run=run,
            keep_metrics=keep_metrics,
            aml_workspace=aml_workspace,
            workspace_config_path=workspace_config_path,
        )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(metrics)
    assert list(df.index) == [metric_name]
    assert df.columns == [child_run_tag]


@pytest.mark.fast
def test_aggregate_hyperdrive_metrics_from_run() -> None:
    """Test hyperdrive metrics aggregation when a run is provided"""
    expected_metrics = MockChildRun(run_id="", cross_val_index=1).get_metrics()

    num_crossval_splits = 2
    dummy_hyperdrive_run = MockHyperDriveRun(num_children=num_crossval_splits)

    df = util.aggregate_hyperdrive_metrics(
        child_run_arg_name="child_run_index",
        run=dummy_hyperdrive_run,
    )
    num_rows, num_cols = df.shape
    assert num_rows == len(expected_metrics)
    assert num_cols == num_crossval_splits
    for metric_name in expected_metrics:
        assert metric_name in df.index
        assert len(df.loc[metric_name]) == num_crossval_splits
    assert len(df.index) == len(expected_metrics)
    epochs = df.loc["epoch"]
    assert isinstance(epochs[0], list)
    assert len(epochs[0]) == len(expected_metrics["epoch"])  # type: ignore
    test_accuracies = df.loc["test/accuracy"]
    assert isinstance(test_accuracies[0], float)


@pytest.mark.fast
def test_aggregate_hyperdrive_metrics_keep() -> None:
    """Test hyperdrive metrics aggregation when restricting the set of metrics"""
    num_crossval_splits = 2
    dummy_hyperdrive_run = MockHyperDriveRun(num_children=num_crossval_splits)
    valid_metric = "test/accuracy"
    keep_metrics = [valid_metric, "idontexist"]
    df = util.aggregate_hyperdrive_metrics(
        run=dummy_hyperdrive_run,
        child_run_arg_name="child_run_index",
        keep_metrics=keep_metrics,
    )
    assert len(df.index) == 1
    assert valid_metric in df.index
    assert len(df.loc[valid_metric]) == num_crossval_splits
    # Test the metric that is present on the run: test/accuracy should be a float for each child run
    for item in df.loc[valid_metric]:
        assert isinstance(item, float)


@pytest.mark.fast
def test_get_metrics_for_run() -> None:
    ws = DEFAULT_WORKSPACE.workspace
    dummy_run_id = "run_abc_123"
    dummy_run = MockRunWithMetrics(dummy_run_id)
    # test the case where a run id is passed
    with patch("health_azure.utils.get_aml_run_from_run_id") as mock_get_run:
        mock_get_run.return_value = dummy_run
        expected_metrics_dict = dummy_run.get_metrics()
        metrics_1 = util.get_metrics_for_run(run_id=dummy_run_id, aml_workspace=ws)
        assert metrics_1 == expected_metrics_dict

    # test the case where a run is passed
    metrics_2 = util.get_metrics_for_run(run=dummy_run)
    assert metrics_2 == expected_metrics_dict

    # if neither a run or a run_id is passed, an error should be raised
    with pytest.raises(ValueError, match="Either run or run_id must be provided"):
        util.get_metrics_for_run(aml_workspace=ws)


@pytest.mark.fast
def test_get_metrics_for_run_keep() -> None:
    """Test getting metrics from a run when passing a filter list"""
    run = MockRunWithMetrics("run_id")
    restrict_metrics = ["test/accuracy", "val/loss"]
    metrics = util.get_metrics_for_run(run=run, keep_metrics=restrict_metrics)
    assert set(metrics.keys()) == set(restrict_metrics)
    assert len(metrics) == len(restrict_metrics)
    for metric_name in restrict_metrics:
        assert metrics[metric_name] == run.metrics[metric_name]


@pytest.mark.fast
def test_get_metrics_for_run_keep_missing(caplog: LogCaptureFixture) -> None:
    """Test getting metrics from a run when passing a filter list with a non-existent metric. The nonexistnet metric
    should be ignored"""
    run = MockRunWithMetrics("run_id")
    valid_metric = "test/accuracy"
    invalid_metrics = [valid_metric, "idontexist"]
    with caplog.at_level(logging.WARNING):
        metrics = util.get_metrics_for_run(run=run, keep_metrics=invalid_metrics)
    assert len(caplog.messages) == 1
    assert caplog.messages[0] == f"Metric idontexist not found in run {run.id}"
    assert len(metrics) == 1
    assert valid_metric in metrics
    assert metrics[valid_metric] == run.metrics[valid_metric]


@pytest.mark.fast
def test_get_metrics_for_run_not_completed(caplog: LogCaptureFixture) -> None:
    """Test getting metrics from a run should print a warning if the run is not completed."""
    run = MockRunWithMetrics("run_id")
    with caplog.at_level(logging.WARNING):
        metrics = util.get_metrics_for_run(run=run)
    assert len(caplog.messages) == 0

    run.status = "not_completed"
    assert metrics == run.get_metrics()
    with caplog.at_level(logging.WARNING):
        metrics = util.get_metrics_for_run(run=run)
    assert len(caplog.messages) == 1
    assert caplog.messages[0] == (
        "Run run_id is not completed, but has status 'not_completed'. Metrics may be incomplete."
    )
    assert metrics == run.get_metrics()


@pytest.mark.fast
@pytest.mark.parametrize(
    "metrics", [{"metric1": [1.234], "metric2": [2.345, 3.456]}, {"metric1": [2.345, 3.456], "metric2": [1.234]}]
)
def test_get_metrics_for_run_2(metrics: Dict[str, List[float]]) -> None:
    """
    Test if we can get metrics for a run with no children, and that the order in which metrics are read does
    not create a problem.
    """
    run_name = "test_get_metrics_for_run_2"
    experiment_name = effective_experiment_name("himl-tests")
    run = util.create_aml_run_object(
        experiment_name=experiment_name, run_name=run_name, workspace=DEFAULT_WORKSPACE.workspace
    )
    try:
        for metric_name, metric_values in metrics.items():
            for metric_value in metric_values:
                run.log(metric_name, metric_value)
        run.flush()
        metrics_dict = util.get_metrics_for_run(run=run)
        assert isinstance(metrics_dict, Dict)
        for metric_name, expected_metrics in metrics.items():
            assert metric_name in metrics_dict
            value_from_run = metrics_dict[metric_name]
            if len(expected_metrics) == 1:
                assert value_from_run == expected_metrics[0]
            else:
                assert isinstance(value_from_run, list)
                assert value_from_run == expected_metrics
    finally:
        run.complete()


def test_create_run() -> None:
    """
    Test if we can create an AML run object here in the test suite, write logs and read them back in.
    """
    run_name = "test_create_run"
    experiment_name = effective_experiment_name("himl-tests")
    run: Optional[Run] = None
    try:
        run = util.create_aml_run_object(
            experiment_name=experiment_name, run_name=run_name, workspace=DEFAULT_WORKSPACE.workspace
        )
        assert run is not None
        assert run.display_name == run_name
        assert run.experiment.name == experiment_name
        metric_name = "mymetric"
        metric_value = 1.234
        run.log(metric_name, metric_value)
        run.flush()
        metrics = run.get_metrics(name=metric_name)
        assert metrics[metric_name] == metric_value
    finally:
        if run is not None:
            run.complete()


def test_filter_v2_input_output_args() -> None:
    def _compare_args(expected: List[str], actual: List[str]) -> None:
        assert len(actual) == len(expected)
        for actual_entry in actual:
            assert actual_entry in expected

    args_to_filter = ["--a=foo", "--INPUT_0=input0", "--b=bar", "--INPUT_1=input1"]
    expected_filtered = ["--a=foo", "--b=bar"]
    actual_filtered = util.filter_v2_input_output_args(args_to_filter)
    _compare_args(expected_filtered, actual_filtered)

    # try passing empty list
    empty_list: List[str] = []
    actual_filtered = util.filter_v2_input_output_args(empty_list)
    assert actual_filtered == empty_list

    # pass args with similar but different input and output args
    args_to_filter = ["--input_0=input0", "--a=foo"]
    expected_filtered = ["--input_0=input0", "--a=foo"]
    actual_filtered = util.filter_v2_input_output_args(args_to_filter)
    _compare_args(expected_filtered, actual_filtered)


def test_download_from_run(tmp_path: Path) -> None:
    """Test if a file can be downloaded from a run, and if download is skipped on ranks other than zero."""
    path_on_aml = "outputs/test.txt"
    local_file = tmp_path / "test.txt"
    local_content = "mock content"
    local_file.write_text(local_content)
    run = create_unittest_run_object()
    try:
        run.upload_file(path_on_aml, str(local_file))
        run.flush()
        download_path = tmp_path / "downloaded.txt"
        with mock.patch("health_azure.utils.is_local_rank_zero", return_value=True):
            download_file_if_necessary(run, path_on_aml, download_path)
        assert download_path.is_file(), "File was not downloaded"
        assert download_path.read_text() == local_content, "Downloaded file content is incorrect"

        download_path2 = tmp_path / "downloaded2.txt"
        with mock.patch("health_azure.utils.is_local_rank_zero", return_value=False):
            download_file_if_necessary(run, path_on_aml, download_path2)
        assert not download_path2.is_file(), "No file should have been downloaded"
    finally:
        run.complete()


@pytest.mark.parametrize('overwrite', [False, True])
def test_download_from_run_if_necessary(tmp_path: Path, overwrite: bool) -> None:
    """Test if downloading a file from a run works."""
    filename = "test_output.csv"
    download_dir = tmp_path
    remote_filename = "outputs/" + filename
    expected_local_path = download_dir / filename

    def create_mock_file(name: str, output_file_path: str, _validate_checksum: bool) -> None:
        output_path = Path(output_file_path)
        print(f"Writing mock content to file {output_path}")
        output_path.write_text("mock content")

    run = MagicMock()
    run.download_file.side_effect = create_mock_file

    with mock.patch("health_azure.utils.is_local_rank_zero", return_value=True):
        local_path = download_file_if_necessary(run, remote_filename, expected_local_path)
        run.download_file.assert_called_once()
        assert local_path == expected_local_path
        assert local_path.exists()

        run.reset_mock()
        new_local_path = download_file_if_necessary(run, remote_filename, expected_local_path, overwrite=overwrite)
        if overwrite:
            run.download_file.assert_called_once()
        else:
            run.download_file.assert_not_called()
        assert new_local_path == local_path
        assert new_local_path.exists()


def test_download_from_run_if_necessary_rank_nonzero(tmp_path: Path) -> None:
    filename = "test_output.csv"
    download_dir = tmp_path
    remote_filename = "outputs/" + filename
    expected_local_path = download_dir / filename

    run = MagicMock()
    run.download_file.side_effect = ValueError

    with mock.patch("health_azure.utils.is_local_rank_zero", return_value=False):
        local_path = download_file_if_necessary(run, remote_filename, expected_local_path)
        assert local_path is not None
        run.download_file.assert_not_called()


@pytest.mark.parametrize(
    ['files', 'expected_downloaded'],
    [([], []), (["a.txt", "b.txt"], ["a.txt", "b.txt"]), (["e.txt", "f.csv"], ["e.txt"])],
)
def test_download_files_by_suffix(tmp_path: Path, files: List[str], expected_downloaded: List[str]) -> None:
    """Test downloading files from a run returning a Generator

    :param files: The files that should be available in the mocked run.
    :param expected_downloaded: The names of the files that should be downloaded (filtered)
    """

    def mock_download(run: Run, file: str, output_file: Path, validate_checksum: bool) -> None:
        output_file.write_text("mock content")

    with mock.patch("health_azure.utils.get_run_file_names", return_value=files):
        with mock.patch("health_azure.utils._download_file_from_run", side_effect=mock_download):
            downloaded = download_files_by_suffix("outputs", tmp_path, ".txt")
            assert isinstance(downloaded, Generator)
            downloaded_list = list(downloaded)
            assert len(downloaded_list) == len(expected_downloaded)
            for f in downloaded_list:
                assert f.is_file()
            downloaded_filenames = [f.name for f in downloaded_list]
            assert downloaded_filenames == expected_downloaded


def test_resolve_workspace_config_path_no_argument(tmp_path: Path) -> None:
    """Test for resolve_workspace_config_path without argument: It should try to find a config file in the folders.
    If the file exists, it should return the path"""
    mocked_file = tmp_path / "foo.json"
    with patch("health_azure.utils.find_file_in_parent_to_pythonpath", return_value=mocked_file):
        result = resolve_workspace_config_path()
        assert result == mocked_file


def test_resolve_workspace_config_path_no_argument_no_file() -> None:
    """Test for resolve_workspace_config_path without argument: It should try to find a config file in the folders.
    If the file does not exist, return None"""
    with patch("health_azure.utils.find_file_in_parent_to_pythonpath", return_value=None):
        result = resolve_workspace_config_path()
        assert result is None


def test_resolve_workspace_config_path_file_exists(tmp_path: Path) -> None:
    mocked_file = tmp_path / "foo.json"
    mocked_file.touch()
    result = resolve_workspace_config_path(mocked_file)
    assert result == mocked_file


def test_resolve_workspace_config_path_missing(tmp_path: Path) -> None:
    mocked_file = tmp_path / "foo.json"
    with pytest.raises(FileNotFoundError, match="Workspace config file does not exist"):
        resolve_workspace_config_path(mocked_file)


@pytest.mark.fast
def test_sanitize_snapshot_root(tmp_path: Path) -> None:
    """Test if the snapshot root directory will default to the current working directory if not provided.
    Otherwise, pass through unchanged.
    """
    with change_working_directory(tmp_path):
        assert sanitize_snapshoot_directory(None) == tmp_path
    folder = tmp_path / "foo"
    assert sanitize_snapshoot_directory(folder) == folder
    assert sanitize_snapshoot_directory(str(folder)) == folder


@pytest.mark.fast
def test_sanitize_entry_script(tmp_path: Path) -> None:
    script = "some_script"
    # If no entry script is given, derive from sys.argv
    with patch("sys.argv", [script]):
        assert sanitize_entry_script(None, tmp_path) == script
    # If an entry script is given and it is valid, pass through unchanged
    assert sanitize_entry_script(script, tmp_path) == script
    # If the entry script is in a subfolder, we want to get the script with folder back, as a string.
    folder = "folder"
    script_with_folder = tmp_path / folder / script
    assert sanitize_entry_script(script_with_folder, tmp_path) == f"{folder}/{script}"
    # If the entry script is of the format "-m Foo.bar", return unchanged.
    entry_module = "-m Foo.bar"
    assert sanitize_entry_script(entry_module, tmp_path) == entry_module

    # Error case: Invalid argument
    with pytest.raises(ValueError, match="entry_script must be a string or Path"):
        assert sanitize_entry_script([], tmp_path)  # type: ignore
    # Error case: Entry script is not in the snapshot
    with pytest.raises(ValueError, match="entry script must be inside of the snapshot root"):
        assert sanitize_entry_script(script_with_folder, tmp_path / "other_folder")


@pytest.mark.fast
def test_generate_unique_environment_name_from_directory(tmp_path: Path) -> None:
    # Create a sample subdirectory and test that each file is read
    folder_path = mock.MagicMock(spec_set=Path)
    folder_path.is_file.return_value = False
    folder_path.is_dir.return_value = True
    folder_path.rglob.return_value = [Path("file_1"), Path("file_2")]
    with patch("health_azure.utils.Path.is_file", side_effect=[True, True]):
        with patch("health_azure.utils.Path.open", mock_open(read_data=b'foo'), create=True) as m_open:
            _ = generate_unique_environment_name(folder_path)
            # Ensure each file's read method was called
            assert m_open.call_count == 2
            # Verify that the read method was called for each file
            handle = m_open()
            handle.read.assert_called()
            assert handle.read.call_count == 2

    # Create two subdirectories. Put the same files in there and check that the hashes are the same
    subdir_1 = tmp_path / "subdir_1"
    subdir_2 = tmp_path / "subdir_2"
    subdir_1.mkdir()
    subdir_2.mkdir()
    file_names = ["file1.txt", "file2.txt", "file3.txt"]
    file_contents = ["Content of file 1", "Content of file 2", "Content of file 3"]
    for file_name, content in zip(file_names, file_contents):
        (subdir_1 / file_name).write_text(content)
        (subdir_2 / file_name).write_text(content)
    hash_1 = generate_unique_environment_name(subdir_1)
    hash_2 = generate_unique_environment_name(subdir_2)
    assert hash_1 == hash_2

    # Create a third folder with different contents and check that the hashes are different
    subdir_3 = tmp_path / "subdir_3"
    subdir_3.mkdir()
    file_names = ["file1.txt", "file2.txt", "file3.txt"]
    file_contents = ["new content of file 1", "New content of file 2", "New content of file 3"]
    for file_name, content in zip(file_names, file_contents):
        (subdir_3 / file_name).write_text(content)
    hash_3 = generate_unique_environment_name(subdir_3)
    assert hash_1 != hash_3
