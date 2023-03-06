#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
Test the data input and output functionality
"""
from pathlib import Path
from typing import List, Optional, Union
from unittest.mock import DEFAULT, MagicMock, create_autospec, patch

import pytest
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.operations import DatastoreOperations
from azure.core.exceptions import HttpResponseError, ResourceNotFoundError
from azureml._restclient.exceptions import ServiceException
from azureml.core import Dataset, Workspace
from azureml.data import FileDataset, OutputFileDatasetConfig
from azureml.data.azure_storage_datastore import AzureBlobDatastore
from azureml.data.dataset_consumption_config import DatasetConsumptionConfig
from azureml.exceptions._azureml_exception import UserErrorException
from testazure.utils_testazure import (
    DEFAULT_DATASTORE, DEFAULT_WORKSPACE, TEST_DATASET_NAME, TEST_DATA_ASSET_NAME, TEST_INVALID_DATA_ASSET_NAME,
    TEST_DATASTORE_NAME, get_test_ml_client,
)

from health_azure.datasets import (
    DatasetConfig, _create_v1_dataset, _create_v2_data_asset, _get_or_create_v1_dataset, _get_or_create_v2_data_asset,
    _input_dataset_key, _output_dataset_key, _replace_string_datasets, _retrieve_v1_dataset, _retrieve_v2_data_asset,
    create_dataset_configs, get_datastore, get_or_create_dataset, _get_latest_v2_asset_version,
)
from health_azure.utils import PathOrString, get_ml_client


TEST_ML_CLIENT = get_test_ml_client()


def test_datasetconfig_init() -> None:
    with pytest.raises(ValueError) as ex:
        DatasetConfig(name=" ")
    assert "name of the dataset must be a non-empty string" in str(ex)


def test_get_datastore_v1() -> None:
    """
    Test retrieving a datastore from the AML workspace using the v1 SDK.
    """
    # Retrieving a datastore that does not exist should fail
    does_not_exist = "does_not_exist"
    workspace = DEFAULT_WORKSPACE.workspace
    with pytest.raises(ValueError) as ex:
        get_datastore(workspace=workspace, datastore_name=does_not_exist)
    assert f"Datastore \"{does_not_exist}\" was not found" in str(ex)

    # Trying to get a datastore when no name is specified should return the workspace default datastore
    assert len(workspace.datastores) > 1
    default_datastore = get_datastore(workspace=workspace, datastore_name="")
    assert default_datastore is not None
    assert default_datastore.name == workspace.get_default_datastore().name

    # Retrieve a datastore by name
    name = DEFAULT_DATASTORE
    datastore = get_datastore(workspace=workspace, datastore_name=name)
    assert isinstance(datastore, AzureBlobDatastore)
    assert datastore.name == name
    assert len(workspace.datastores) > 1
    # Now mock the datastores property of the workspace, to pretend there is only a single datastore.
    # With that in place, we can get the datastore without the name
    faked_stores = {name: datastore}
    with patch("azureml.core.Workspace.datastores", faked_stores):
        single_store = get_datastore(workspace=workspace, datastore_name="")
    assert isinstance(single_store, AzureBlobDatastore)
    assert single_store.name == name


def test_get_datastore_v2() -> None:
    """
    Test retrieving a datastore from the AML workspace using the v2 SDK.
    """
    # Now test retrieving a v2 datastore by name
    mock_v2_datastore_name = "dummy_v2_datastore"
    mock_returned_datastore = MagicMock()
    mock_returned_datastore.name = mock_v2_datastore_name
    mock_workspace = MagicMock()
    mock_workspace.datastores = create_autospec(DatastoreOperations)
    mock_workspace.datastores.get.return_value = mock_returned_datastore
    v2_datastore = get_datastore(mock_workspace, datastore_name=mock_v2_datastore_name)
    assert v2_datastore.name == mock_v2_datastore_name

    # Test retrieving a default v2 datastore
    mock_workspace.datastores.list.return_value = [mock_returned_datastore]
    v2_default_datastore = get_datastore(mock_workspace, datastore_name="")
    assert v2_default_datastore.name == mock_v2_datastore_name

    # Mock case where list is empty but get_default returns a value
    mock_workspace.datastores.list.return_value = []
    mock_workspace.datastores.get_default.return_value = mock_returned_datastore
    v2_default_datastore = get_datastore(mock_workspace, datastore_name="")
    assert v2_default_datastore.name == mock_v2_datastore_name

    # If datastores has an unknown format, an exception should be raised
    mock_workspace = MagicMock()
    mock_workspace.datastores.return_value = ["dummy_datastore_name"]
    with pytest.raises(Exception) as e:
        get_datastore(workspace=mock_workspace, datastore_name="")
        assert "Unrecognised type for datastores" in str(e)


def test_dataset_input() -> None:
    """
    Test turning a dataset setup object to an actual AML input dataset.
    """
    workspace = DEFAULT_WORKSPACE.workspace
    # This dataset must exist in the workspace already, or at least in blob storage.
    dataset_config = DatasetConfig(name="hello_world", datastore=DEFAULT_DATASTORE)
    aml_dataset = dataset_config.to_input_dataset(dataset_index=1, workspace=workspace, strictly_aml_v1=True)
    assert isinstance(aml_dataset, DatasetConsumptionConfig)
    assert aml_dataset.path_on_compute is None  # type: ignore
    assert aml_dataset.mode == "download"  # type: ignore
    # Downloading or mounting to a given path
    target_folder = "/tmp/foo"
    dataset_config = DatasetConfig(name="hello_world", datastore=DEFAULT_DATASTORE, target_folder=target_folder)
    aml_dataset = dataset_config.to_input_dataset(
        dataset_index=1, workspace=workspace, strictly_aml_v1=True)
    assert isinstance(aml_dataset, DatasetConsumptionConfig)
    assert aml_dataset.path_on_compute == target_folder  # type: ignore
    # Use mounting instead of downloading
    dataset_config = DatasetConfig(name="hello_world", datastore=DEFAULT_DATASTORE, use_mounting=True)
    aml_dataset = dataset_config.to_input_dataset(
        dataset_index=1, workspace=workspace, strictly_aml_v1=True)
    assert isinstance(aml_dataset, DatasetConsumptionConfig)
    assert aml_dataset.mode == "mount"  # type: ignore


@pytest.mark.parametrize("target_folder", [
    "",
    None,
])
def test_dataset_input_target_empty(target_folder: PathOrString) -> None:
    """
    Leaving the target folder empty should NOT create a path_on_compute that is "."
    """
    workspace = DEFAULT_WORKSPACE.workspace
    # This dataset must exist in the workspace already, or at least in blob storage.
    dataset_config = DatasetConfig(name="hello_world", datastore=DEFAULT_DATASTORE, target_folder=target_folder)
    aml_dataset: DatasetConsumptionConfig = dataset_config.to_input_dataset(
        workspace=workspace, dataset_index=1, strictly_aml_v1=True)
    assert isinstance(aml_dataset, DatasetConsumptionConfig)
    assert aml_dataset.path_on_compute is None


@pytest.mark.parametrize("target_folder", [
    ".",
    Path(),
    Path("."),
])
def test_dataset_invalid_target(target_folder: PathOrString) -> None:
    """
    Passing in "." as a target_folder shouold raise an exception.
    """
    with pytest.raises(ValueError) as ex:
        DatasetConfig(name="hello_world", datastore=DEFAULT_DATASTORE, target_folder=target_folder)
    assert "current working directory" in str(ex)


def test_dataset_output() -> None:
    """
    Test turning a dataset setup object to an actual AML output dataset.
    """
    name = "new_dataset"
    workspace = DEFAULT_WORKSPACE.workspace
    dataset_config = DatasetConfig(name=name, datastore=DEFAULT_DATASTORE)
    aml_dataset = dataset_config.to_output_dataset(workspace=workspace, dataset_index=1)
    assert isinstance(aml_dataset, OutputFileDatasetConfig)
    assert isinstance(aml_dataset.destination, tuple)
    assert aml_dataset.destination[0].name == DEFAULT_DATASTORE
    assert aml_dataset.destination[1] == name + "/"
    assert aml_dataset.mode == "mount"
    # Use downloading instead of mounting
    dataset_config = DatasetConfig(name="hello_world", datastore=DEFAULT_DATASTORE, use_mounting=False)
    aml_dataset = dataset_config.to_output_dataset(workspace=workspace, dataset_index=1)
    assert isinstance(aml_dataset, OutputFileDatasetConfig)
    assert aml_dataset.mode == "upload"
    # Mounting at a fixed folder is not possible
    with pytest.raises(ValueError) as ex:
        dataset_config = DatasetConfig(name=name, datastore=DEFAULT_DATASTORE, target_folder="something")
        dataset_config.to_output_dataset(workspace=workspace, dataset_index=1)
    assert "Output datasets can't have a target_folder set" in str(ex)


def test_datasets_from_string() -> None:
    """
    Test the conversion of datasets that are only specified as strings.
    """
    dataset1 = "foo"
    dataset2 = "bar"
    store = "store"
    default_store = "default"
    original: List[Union[str, DatasetConfig]] = [dataset1, DatasetConfig(name=dataset2, datastore=store)]
    replaced = _replace_string_datasets(original, default_datastore_name=default_store)
    assert len(replaced) == len(original)
    for d in replaced:
        assert isinstance(d, DatasetConfig)
    assert replaced[0].name == dataset1
    assert replaced[0].datastore == default_store
    assert replaced[1] == original[1]


def test_get_or_create_dataset() -> None:
    def _mock_retrieve_or_create_v2_dataset_fails(
        datastore_name: str, dataset_name: str, ml_client: MLClient
    ) -> None:
        raise HttpResponseError("Cannot create v2 Data Version in v1 Data Container")

    data_asset_name = "himl_tiny_data_asset"
    workspace = DEFAULT_WORKSPACE.workspace
    ml_client = get_ml_client(aml_workspace=workspace)
    # When creating a dataset, we need a non-empty name
    with pytest.raises(ValueError) as ex:
        get_or_create_dataset(workspace=workspace,
                              ml_client=ml_client,
                              datastore_name="himldatasetsv2",
                              dataset_name="",
                              strictly_aml_v1=True)
    assert "No dataset name" in str(ex)

    # pass strictly_aml_v1 = True and check the expected function is called
    mock_v1_dataset = "v1_dataset"
    with patch.multiple("health_azure.datasets",
                        _get_or_create_v1_dataset=DEFAULT,
                        _get_or_create_v2_data_asset=DEFAULT) as mocks:
        mocks["_get_or_create_v1_dataset"].return_value = mock_v1_dataset
        dataset = get_or_create_dataset(workspace=workspace,
                                        ml_client=ml_client,
                                        datastore_name="himldatasetsv2",
                                        dataset_name=data_asset_name,
                                        strictly_aml_v1=True)
        mocks["_get_or_create_v1_dataset"].assert_called_once()
        mocks["_get_or_create_v2_data_asset"].assert_not_called()
        assert dataset == mock_v1_dataset

        # Now pass strictly_aml_v1 as False
        mock_v2_dataset = "v2_dataset"
        mocks["_get_or_create_v2_data_asset"].return_value = mock_v2_dataset
        dataset = get_or_create_dataset(workspace=workspace,
                                        ml_client=ml_client,
                                        datastore_name="himldatasetsv2",
                                        dataset_name=data_asset_name,
                                        strictly_aml_v1=False)
        mocks["_get_or_create_v1_dataset"].assert_called_once()
        mocks["_get_or_create_v2_data_asset"].assert_called_once()
        assert dataset == mock_v2_dataset

        # if  trying to get or create a v2 dataset fails, should revert back to _get_or_create_v1_dataset
        mocks["_get_or_create_v2_data_asset"].side_effect = _mock_retrieve_or_create_v2_dataset_fails
        dataset = get_or_create_dataset(workspace=workspace,
                                        ml_client=ml_client,
                                        datastore_name="himldatasetsv2",
                                        dataset_name=data_asset_name,
                                        strictly_aml_v1=False)
        assert mocks["_get_or_create_v1_dataset"].call_count == 2
        assert mocks["_get_or_create_v2_data_asset"].call_count == 2
        assert dataset == mock_v1_dataset


def test_get_or_create_v1_dataset() -> None:
    def _mock_error_from_retrieve_v1_dataset(dataset_name: str, workspace: Workspace) -> None:
        raise UserErrorException("Error Message")

    workspace = DEFAULT_WORKSPACE.workspace
    datastore = workspace.get_default_datastore()
    dataset_name = "foo"

    with patch.multiple("health_azure.datasets",
                        _retrieve_v1_dataset=DEFAULT,
                        _create_v1_dataset=DEFAULT) as mocks:
        _get_or_create_v1_dataset(datastore, dataset_name, workspace)
        mocks["_retrieve_v1_dataset"].assert_called_once()
        mocks["_create_v1_dataset"].assert_not_called()

        mocks["_retrieve_v1_dataset"].side_effect = _mock_error_from_retrieve_v1_dataset
        _get_or_create_v1_dataset(datastore, dataset_name, workspace)
        assert mocks["_retrieve_v1_dataset"].call_count == 2
        mocks["_create_v1_dataset"].assert_called_once()


def test_get_or_create_v2_data_asset() -> None:
    def _mock_error_from_retrieve_v2_data_asset(
        ml_client: MLClient,
        data_asset_name: str,
        version: Optional[str] = None,
    ) -> None:
        raise ResourceNotFoundError("Error Message")

    ml_client = MagicMock()
    datastore = "dummy_datastore"
    dataset_name = "foo"

    with patch.multiple("health_azure.datasets",
                        _retrieve_v2_data_asset=DEFAULT,
                        _create_v2_data_asset=DEFAULT) as mocks:
        _get_or_create_v2_data_asset(ml_client, datastore, dataset_name)
        mocks["_retrieve_v2_data_asset"].assert_called_once()
        mocks["_create_v2_data_asset"].assert_not_called()

        mocks["_retrieve_v2_data_asset"].side_effect = _mock_error_from_retrieve_v2_data_asset
        _get_or_create_v2_data_asset(ml_client, datastore, dataset_name)
        assert mocks["_retrieve_v2_data_asset"].call_count == 2
        mocks["_create_v2_data_asset"].assert_called_once()


def test_retrieve_v1_dataset() -> None:
    nonexistent_dataset = "idontexist"
    workspace = DEFAULT_WORKSPACE.workspace
    # patch get_by_name to ensure it is called
    with patch("azureml.core.Dataset.get_by_name") as mock_get_dataset:
        _retrieve_v1_dataset(nonexistent_dataset, workspace)
        mock_get_dataset.assert_called_once()

    # Expect a ValueError to be raised if the dataset doesnt exist
    with pytest.raises(Exception) as e:
        _retrieve_v1_dataset(nonexistent_dataset, workspace)
        assert "Cannot find dataset registered with name \"idontexist\"" in str(e)


def test_create_v1_dataset() -> None:
    # If dataset_name or datastore_name are empty strings expect an Exception
    empty_dataset_name = ""
    empty_datastore_name = ""
    nonempty_dataset_name = "foo"
    nonempty_datastore_name = "bar"

    workspace = DEFAULT_WORKSPACE.workspace
    tiny_dataset = "himl_tiny_dataset"

    with pytest.raises(Exception) as e:
        _create_v1_dataset(empty_datastore_name, nonempty_dataset_name, workspace)
        expected_str = "Cannot create dataset without a valid datastore name (received '') and a valid dataset name"
        f" (received '{nonempty_dataset_name}')"
        assert expected_str in str(e)

        _create_v1_dataset(nonempty_datastore_name, empty_dataset_name, workspace)
        expected_str = f"Cannot create dataset without a valid datastore name (received '{empty_dataset_name}') and "
        "a valid dataset name (received '')"
        assert expected_str in str(e)

    try:
        existing_dataset = Dataset.get_by_name(workspace, name=tiny_dataset)
        try:
            existing_dataset.unregister_all_versions()
        except ServiceException:
            # Sometimes unregister_all_versions() raises a ServiceException.
            pass
    except Exception as ex:
        assert "Cannot find dataset registered" in str(ex)

    dataset = _create_v1_dataset(DEFAULT_DATASTORE, tiny_dataset, workspace)
    assert isinstance(dataset, FileDataset)

    # We should now be able to get that dataset without special means
    dataset2 = Dataset.get_by_name(workspace, name=tiny_dataset)
    try:
        # Delete the dataset again
        dataset2.unregister_all_versions()
    except (ServiceException, UserErrorException):
        # Sometimes unregister_all_versions() raises a ServiceException or a UserErrorException.
        pass


@pytest.mark.parametrize(["asset_name", "asset_version"],
                         [(TEST_DATA_ASSET_NAME, None),
                          (TEST_DATA_ASSET_NAME, "1"),
                          (TEST_INVALID_DATA_ASSET_NAME, None),
                          (TEST_INVALID_DATA_ASSET_NAME, "1")])
def test_retrieve_v2_data_asset(asset_name: str, asset_version: Optional[str]) -> None:
    with patch("health_azure.datasets._get_latest_v2_asset_version") as mock_get_v2_asset_version:
        mock_get_v2_asset_version.side_effect = _get_latest_v2_asset_version
        try:
            data_asset = _retrieve_v2_data_asset(
                ml_client=TEST_ML_CLIENT,
                data_asset_name=asset_name,
                version=asset_version,
            )
        except ResourceNotFoundError as ex:
            if asset_name == TEST_INVALID_DATA_ASSET_NAME:
                if asset_version is None:
                    expected_error_message = f"{TEST_INVALID_DATA_ASSET_NAME} container was not found."
                else:
                    expected_error_message = \
                        f"{TEST_INVALID_DATA_ASSET_NAME}:{asset_version} (dataContainerName:version) not found."

                assert expected_error_message in str(ex)
            else:
                pytest.fail(f"Unexpected error: {ex}")
        else:
            assert isinstance(data_asset, Data)
            if asset_version is not None:
                assert data_asset.version == asset_version
            else:
                mock_get_v2_asset_version.assert_called_once()
            data_asset_path = data_asset.path
            assert isinstance(data_asset_path, str)  # makes pyright happy
            assert f"{TEST_DATASTORE_NAME}/paths/{TEST_DATA_ASSET_NAME}" in data_asset_path


def test_retrieve_v2_data_asset_invalid_version() -> None:
    invalid_asset_version = str(int(_get_latest_v2_asset_version(TEST_ML_CLIENT, TEST_DATA_ASSET_NAME)) + 1)
    with pytest.raises(ResourceNotFoundError) as ex:
        _retrieve_v2_data_asset(
            ml_client=TEST_ML_CLIENT,
            data_asset_name=TEST_DATA_ASSET_NAME,
            version=invalid_asset_version,
        )
        assert f"{TEST_DATA_ASSET_NAME}:{invalid_asset_version} (dataContainerName:version) not found." in str(ex)


def test_retrieving_v2_data_asset_does_not_increment() -> None:
    """Test if calling the get_or_create_data_asset on an existing asset does not increment the version number.
    """

    with patch("health_azure.datasets._create_v2_data_asset") as mock_create_v2_data_asset:
        asset_version_before_get_or_create = _get_latest_v2_asset_version(TEST_ML_CLIENT, TEST_DATA_ASSET_NAME)
        get_or_create_dataset(
            TEST_DATASTORE_NAME,
            TEST_DATA_ASSET_NAME,
            DEFAULT_WORKSPACE,
            strictly_aml_v1=False,
            ml_client=TEST_ML_CLIENT,
        )
        asset_version_after_get_or_create = _get_latest_v2_asset_version(TEST_ML_CLIENT, TEST_DATA_ASSET_NAME)

        mock_create_v2_data_asset.assert_not_called()
        assert asset_version_before_get_or_create == asset_version_after_get_or_create


@pytest.mark.parametrize(["asset_name", "version"],
                         [(TEST_DATA_ASSET_NAME, None),
                          (TEST_DATA_ASSET_NAME, "1"),
                          ("", 1)])
def test_create_v2_data_asset(asset_name: str, version: Optional[str]) -> None:
    try:
        data_asset = _create_v2_data_asset(
            ml_client=TEST_ML_CLIENT,
            datastore_name=TEST_DATASTORE_NAME,
            data_asset_name=asset_name,
            version=version,
        )
    except HttpResponseError as ex:
        if version is not None:
            assert "A data version with this name and version already exists" in str(ex)
        else:
            pytest.fail(f"Unexpected error: {ex}")

    except ValueError as ex:
        if asset_name == "":
            assert "Cannot create data asset with empty name." in str(ex)
        else:
            pytest.fail(f"Unexpected error: {ex}")

    else:
        assert isinstance(data_asset, Data)
        assert f"{TEST_DATASTORE_NAME}/paths/{TEST_DATA_ASSET_NAME}/" in str(data_asset.path)
        assert data_asset.type == "uri_folder"
        assert data_asset.name == TEST_DATASET_NAME
        if version:
            assert data_asset.version == version


def test_dataset_keys() -> None:
    """
    Check that dataset keys are non-e
    mpty strings, and that inputs and outputs have different keys.
    """
    in1 = _input_dataset_key(1)
    out1 = _output_dataset_key(1)
    assert in1
    assert out1
    assert in1 != out1


def test_create_dataset_configs() -> None:
    azure_datasets: List[str] = []
    dataset_mountpoints: List[str] = []
    local_datasets: List[Optional[Path]] = []
    datastore = None
    use_mounting = False
    datasets = create_dataset_configs(azure_datasets,
                                      dataset_mountpoints,
                                      local_datasets,
                                      datastore,
                                      use_mounting)
    assert datasets == []

    # if local_datasets is not empty but azure_datasets still is, expect an empty list
    local_datasets = [Path("dummy")]
    datasets = create_dataset_configs(azure_datasets,
                                      dataset_mountpoints,
                                      local_datasets,
                                      datastore,
                                      use_mounting)
    assert datasets == []

    with pytest.raises(Exception) as e:
        azure_datasets = ["dummy"]
        local_datasets = [Path("another_dummy"), Path("another_extra_dummy")]
        create_dataset_configs(azure_datasets,
                               dataset_mountpoints,
                               local_datasets,
                               datastore,
                               use_mounting)
        assert "Invalid dataset setup" in str(e)

    az_dataset_name = "dummy"
    azure_datasets = [az_dataset_name]
    local_datasets = [Path("another_dummy")]
    datasets = create_dataset_configs(azure_datasets,
                                      dataset_mountpoints,
                                      local_datasets,
                                      datastore,
                                      use_mounting)
    assert len(datasets) == 1
    assert isinstance(datasets[0], DatasetConfig)
    assert datasets[0].name == az_dataset_name

    # If azure dataset name is empty, should still create
    azure_datasets = [" "]
    with pytest.raises(Exception) as e:
        create_dataset_configs(azure_datasets,
                               dataset_mountpoints,
                               local_datasets,
                               datastore,
                               use_mounting)
        assert "Invalid dataset setup" in str(e)
