#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
Test the data input and output functionality
"""
from unittest import mock

import pytest
from azureml.core import Dataset
from azureml.data import FileDataset, OutputFileDatasetConfig
from azureml.data.azure_storage_datastore import AzureBlobDatastore
from azureml.data.dataset_consumption_config import DatasetConsumptionConfig
from health.azure.datasets import (DatasetConfig, _input_dataset_key, _output_dataset_key,
                                   _replace_string_datasets, get_datastore, get_or_create_dataset)
from testhiml.health.azure.util import DEFAULT_DATASTORE, DEFAULT_WORKSPACE, default_aml_workspace


def test_datasetconfig_init() -> None:
    with pytest.raises(ValueError) as ex:
        DatasetConfig(name=" ")
    assert "name of the dataset must be a non-empty string" in str(ex)


def test_get_datastore() -> None:
    """
    Test retrieving a datastore from the AML workspace.
    """
    # Retrieving a datastore that does not exist should fail
    does_not_exist = "does_not_exist"
    workspace = DEFAULT_WORKSPACE.workspace
    with pytest.raises(ValueError) as ex:
        get_datastore(workspace=workspace, datastore_name=does_not_exist)
    assert f"Datastore {does_not_exist} was not found" in str(ex)

    # Trying to get a datastore without name should only work if there is a single datastore
    assert len(workspace.datastores) > 1
    with pytest.raises(ValueError) as ex:
        get_datastore(workspace=workspace, datastore_name="")
    assert "No datastore name provided" in str(ex)

    # Retrieve a datastore by name
    name = DEFAULT_DATASTORE
    datastore = get_datastore(workspace=workspace, datastore_name=name)
    assert isinstance(datastore, AzureBlobDatastore)
    assert datastore.name == name
    assert len(workspace.datastores) > 1
    # Now mock the datastores property of the workspace, to pretend there is only a single datastore.
    # With that in place, we can get the datastore without the name
    faked_stores = {name: datastore}
    with mock.patch("azureml.core.Workspace.datastores", faked_stores):
        single_store = get_datastore(workspace=workspace, datastore_name="")
    assert isinstance(single_store, AzureBlobDatastore)
    assert single_store.name == name


def test_dataset_input() -> None:
    """
    Test turning a dataset setup object to an actual AML input dataset.
    """
    workspace = DEFAULT_WORKSPACE.workspace
    # This dataset must exist in the workspace already, or at least in blob storage.
    dataset_config = DatasetConfig(name="hello_world", datastore=DEFAULT_DATASTORE)
    aml_dataset = dataset_config.to_input_dataset(workspace=workspace, dataset_index=1)
    assert isinstance(aml_dataset, DatasetConsumptionConfig)
    assert aml_dataset.path_on_compute is None
    assert aml_dataset.mode == "download"
    # Downloading or mounting to a given path
    target_folder = "/tmp/foo"
    dataset_config = DatasetConfig(name="hello_world", datastore=DEFAULT_DATASTORE, target_folder=target_folder)
    aml_dataset = dataset_config.to_input_dataset(workspace=workspace, dataset_index=1)
    assert isinstance(aml_dataset, DatasetConsumptionConfig)
    assert aml_dataset.path_on_compute == target_folder
    # Use mounting instead of downloading
    dataset_config = DatasetConfig(name="hello_world", datastore=DEFAULT_DATASTORE, use_mounting=True)
    aml_dataset = dataset_config.to_input_dataset(workspace=workspace, dataset_index=1)
    assert isinstance(aml_dataset, DatasetConsumptionConfig)
    assert aml_dataset.mode == "mount"


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
    original = [dataset1, DatasetConfig(name=dataset2, datastore=store)]
    replaced = _replace_string_datasets(original, default_datastore_name=default_store)
    assert len(replaced) == len(original)
    for d in replaced:
        assert isinstance(d, DatasetConfig)
    assert replaced[0].name == dataset1
    assert replaced[0].datastore == default_store
    assert replaced[1] == original[1]


def test_get_dataset() -> None:
    """
    Test if a dataset that does not yet exist can be created from a folder in blob storage
    """
    # A folder with a single tiny file
    tiny_dataset = "himl-tiny_dataset"
    workspace = default_aml_workspace()
    # When creating a dataset, we need a non-empty name
    with pytest.raises(ValueError) as ex:
        get_or_create_dataset(workspace=workspace,
                              datastore_name=DEFAULT_DATASTORE,
                              dataset_name="")
    assert "No dataset name" in str(ex)
    # Check first that there is no dataset yet of that name. If there is, delete that dataset (it would come
    # from previous runs of this test)
    try:
        existing_dataset = Dataset.get_by_name(workspace, name=tiny_dataset)
        existing_dataset.unregister_all_versions()
    except Exception as ex:
        assert "Cannot find dataset registered" in str(ex)
    dataset = get_or_create_dataset(workspace=workspace,
                                    datastore_name=DEFAULT_DATASTORE,
                                    dataset_name=tiny_dataset)
    assert isinstance(dataset, FileDataset)
    # We should now be able to get that dataset without special means
    dataset2 = Dataset.get_by_name(workspace, name=tiny_dataset)
    # Delete the dataset again
    dataset2.unregister_all_versions()


def test_dataset_keys() -> None:
    """
    Check that dataset keys are non-empty strings, and that inputs and outputs have different keys.
    """
    in1 = _input_dataset_key(1)
    out1 = _output_dataset_key(1)
    assert in1
    assert out1
    assert in1 != out1
