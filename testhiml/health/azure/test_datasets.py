from unittest import mock

import pytest
from azureml.data import OutputFileDatasetConfig
from azureml.data.azure_storage_datastore import AzureBlobDatastore
from azureml.data.dataset_consumption_config import DatasetConsumptionConfig

from health.azure.datasets import DatasetConfig
from health.azure.datasets import get_datastore
from testhiml.health.azure.utils import DEFAULT_DATASTORE
from testhiml.health.azure.utils import DEFAULT_WORKSPACE


def test_datasetconfig_init() -> None:
    with pytest.raises(ValueError) as ex:
        DatasetConfig(name=" ")
    assert "name of the dataset must be a non-empty string" in str(ex)


def test_get_datastore_fails() -> None:
    """
    Retrieving a datastore that does not exist should fail
    """
    does_not_exist = "does_not_exist"
    with pytest.raises(ValueError) as ex:
        get_datastore(workspace=DEFAULT_WORKSPACE, datastore_name=does_not_exist)
    assert f"Datastore {does_not_exist} was not found" in str(ex)


def test_get_datastore_without_name() -> None:
    """
    Trying to get a datastore without name should only work if there is a single datastore
    """
    assert len(DEFAULT_WORKSPACE.datastores) > 1
    with pytest.raises(ValueError) as ex:
        get_datastore(workspace=DEFAULT_WORKSPACE, datastore_name="")
    assert "No datastore name provided" in str(ex)


def test_get_datastore() -> None:
    """
    Tests getting a datastore by name.
    """
    name = DEFAULT_DATASTORE
    datastore = get_datastore(workspace=DEFAULT_WORKSPACE, datastore_name=name)
    assert isinstance(datastore, AzureBlobDatastore)
    assert datastore.name == name
    assert len(DEFAULT_WORKSPACE.datastores) > 1
    # Now mock the datastores property of the workspace, to pretend there is only a single datastore.
    # With that in place, we can get the datastore without the name
    faked_stores = {name: datastore}
    with mock.patch("azureml.core.Workspace.datastores", faked_stores):
        single_store = get_datastore(workspace=DEFAULT_WORKSPACE, datastore_name="")
    assert isinstance(single_store, AzureBlobDatastore)
    assert single_store.name == name


def test_dataset_input() -> None:
    """
    Test turning a dataset setup object to an actual AML input dataset.
    """
    dataset_config = DatasetConfig(name="hello_world", datastore=DEFAULT_DATASTORE)
    aml_dataset = dataset_config.to_input_dataset(workspace=DEFAULT_WORKSPACE, dataset_index=1)
    assert isinstance(aml_dataset, DatasetConsumptionConfig)
    assert aml_dataset.path_on_compute is None
    assert aml_dataset.mode == "download"
    # Downloading or mounting to a given path
    target_folder = "/tmp/foo"
    dataset_config = DatasetConfig(name="hello_world", datastore=DEFAULT_DATASTORE, target_folder=target_folder)
    aml_dataset = dataset_config.to_input_dataset(workspace=DEFAULT_WORKSPACE, dataset_index=1)
    assert isinstance(aml_dataset, DatasetConsumptionConfig)
    assert aml_dataset.path_on_compute == target_folder
    # Use mounting instead of downloading
    dataset_config = DatasetConfig(name="hello_world", datastore=DEFAULT_DATASTORE, use_mounting=True)
    aml_dataset = dataset_config.to_input_dataset(workspace=DEFAULT_WORKSPACE, dataset_index=1)
    assert isinstance(aml_dataset, DatasetConsumptionConfig)
    assert aml_dataset.mode == "mount"


def test_dataset_output() -> None:
    """
    Test turning a dataset setup object to an actual AML output dataset.
    """
    name = "new_dataset"
    dataset_config = DatasetConfig(name=name, datastore=DEFAULT_DATASTORE)
    aml_dataset = dataset_config.to_output_dataset(workspace=DEFAULT_WORKSPACE, dataset_index=1)
    assert isinstance(aml_dataset, OutputFileDatasetConfig)
    assert isinstance(aml_dataset.destination, tuple)
    assert aml_dataset.destination[0]["name"] == DEFAULT_DATASTORE
    assert aml_dataset.destination[1] == name
    assert aml_dataset.mode == "mount"
    # Use downloading instead of mounting
    dataset_config = DatasetConfig(name="hello_world", datastore=DEFAULT_DATASTORE, use_mounting=False)
    aml_dataset = dataset_config.to_input_dataset(workspace=DEFAULT_WORKSPACE, dataset_index=1)
    assert aml_dataset.mode == "download"
    # Mounting at a fixed folder is not possible
    with pytest.raises(ValueError) as ex:
        dataset_config = DatasetConfig(name=name, datastore=DEFAULT_DATASTORE, target_folder="something")
        dataset_config.to_output_dataset(workspace=DEFAULT_WORKSPACE, dataset_index=1)
    assert "Output datasets can't have a target_folder set" in str(ex)
