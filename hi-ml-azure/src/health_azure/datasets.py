#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.entities import Datastore as V2Datastore
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.operations import DatastoreOperations
from azure.core.exceptions import HttpResponseError, ResourceNotFoundError
from azureml.core import Dataset, Workspace, Datastore
from azureml.data import FileDataset, OutputFileDatasetConfig
from azureml.data.azure_storage_datastore import AzureBlobDatastore
from azureml.data.dataset_consumption_config import DatasetConsumptionConfig
from azureml.exceptions._azureml_exception import UserErrorException

from health_azure.utils import PathOrString, get_ml_client

logger = logging.getLogger(__name__)


V1OrV2DataType = Union[FileDataset, Data]


def get_datastore(workspace: Workspace, datastore_name: str) -> Union[AzureBlobDatastore, V2Datastore]:
    """
    Retrieves a datastore of a given name from an AzureML workspace. The datastore_name argument can be omitted if
    the workspace only contains a single datastore. Raises a ValueError if there is no datastore of the given name.

    :param workspace: The AzureML workspace to read from.
    :param datastore_name: The name of the datastore to retrieve.
    :return: An AzureML datastore.
    """

    def _retrieve_v1_datastore(datastores: Dict[str, Datastore], datastore_name: str) -> Datastore:
        # First check if there is only one datastore, which is then obviously unique.
        # Only then try to use the default datastore, because there may not be a default set.
        existing_stores = list(datastores.keys())
        if not datastore_name:
            if len(existing_stores) == 1:
                return datastores[existing_stores[0]]
            datastore = workspace.get_default_datastore()
            logger.info(f"Using the workspace default datastore {datastore.name} to access datasets.")
            return datastore

        if datastore_name in datastores:
            return datastores[datastore_name]
        raise ValueError(
            f"Datastore \"{datastore_name}\" was not found in the \"{workspace.name}\" workspace. "
            f"Existing datastores: {existing_stores}"
        )

    def _retrieve_v2_datastore(datastores: DatastoreOperations, datastore_name: str) -> V2Datastore:
        existing_stores = list(datastores.list())
        if not datastore_name:
            if len(existing_stores) == 1:
                return existing_stores[0]
            datastore = datastores.get_default()
            logger.info(f"Using the workspace default datastore {datastore.name} to access datasets.")
            return datastore

        try:
            datastore = datastores.get(datastore_name)
        except ResourceNotFoundError:
            raise ValueError(f"Datastore \"{datastore_name}\" was not found in the workspace")
        return datastore

    datastores = workspace.datastores
    if isinstance(datastores, DatastoreOperations):
        return _retrieve_v2_datastore(datastores, datastore_name)
    elif isinstance(datastores, dict):
        return _retrieve_v1_datastore(datastores, datastore_name)
    else:
        raise ValueError(f"Unrecognised type for datastores: {type(datastores)}")


def _retrieve_v1_dataset(dataset_name: str, workspace: Workspace) -> Optional[FileDataset]:
    """
    Retrieve an Azure ML v1 Dataset if it exists, otherwise return None

    :param dataset_name: The name of the Dataset to look for.
    :param workspace: An Azure ML Workspace object for retrieving the Dataset.
    :return: A Dataset object if it is found, else None.
    """
    logger.info(f"Trying to retrieve AzureML Dataset '{dataset_name}'")
    azureml_dataset = Dataset.get_by_name(workspace, name=dataset_name)
    return azureml_dataset


def _create_v1_dataset(datastore_name: str, dataset_name: str, workspace: Workspace) -> FileDataset:
    """
    Create a v1 Dataset in the specified Datastore

    :param datastore_name: The AML Datastore to create the Dataset in.
    :param dataset_name: The name of the Dataset to create.
    :param workspace: An AML Workspace object.
    :return: An Azure ML (v1) FileDataset object.
    """
    if not dataset_name:
        raise ValueError(f"Cannot create dataset without a valid dataset name (received '{dataset_name}')")

    datastore = get_datastore(workspace, datastore_name)

    assert isinstance(datastore, AzureBlobDatastore)
    logger.info(f"Creating a new dataset from data in folder '{dataset_name}' in the datastore")
    # Ensure that there is a / at the end of the file path, otherwise folder that share a prefix could create
    # trouble (for example, folders foo and foo_bar exist, and I'm trying to create a dataset from "foo")
    azureml_dataset = Dataset.File.from_files(path=(datastore, dataset_name + "/"))
    logger.info("Registering the dataset for future use.")
    azureml_dataset.register(workspace, name=dataset_name)
    return azureml_dataset


def _get_or_create_v1_dataset(datastore_name: str, dataset_name: str, workspace: Workspace) -> Dataset:
    """
    Attempt to retrieve a v1 Dataset object and return that, otherwise attempt to create and register
    a v1 Dataset and return that.

    :param datastore_name: The name of the Datastore to either retrieve or create and register the Dataset in.
    :param dataset_name: The name of the Dataset to be retrieved or registered.
    :param workspace: An Azure ML Workspace object.
    :return: An Azure ML Dataset object with the provided dataset name, in the provided datastore.
    """
    try:
        azureml_dataset = _retrieve_v1_dataset(dataset_name, workspace)
    except UserErrorException:
        logger.warning(f"Dataset '{dataset_name}' was not found, or is not an AzureML SDK v1 dataset.")
        logger.info(f"Trying to create a new dataset '{dataset_name}' from files in folder '{dataset_name}'")
        if datastore_name == "":
            raise ValueError(
                "When creating a new dataset, a datastore name must be provided. Please specify a datastore name using "
                "the --datastore flag"
            )
        logger.info(f"Trying to create a new dataset '{dataset_name}' in datastore '{datastore_name}'")
        azureml_dataset = _create_v1_dataset(datastore_name, dataset_name, workspace)
    return azureml_dataset


def _get_latest_v2_asset_version(ml_client: MLClient, data_asset_name: str) -> str:
    """
    Retrieve the version of a v2 data asset that has the highest value (version numbers are assumed to be integers).

    :param ml_client: An Azure MLClient object for interacting with Azure resources.
    :param data_asset_name: The name of the data asset to look for.
    :raises ResourceNotFoundError: If no data asset can be found with a matching name.
    :return: The latest version of the data asset if found, else None.
    """
    data_assets = ml_client.data.list(name=data_asset_name)
    highest_version = float('-inf')

    for data_asset in data_assets:
        try:
            data_asset_version = int(data_asset.version)
        except ValueError as val_er:
            logging.warning(f"Failed to convert data asset version to int: {val_er}")
            continue

        if data_asset_version > highest_version:
            highest_version = data_asset_version

    if highest_version == float('-inf'):
        raise ResourceNotFoundError(f"No data asset found with the provided name: {data_asset_name}")

    return str(highest_version)


def _retrieve_v2_data_asset(
    ml_client: MLClient,
    data_asset_name: str,
    version: Optional[str] = None,
) -> Data:
    """
    Attempt to retrieve a v2 Data Asset using a provided Azure ML Workspace connection. If
    no Data asset can be found with a matching name, the underlying code will raise an Exception

    :param ml_client: An Azure MLClient object for interacting with Azure resources.
    :param data_asset_name: The name of the data asset to look for.
    :param version: version of the data asset to retrieve, latest version retrieved if None.
    :return: An Azure Data asset representing the data asset if found.
    """

    if version is None:
        version = _get_latest_v2_asset_version(ml_client, data_asset_name)

    aml_data = ml_client.data.get(name=data_asset_name, version=version)
    assert aml_data.version == version, f"Data asset version ({aml_data.version}) does not match version ({version})"
    return aml_data


def _create_v2_data_asset(
    ml_client: MLClient, datastore_name: str, data_asset_name: str, version: Optional[str] = None
) -> Data:
    """
    Create or update a v2 Data asset in the specified Datastore

    :param ml_client: An Azure MLClient object for interacting with Azure resources.
    :param datastore_name: The name of the datastore in which to create or update the Data asset.
    :param data_asset_name: The name of the data asset to be created.
    :param version: The version of the data asset to be created.
    :raises ValueError: If no data asset name is provided to define the new asset.
    :raises ValueError: If no datastore name is provided to define where to create the data.
    :return: The created or updated Data asset.
    """
    if data_asset_name == "":
        raise ValueError("Cannot create data asset with empty name.")

    if datastore_name == "":
        raise ValueError(
            "Cannot create data asset with empty datastore name. Please specify a datastore name using the --datastore flag."
        )

    logger.info(
        f"Creating a new Data asset from data in folder '{data_asset_name}' in the datastore '{datastore_name}'"
    )
    azureml_data_asset = Data(
        path=f"azureml://datastores/{datastore_name}/paths/{data_asset_name}/",
        type=AssetTypes.URI_FOLDER,
        description="<description>",
        name=data_asset_name,
        version=version,
    )

    azureml_data_asset = ml_client.data.create_or_update(azureml_data_asset)
    return azureml_data_asset


def _get_or_create_v2_data_asset(
    ml_client: MLClient, datastore_name: str, data_asset_name: str, version: Optional[str] = None
) -> Data:
    """
    Attempt to retrieve a v2 data asset object and return that, otherwise attempt to create and register
    a v2 data asset and return that.

    :param ml_client: An Azure MLClient object for interacting with Azure resources.
    :param datastore_name: The name of the Datastore to either retrieve or create and register the Data asset in.
    :param data_asset_name: The name of the Data asset to be retrieved or registered.
    :return: An Azure Data asset object with the provided asset name, in the provided datastore
    """
    try:
        azureml_data_asset = _retrieve_v2_data_asset(ml_client, data_asset_name, version)
    except ResourceNotFoundError:  # catch the exception and create the dataset, raise all other types of exceptions
        logger.info(
            f"Data asset {data_asset_name} not found in datastore {datastore_name}. Version specified: {version}."
            "Attempting to create a new data asset with specified name and version."
        )
        azureml_data_asset = _create_v2_data_asset(ml_client, datastore_name, data_asset_name, version)

    return azureml_data_asset


def get_or_create_dataset(
    datastore_name: str,
    dataset_name: str,
    workspace: Workspace,
    strictly_aml_v1: bool,
    ml_client: Optional[MLClient] = None,
) -> V1OrV2DataType:
    """
    Looks in the AzureML datastore for a dataset of the given name. If there is no such dataset, a dataset is
    created and registered, assuming that the files are in a folder that has the same name as the dataset.
    For example, if dataset_name is 'foo', then the 'foo' dataset should be pointing to the folder
    <container_root>/datasets/dataset_name/.

    If the command line arg to strictly use AML SDK v1 is set to True, will attempt to retrieve a dataset using
    v1 of the SDK. Otherwise, will attempt to use v2 of the SDK. If no data of this name is found in the v2 datastore,
    will attempt to create it, but if the data container provided is v1 version, will fall back to using the
    v1 SDK to create and register this dataset.

    :param datastore_name: The name of the datastore in which to look for, or create and register, the dataset.
    :param dataset_name: The name of the dataset to find or create.
    :param workspace: An AML Workspace object for interacting with AML v1 datastores.
    :param strictly_aml_v1: If True, use Azure ML SDK v1 to attempt to find or create and reigster the dataset.
        Otherwise, attempt to use Azure ML SDK v2.
    :param ml_client: An optional MLClient object for interacting with AML v2 datastores.
    """
    if not dataset_name:
        raise ValueError("No dataset name provided.")
    if strictly_aml_v1:
        aml_dataset = _get_or_create_v1_dataset(datastore_name, dataset_name, workspace)
        return aml_dataset
    else:
        try:
            ml_client = get_ml_client(ml_client=ml_client)
            aml_dataset = _get_or_create_v2_data_asset(ml_client, datastore_name, dataset_name)
        except HttpResponseError as e:
            if "Cannot create v2 Data Version in v1 Data Container" in e.message:
                logger.info("This appears to be a v1 Data Container. Reverting to API v1 to create this Dataset")
            aml_dataset = _get_or_create_v1_dataset(datastore_name, dataset_name, workspace)

        return aml_dataset


def _input_dataset_key(index: int) -> str:
    return f"INPUT_{index}"


def _output_dataset_key(index: int) -> str:
    return f"OUTPUT_{index}"


class DatasetConfig:
    """
    Contains information to use AzureML datasets as inputs or outputs.
    """

    def __init__(
        self,
        name: str,
        datastore: str = "",
        overwrite_existing: bool = True,
        version: Optional[int] = None,
        use_mounting: Optional[bool] = None,
        target_folder: Optional[PathOrString] = None,
        local_folder: Optional[PathOrString] = None,
        data_name: Optional[str] = None,
    ):
        """
        :param name: The name of the dataset, as it was registered in the AzureML workspace. For output datasets,
            this will be the name given to the newly created dataset.
        :param datastore: The name of the AzureML datastore that holds the dataset. This can be empty if the AzureML
            workspace has only a single datastore, or if the default datastore should be used.
        :param overwrite_existing: Only applies to uploading datasets. If True, the dataset will be overwritten if it
            already exists. If False, the dataset creation will fail if the dataset already exists.
        :param version: The version of the dataset that should be used. This is only used for input datasets.
            If the version is not specified, the latest version will be used.
        :param use_mounting: If True, the dataset will be "mounted", that is, individual files will be read
            or written on-demand over the network. If False, the dataset will be fully downloaded before the job starts,
            respectively fully uploaded at job end for output datasets.
            Defaults: False (downloading) for datasets that are script inputs, True (mounting) for datasets that are
            script outputs.
        :param target_folder: The folder into which the dataset should be downloaded or mounted. If left empty, a
            random folder on /tmp will be chosen. Do NOT use "." as the target_folder.
        :param local_folder: The folder on the local machine at which the dataset is available. This
            is used only for runs outside of AzureML. If this is empty then the target_folder will be used to
            mount or download the dataset.
        :param data_name: Name of the input/output, used to infer the AML folder on which the data will be mounted or
            downloaded. For example, if the `data_name` of an input dataset is `"mydata_dir"`, the argument
            `'${{inputs.mydata_dir}}'` will be added to the script arguments to indicate the folder where the data is
            mounted/downloaded inside the run. If used for an output dataset, the argument `'${{outputs.mydata_dir}}'`
            may be used.
        """
        # This class would be a good candidate for a dataclass, but having an explicit constructor makes
        # documentation tools in the editor work nicer.
        name = name.strip()
        if not name:
            raise ValueError("The name of the dataset must be a non-empty string.")
        self.name = name
        self.datastore = datastore
        self.overwrite_existing = overwrite_existing
        self.version = version
        self.use_mounting = use_mounting
        # If target_folder is "" then convert to None
        self.target_folder = Path(target_folder) if target_folder else None
        if str(self.target_folder) == ".":
            raise ValueError("Can't mount or download a dataset to the current working directory.")
        self.local_folder = Path(local_folder) if local_folder else None
        self.data_name = data_name

    def to_input_dataset_local(
        self,
        workspace: Workspace,
    ) -> Tuple[Path, Any]:
        """
        Return a local path to the dataset when outside of an AzureML run.
        If local_folder is supplied, then this is assumed to be a local dataset, and this is returned.
        Otherwise the dataset is mounted or downloaded to either the target folder or a temporary folder and that is
        returned. If self.name refers to a v2 dataset, it is not possible to mount the data here,
        therefore a tuple of Nones will be returned.

        :param workspace: The AzureML workspace to read from.
        :return: Tuple of (path to dataset, optional mountcontext)
        """
        status = f"Dataset '{self.name}' will be "

        if self.local_folder is not None:
            status += f"obtained from local folder {str(self.local_folder)}"
            print(status)
            return self.local_folder, None

        if workspace is None:
            raise ValueError(
                f"Unable to make dataset '{self.name} available for a local run because no AzureML "
                "workspace has been provided. Provide a workspace, or set a folder for local execution."
            )
        azureml_dataset = _get_or_create_v1_dataset(
            datastore_name=self.datastore,
            dataset_name=self.name,
            workspace=workspace,
        )
        if isinstance(azureml_dataset, FileDataset):
            target_path = self.target_folder or Path(tempfile.mkdtemp())
            use_mounting = self.use_mounting if self.use_mounting is not None else False
            if use_mounting:
                status += f"mounted at {target_path}"

                mount_context = azureml_dataset.mount(mount_point=str(target_path))  # type: ignore
                result = target_path, mount_context
            else:
                status += f"downloaded to {target_path}"

                azureml_dataset.download(target_path=str(target_path), overwrite=False)  # type: ignore
                result = target_path, None
            print(status)
            return result
        else:
            raise ValueError(f"Don't know how to handle dataset '{self.name}' of type {type(azureml_dataset)}")

    def to_input_dataset(
        self,
        dataset_index: int,
        workspace: Workspace,
        strictly_aml_v1: bool,
        ml_client: Optional[MLClient] = None,
    ) -> Optional[DatasetConsumptionConfig]:
        """
        Creates a configuration for using an AzureML dataset inside of an AzureML run. This will make the AzureML
        dataset with given name available as a named input, using INPUT_0 as the key for dataset index 0.

        :param workspace: The AzureML workspace to read from.
        :param dataset_index: Suffix for using datasets as named inputs, the dataset will be marked INPUT_{index}
        :param strictly_aml_v1: If True, use Azure ML SDK v1. Otherwise, attempt to use Azure ML SDK v2.
        :param ml_client: An Azure MLClient object for interacting with Azure resources.
        """
        status = f"In AzureML, dataset {self.name} (index {dataset_index}) will be "
        azureml_dataset = get_or_create_dataset(
            datastore_name=self.datastore,
            dataset_name=self.name,
            workspace=workspace,
            strictly_aml_v1=strictly_aml_v1,
            ml_client=ml_client,
        )
        # If running on windows then self.target_folder may be a WindowsPath, make sure it is
        # in posix format for Azure.
        use_mounting = False if self.use_mounting is None else self.use_mounting
        if isinstance(azureml_dataset, FileDataset):
            named_input = azureml_dataset.as_named_input(_input_dataset_key(index=dataset_index))  # type: ignore
            path_on_compute = self.target_folder.as_posix() if self.target_folder is not None else None
            if use_mounting:
                status += "mounted at "
                result = named_input.as_mount(path_on_compute)
            else:
                status += "downloaded to "
                result = named_input.as_download(path_on_compute)
            if path_on_compute:
                status += f"{path_on_compute}."
            else:
                status += "a randomly chosen folder."
            print(status)
            return result
        else:
            return None

    def to_output_dataset(self, workspace: Workspace, dataset_index: int) -> OutputFileDatasetConfig:
        """
        Creates a configuration to write a script output to an AzureML dataset. The name and datastore of this new
        dataset will be taken from the present object.

        :param workspace: The AzureML workspace to read from.
        :param dataset_index: Suffix for using datasets as named inputs, the dataset will be marked OUTPUT_{index}
        :return: An AzureML OutputFileDatasetConfig object, representing the output dataset.
        """
        status = f"Output dataset {self.name} (index {dataset_index}) will be "
        datastore = get_datastore(workspace, self.datastore)
        dataset = OutputFileDatasetConfig(
            name=_output_dataset_key(index=dataset_index), destination=(datastore, self.name + "/")
        )
        # TODO: Can we get tags into here too?
        dataset = dataset.register_on_complete(name=self.name)
        if self.target_folder:
            raise ValueError("Output datasets can't have a target_folder set.")
        use_mounting = True if self.use_mounting is None else self.use_mounting
        if use_mounting:
            status += "uploaded while the job runs."
            result = dataset.as_mount()
        else:
            status += "uploaded when the job completes."
            result = dataset.as_upload(overwrite=self.overwrite_existing)
        logger.info(status)
        return result


StrOrDatasetConfig = Union[str, DatasetConfig]


def _replace_string_datasets(datasets: List[StrOrDatasetConfig], default_datastore_name: str) -> List[DatasetConfig]:
    """
    Processes a list of input or output datasets. All entries in the list that are strings are turned into
    DatasetConfig objects, using the string as the dataset name, and pointing to the default datastore.

    :param datasets: A list of datasets, each given either as a string or a DatasetConfig object.
    :param default_datastore_name: The datastore to use for all datasets that are only specified via their name.
    :return: A list of DatasetConfig objects, in the same order as the input list.
    """
    return [DatasetConfig(name=d, datastore=default_datastore_name) if isinstance(d, str) else d for d in datasets]


def create_dataset_configs(
    all_azure_dataset_ids: List[str],
    all_dataset_mountpoints: Sequence[PathOrString],
    all_local_datasets: List[Optional[Path]],
    datastore: Optional[str] = None,
    use_mounting: bool = False,
) -> List[DatasetConfig]:
    """
    Sets up all the dataset consumption objects for the datasets provided. The returned list will have the same length
    as there are non-empty azure dataset IDs.

    Valid arguments combinations:
    N azure datasets, 0 or N mount points, 0 or N local datasets

    :param all_azure_dataset_ids: The name of all datasets on blob storage that will be used for this run.
    :param all_dataset_mountpoints: When using the datasets in AzureML, these are the per-dataset mount points.
    :param all_local_datasets: The paths for all local versions of the datasets.
    :param datastore: The name of the AzureML datastore that holds the dataset. This can be empty if the AzureML
        workspace has only a single datastore, or if the default datastore should be used.
    :param use_mounting: If True, the dataset will be "mounted", that is, individual files will be read
        or written on-demand over the network. If False, the dataset will be fully downloaded before the job starts,
        respectively fully uploaded at job end for output datasets.
    :return: A list of DatasetConfig objects, in the same order as datasets were provided in all_azure_dataset_ids,
    omitting datasets with an empty name.
    """
    datasets: List[DatasetConfig] = []
    num_local = len(all_local_datasets)
    num_azure = len(all_azure_dataset_ids)
    num_mount = len(all_dataset_mountpoints)
    if num_azure > 0 and (num_local == 0 or num_local == num_azure) and (num_mount == 0 or num_mount == num_azure):
        # Test for valid settings: If we have N azure datasets, the local datasets and mount points need to either
        # have exactly the same length, or 0. In the latter case, empty mount points and no local dataset will be
        # assumed below.
        count = num_azure
    elif num_azure == 0 and num_mount == 0:
        # No datasets in Azure at all: This is possible for runs that for example download their own data from the web.
        # There can be any number of local datasets, but we are not checking that. In TrainingRunner.setup, there is a check
        # that leaves local datasets intact if there are no Azure datasets.
        return []
    else:
        raise ValueError(
            "Invalid dataset setup. You need to specify N entries in azure_datasets and a matching "
            "number of local_datasets and dataset_mountpoints"
        )
    for i in range(count):
        azure_dataset = all_azure_dataset_ids[i] if i < num_azure else ""
        if not azure_dataset:
            continue
        mount_point = all_dataset_mountpoints[i] if i < num_mount else ""
        local_dataset = all_local_datasets[i] if i < num_local else None
        config = DatasetConfig(
            name=azure_dataset,
            target_folder=mount_point,
            local_folder=local_dataset,
            use_mounting=use_mounting,
            datastore=datastore or "",
        )
        datasets.append(config)
    return datasets


def setup_local_datasets(
    dataset_configs: List[DatasetConfig],
    workspace: Optional[Workspace],
) -> Tuple[List[Path], List[Any]]:
    """
    When running outside of AzureML, setup datasets to be used locally.

    For each DatasetConfig, if local_folder is supplied, then this is assumed to be a local dataset, and this is
    used. Otherwise the dataset is mounted or downloaded to either the target folder or a temporary folder and that is
    used.

    If a dataset does not exist, an AzureML SDK v1 dataset will be created, assuming that the dataset is given
    in a folder of the same name (for example, if a dataset is given as "mydataset", then it is created from the files
    in folder "mydataset" in the datastore).

    :param workspace: The AzureML workspace to work with. Can be None if the list of datasets is empty, or if
        the datasets are available local.
    :param dataset_configs: List of DatasetConfig describing the input data assets.
    :return: Pair of: list of paths to the input datasets, list of mountcontexts, one for each mounted dataset.
    """
    mounted_input_datasets: List[Path] = []
    mount_contexts: List[Any] = []

    for data_config in dataset_configs:
        target_path, mount_context = data_config.to_input_dataset_local(workspace)

        mounted_input_datasets.append(target_path)

        if mount_context is not None:
            mount_context.start()
            mount_contexts.append(mount_context)

    return mounted_input_datasets, mount_contexts
