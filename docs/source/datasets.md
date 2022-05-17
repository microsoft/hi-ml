# Datasets

## Key concepts
We'll first outline a few concepts that are helpful for understanding datasets.

### Blob Storage
Firstly, there is [Azure Blob Storage](https://docs.microsoft.com/en-us/azure/storage/blobs/storage-blobs-introduction).
Each blob storage account has multiple containers - you can think of containers as big disks that store files.
The `hi-ml` package assumes that your datasets live in one of those containers, and each top level folder corresponds
to one dataset.


### AzureML Data Stores
Secondly, there are data stores. This is a concept coming from Azure Machine Learning, described
[here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-access-data). Data stores provide access to
one blob storage account. They exist so that the credentials to access blob storage do not have to be passed around
in the code - rather, the credentials are stored in the data store once and for all.

You can view all data stores in your AzureML workspace by clicking on one of the bottom icons in the left-hand
navigation bar of the AzureML studio.

One of these data stores is designated as the default data store.

### AzureML Datasets

Thirdly, there are datasets. Again, this is a concept coming from Azure Machine Learning. A dataset is defined by
* A data store
* A set of files accessed through that data store

You can view all datasets in your AzureML workspace by clicking on one of the icons in the left-hand
navigation bar of the AzureML studio.

### Preparing data
To simplify usage, the `hi-ml` package creates AzureML datasets for you. All you need to do is to
* Create a blob storage account for your data, and within it, a container for your data.
* Create a data store that points to that storage account, and store the credentials for the blob storage account in it

From that point on, you can drop a folder of files in the container that holds your data. Within the `hi-ml` package,
just reference the name of the folder, and the package will create a dataset for you, if it does not yet exist.

## Using the datasets

The simplest way of specifying that your script uses a folder of data from blob storage is as follows: Add the
`input_datasets` argument to your call of `submit_to_azure_if_needed` like this:
```python
from health_azure import submit_to_azure_if_needed
run_info = submit_to_azure_if_needed(...,
                                     input_datasets=["my_folder"],
                                     default_datastore="my_datastore")
input_folder = run_info.input_datasets[0]
```
What will happen under the hood?
* The toolbox will check if there is already an AzureML dataset called "my_folder". If so, it will use that. If there
is no dataset of that name, it will create one from all the files in blob storage in folder "my_folder". The dataset
will be created using the data store provided, "my_datastore".
* Once the script runs in AzureML, it will download the dataset "my_folder" to a temporary folder.
* You can access this temporary location by `run_info.input_datasets[0]`, and read the files from it.

More complicated setups are described below.

### Input and output datasets

Any run in AzureML can consume a number of input datasets. In addition, an AzureML run can also produce an output
dataset (or even more than one).

Output datasets are helpful if you would like to run, for example, a script that transforms one dataset into another.

You can use that via the `output_datasets` argument:
```python
from health_azure import submit_to_azure_if_needed
run_info = submit_to_azure_if_needed(...,
                                     input_datasets=["my_folder"],
                                     output_datasets=["new_dataset"],
                                     default_datastore="my_datastore")
input_folder = run_info.input_datasets[0]
output_folder = run_info.output_datasets[0]
```
Your script can now read files from `input_folder`, transform them, and write them to `output_folder`. The latter
will be a folder on the temp file system of the machine. At the end of the script, the contents of that temp folder
will be uploaded to blob storage, and registered as a dataset.

### Mounting and downloading
An input dataset can be downloaded before the start of the actual script run, or it can be mounted. When mounted,
the files are accessed via the network once needed - this is very helpful for large datasets where downloads would
create a long waiting time before the job start.

Similarly, an output dataset can be uploaded at the end of the script, or it can be mounted. Mounting here means that
all files will be written to blob storage already while the script runs (rather than at the end).

Note: If you are using mounted output datasets, you should NOT rename files in the output folder.

Mounting and downloading can be triggered by passing in `DatasetConfig` objects for the `input_datasets` argument,
like this:

```python
from health_azure import DatasetConfig, submit_to_azure_if_needed
input_dataset = DatasetConfig(name="my_folder", datastore="my_datastore", use_mounting=True)
output_dataset = DatasetConfig(name="new_dataset", datastore="my_datastore", use_mounting=True)
run_info = submit_to_azure_if_needed(...,
                                     input_datasets=[input_dataset],
                                     output_datasets=[output_dataset])
input_folder = run_info.input_datasets[0]
output_folder = run_info.output_datasets[0]
```

### Local execution
For debugging, it is essential to have the ability to run a script on a local machine, outside of AzureML.
Clearly, your script needs to be able to access data in those runs too.

There are two ways of achieving that: Firstly, you can specify an equivalent local folder in the
`DatasetConfig` objects:
```python
from pathlib import Path
from health_azure import DatasetConfig, submit_to_azure_if_needed
input_dataset = DatasetConfig(name="my_folder",
                              datastore="my_datastore",
                              local_folder=Path("/datasets/my_folder_local"))
run_info = submit_to_azure_if_needed(...,
                                     input_datasets=[input_dataset])
input_folder = run_info.input_datasets[0]
```

Secondly, if `local_folder` is not specified, then the dataset will either be downloaded or mounted to a temporary folder locally, depending on the `use_mounting` flag. The path to it will be available in `run_info` as above.
```python
input_folder = run_info.input_datasets[0]
```

Note that mounting the dataset locally is only supported on Linux because it requires the use of the native package [libfuse](https://github.com/libfuse/libfuse/), which must first be installed. Also, if running in a Docker container, it must be started with additional arguments. For more details see here: [azureml.data.filedataset.mount](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.data.filedataset?view=azure-ml-py#mount-mount-point-none----kwargs-).

### Making a dataset available at a fixed folder location

Occasionally, scripts expect the input dataset at a fixed location, for example, data is always read from `/tmp/mnist`.
AzureML has the capability to download/mount a dataset to such a fixed location. With the `hi-ml` package, you can
trigger that behaviour via an additional option in the `DatasetConfig` objects:
```python
from health_azure import DatasetConfig, submit_to_azure_if_needed
input_dataset = DatasetConfig(name="my_folder",
                              datastore="my_datastore",
                              use_mounting=True,
                              target_folder="/tmp/mnist")
run_info = submit_to_azure_if_needed(...,
                                     input_datasets=[input_dataset])
# Input_folder will now be "/tmp/mnist"
input_folder = run_info.input_datasets[0]
```

This is also true when running locally - if `local_folder` is not specified and an AzureML workspace can be found, then the dataset will be downloaded or mounted to the `target_folder`.

### Dataset versions
AzureML datasets can have versions, starting at 1. You can view the different versions of a dataset in the AzureML
workspace. In the `hi-ml` toolbox, you would always use the latest version of a dataset unless specified otherwise.
If you do need a specific version, use the `version` argument in the `DatasetConfig` objects:
```python
from health_azure import DatasetConfig, submit_to_azure_if_needed
input_dataset = DatasetConfig(name="my_folder",
                              datastore="my_datastore",
                              version=7)
run_info = submit_to_azure_if_needed(...,
                                     input_datasets=[input_dataset])
input_folder = run_info.input_datasets[0]
```
