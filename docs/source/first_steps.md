# First steps: How to run your Python code in the cloud

The simplest use case for the `hi-ml` toolbox is taking a script that you developed, and run it inside of
Azure Machine Learning (AML) services. This can be helpful because the cloud gives you access to massive GPU
resource, you can consume vast datasets, and access multiple machines at the same time for distributed training.

## Setting up AzureML

To run your code in the cloud, you need to have an AzureML workspace in your Azure subscription.
Please follow the [instructions here](azure_setup.md) to create an AzureML workspace if you don't have one yet.

Download the config file from your AzureML workspace, as described
[here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-environment). **Put this file (it
should be called `config.json`) into the folder where your script lives**, or one of its parent folders. You can use
parent folders up to the last parent that is still included in the `PYTHONPATH` environment variable: `hi-ml` will
try to be smart and search through all folders that it thinks belong to your current project.

## Using the AzureML integration layer

Consider a simple use case, where you have a Python script that does something - this could be training a model,
or pre-processing some data. The `hi-ml` package can help easily run that on Azure Machine Learning (AML) services.

Here is an example script that reads images from a folder, resizes and saves them to an output folder:

```python
from pathlib import Path
if __name__ == '__main__':
    input_folder = Path("/tmp/my_dataset")
    output_folder = Path("/tmp/my_output")
    for file in input_folder.glob("*.jpg"):
        contents = read_image(file)
        resized = contents.resize(0.5)
        write_image(output_folder / file.name)
```

Doing that at scale can take a long time. **We'd like to run that script in AzureML, consume the data from a folder in
blob storage, and write the results back to blob storage**, so that we can later use it as an input for model training.

You can achieve that by adding a call to `submit_to_azure_if_needed` from the `hi-ml` package:

```python
from pathlib import Path
from health_azure import submit_to_azure_if_needed
if __name__ == '__main__':
    current_file = Path(__file__)
    run_info = submit_to_azure_if_needed(compute_cluster_name="preprocess-ds12",
                                         input_datasets=["images123"],
                                         # Omit this line if you don't create an output dataset (for example, in
                                         # model training scripts)
                                         output_datasets=["images123_resized"],
                                         default_datastore="my_datastore")
    # When running in AzureML, run_info.input_datasets and run_info.output_datasets will be populated,
    # and point to the data coming from blob storage. For runs outside AML, the paths will be None.
    # Replace the None with a meaningful path, so that we can still run the script easily outside AML.
    input_dataset = run_info.input_datasets[0] or Path("/tmp/my_dataset")
    output_dataset = run_info.output_datasets[0] or Path("/tmp/my_output")
    files_processed = []
    for file in input_dataset.glob("*.jpg"):
        contents = read_image(file)
        resized = contents.resize(0.5)
        write_image(output_dataset / file.name)
        files_processed.append(file.name)
    # Any other files that you would not consider an "output dataset", like metrics, etc, should be written to
    # a folder "./outputs". Any files written into that folder will later be visible in the AzureML UI.
    # run_info.output_folder already points to the correct folder.
    stats_file = run_info.output_folder / "processed_files.txt"
    stats_file.write_text("\n".join(files_processed))
```

Once these changes are in place, you can submit the script to AzureML by supplying an additional `--azureml` flag
on the commandline, like `python myscript.py --azureml`.

Note that you do not need to modify the argument parser of your script to recognize the `--azureml` flag.

## Essential arguments to `submit_to_azure_if_needed`

When calling `submit_to_azure_if_needed`, you can to supply the following parameters:

* `compute_cluster_name` (**Mandatory**): The name of the AzureML cluster that should run the job. This can be a
cluster with CPU or GPU machines. See
[here for documentation](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-attach-compute-studio#amlcompute)
* `entry_script`: The script that should be run. If omitted, the `hi-ml` package will assume that you would like
to submit the script that is presently running, given in `sys.argv[0]`.
* `snapshot_root_directory`: The directory that contains all code that should be packaged and sent to AzureML. All
Python code that the script uses must be copied over. This defaults to the current working directory, but can be
one of its parents. If you would like to explicitly skip some folders inside the `snapshot_root_directory`, then use
  `ignored_folders` to specify those.
* `conda_environment_file`: The conda configuration file that describes which packages are necessary for your script
to run. If omitted, the `hi-ml` package searches for a file called `environment.yml` in the current folder or its
parents.

You can also supply an input dataset. For data pre-processing scripts, you can add an output dataset
(omit this for ML training scripts).

* To use datasets, you need to provision a data store in your AML workspace, that points to your training data in
  blob storage. This is described
  [here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-connect-data-ui).
* `input_datasets=["images123"]` in the code above means that the script will consume all data in folder `images123`
in blob storage as the input. The folder must exist in blob storage, in the location that you gave when creating the
datastore. Once the script has run, it will also register the data in this folder as an AML dataset.
* `output_datasets=["images123_resized"]` means that the script will create a temporary folder when running in AML,
and while the job writes data to that folder, upload it to blob storage, in the data store.

For more examples, please see [examples.md](examples.md). For more details about datasets, see [here](datasets.md).

## Additional arguments you should know about

`submit_to_azure_if_needed` has a large number of arguments, please check the
[API documentation](api/health.azure.submit_to_azure_if_needed.rst) for an exhaustive list.
The particularly helpful ones are listed below.

* `experiment_name`: All runs in AzureML are grouped in "experiments". By default, the experiment name is determined
  by the name of the script you submit, but you can specify a name explicitly with this argument.
* `environment_variables`: A dictionary with the contents of all environment variables that should be set inside the
  AzureML run, before the script is started.
* `docker_base_image`: This specifies the name of the Docker base image to use for creating the
  Python environment for your script. The amount of memory to allocate for Docker is given by `docker_shm_size`.
* `num_nodes`: The number of nodes on which your script should run. This is essential for distributed training.
* `tags`: A dictionary mapping from string to string, with additional tags that will be stored on the AzureML run.
  This is helpful to add metadata about the run for later use.

## Conda environments, Alternate pips, Private wheels

The function `submit_to_azure_if_needed` tries to locate a Conda environment file in the current folder,
or in the Python path, with the name `environment.yml`. The actual Conda environment file to use can be specified
directly with:

```python
    run_info = submit_to_azure_if_needed(
        ...
        conda_environment_file=conda_environment_file,
```

where `conda_environment_file` is a `pathlib.Path` or a string identifying the Conda environment file to use.

The basic use of Conda assumes that packages listed are published
[Conda packages](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/packages.html) or published
Python packages on [PyPI](https://pypi.org/). However, during development, the Python package may be on
[Test.PyPI](https://test.pypi.org/), or in some other location, in which case the alternative package location can
be specified directly with:

```python
    run_info = submit_to_azure_if_needed(
        ...
        pip_extra_index_url="https://test.pypi.org/simple/",
```

Finally, it is possible to use a private wheel, if the package is only available locally with:

```python
    run_info = submit_to_azure_if_needed(
        ...
        private_pip_wheel_path=private_pip_wheel_path,
```

where `private_pip_wheel_path` is a `pathlib.Path` or a string identifying the wheel package to use. In this case,
this wheel will be copied to the AzureML environment as a private wheel.
