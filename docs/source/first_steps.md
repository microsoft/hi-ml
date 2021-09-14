# First steps: How to get started with hi-ml

## Setting up AzureML
You need to have an AzureML workspace in your Azure subscription.
Download the config file from your AzureML workspace, as described 
[here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-environment). **Put this file (it
should be called `config.json` into the folder where your script lives**, or one of its parent folders. You can use
parent folders up to the last parent that is still included in the `PYTHONPATH` environment variable - `hi-ml` will
try to be smart and search through all code that it thinks belongs to your current project.

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
from health.azure import submit_to_azure_if_needed
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

## Arguments to `submit_to_azure_if_needed`
When calling `submit_to_azure_if_needed`, you can to supply the following parameters:
* `compute_cluster_name` (**Mandatory**): The name of the AzureML cluster that should run the job. This can be a 
cluster with CPU or GPU machines. See
[here for documentation](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-attach-compute-studio#amlcompute)
* `entry_script`: The script that should be run. If omitted, the `hi-ml` package will assume that you would like
to submit the script that is presently running, given in `sys.argv[0]`.
* `snapshot_root_directory`: The directory that contains all code that should be packaged and sent to AzureML. All
Python code that the script uses must be copied over. This defaults to the current working directory, but can be
one of its parents.
* `conda_environment_file`: The conda configuration file that describes which packages are necessary for your script
to run. If omitted, the `hi-ml` package searches for a file called `environment.yml` in the current folder or its
parents.

You also need to supply an input dataset. For data pre-processing scripts, you can also supply an output dataset
(omit this for ML training scripts).
* You need to provision a data store in your AML workspace, that points to your training data in blob storage. This
is described [here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-connect-data-ui).
* `input_datasets=["images123"]` in the code above means that the script will consume all data in folder `images123`
in blob storage as the input. The folder must exist in blob storage, in the location that you gave when creating the
datastore. Once the script has run, it will also register the data in this folder as an AML dataset. 
* `output_datasets=["images123_resized"]` means that the script will create a temporary folder when running in AML,
and while the job writes data to that folder, upload it to blob storage, in the data store.

For more examples, please see [examples.md](examples.md).
