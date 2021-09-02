# Microsoft Health Intelligence Machine Learning Toolbox

## Overview

This toolbox aims at providing low-level and high-level building blocks for Machine Learning / AI researchers and
practitioners. It helps to simplify and streamline work on deep learning models for healthcare and life sciences,
by providing tested components (data loaders, pre-processing), deep learning models, and cloud integration tools.


## Getting started

* Install from `pypi` via `pip`, by running `pip install hi-ml`


## Building documentation
To build the sphinx documentation, you must have sphinx and related packages installed (see [build_requirements.txt](build_requirements.txt)). Then run:
```
cd docs
make html
```

## Using the AzureML integration layer

Use case: you have a Python script that does something - that could be training a model, or pre-processing some data.
The `hi-ml` package can help easily run that on Azure Machine Learning (AML) services.

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
Doing that at scale can take a long time. We'd like to run that script in AzureML, consume the data from a folder in
blob storage, and write the results back to blob storage.

Pre-requisite: Download the config file from your AzureML workspace, as described 
[here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-environment). Put this file into
the folder where your script lives.

```python
from pathlib import Path
from health.azure.himl import submit_to_azure_if_needed
if __name__ == '__main__':
    current_file = Path(__file__)
    run_info = submit_to_azure_if_needed(entry_script=current_file, 
                                          snapshot_root_directory=current_file.parent,
                                          workspace_config_path=Path("config.json"),
                                          compute_cluster_name="preprocess-ds12",
                                          conda_environment_file=Path("environment.yml"),
                                          input_datasets=["images123"],
                                          # Omit this line if you don't create an output dataset (for example, in
                                          # model training scripts)
                                          output_datasets=["images123_resized"],)
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
on the commandline, like `python myscript.py --azureml`
 
In the added code, you need to supply the following parameters:
* `entry_script`: The script that should be run
* `snapshot_root_directory`: The directory that contains all code that should be packaged and sent to AzureML. All
Python code that the script uses must be copied over.
* `compute_cluster_name`: The name of the AzureML cluster that should run the job. This can be a cluster with CPU or 
GPU machines. See [here for documentation](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-attach-compute-studio#amlcompute)
* `conda_environment_file`: The conda configuration file that describes which packages are necessary for your script
to run.

You also need to supply an input dataset. For data pre-processing scripts, you also supply an output dataset - you
can omit this for ML training scripts.
* You need to provision a data store in your AML workspace, that points to your training data in blob storage. This
is described [here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-connect-data-ui).
* `input_datasets=["images123"]` in the code above means that the script will consume all data in folder `images123`
in blob storage as the input. The folder must exist in blob storage, in the location that you gave when creating the
datastore. Once the script has run, it will also register the data in this folder as an AML dataset. 
* `output_datasets=["images123_resized"]` means that the script will create a temporary folder when running in AML,
and while the job writes data to that folder, upload it to blob storage, in the data store.


To be filled in:
* `default_datastore`
* Complex dataset setup.

## Licensing

[MIT License](LICENSE)

**You are responsible for the performance, the necessary testing, and if needed any regulatory clearance for
 any of the models produced by this toolbox.**

## Contact

If you have any feature requests, or find issues in the code, please create an 
[issue on GitHub](https://github.com/microsoft/hi-ml/issues).

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
