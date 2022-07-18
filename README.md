# Microsoft Health Intelligence Machine Learning Toolbox

[![Codecov coverage](https://codecov.io/gh/microsoft/hi-ml/branch/main/graph/badge.svg?token=kMr2pSIJ2U)](https://codecov.io/gh/microsoft/hi-ml)

## Overview

This toolbox aims at providing low-level and high-level building blocks for Machine Learning / AI researchers and
practitioners. It helps to simplify and streamline work on deep learning models for healthcare and life sciences,
by providing tested components (data loaders, pre-processing), deep learning models, and cloud integration tools.

This repository consists of two Python packages, as well as project-specific codebases:

* PyPi package [hi-ml-azure](https://pypi.org/project/hi-ml-azure/) - providing helper functions for running in AzureML.
* PyPi package [hi-ml](https://pypi.org/project/hi-ml/) - providing ML components.
* hi-ml-cpath: Models and workflows for working with histopathology images

## Getting started

For the full toolbox (this will also install `hi-ml-azure`):

* Install from `pypi` via `pip`, by running `pip install hi-ml`

For just the AzureML helper functions:

* Install from `pypi` via `pip`, by running `pip install hi-ml-azure`

For the histopathology workflows, please follow the instructions [here](hi-ml-cpath/README.md).

If you would like to contribute to the code, please check the [developer guide](docs/source/developers.md).

## Documentation

The detailed package documentation, with examples and API reference, is on
[readthedocs](https://hi-ml.readthedocs.io/en/latest/).

## Quick start: Using the Azure layer

Use case: you have a Python script that does something - that could be training a model, or pre-processing some data.
The `hi-ml-azure` package can help easily run that on Azure Machine Learning (AML) services.

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
blob storage, and write the results back to blob storage**.

With the `hi-ml-azure` package, you can turn that script into one that runs on the cloud by adding one function call:

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

That's it!

For details, please refer to the [onboarding page](docs/source/first_steps.md).

For more examples, please see [examples.md](docs/source/examples.md).

## Issues

If you've found a bug in the code, please check the [issues](https://github.com/microsoft/hi-ml/issues) page.
If no existing issue exists, please open a new one. Be sure to include

* A descriptive title
* Expected behaviour (including a code sample if possible)
* Actual behavior

## Contributing

We welcome all contributions that help us achieve our aim of speeding up ML/AI research in health and life sciences.
Examples of contributions are

* Data loaders for specific health & life sciences data
* Network architectures and components for deep learning models
* Tools to analyze and/or visualize data
* ...

Please check the [detailed page about contributions](.github/CONTRIBUTING.md).

## Licensing

[MIT License](LICENSE)

**You are responsible for the performance, the necessary testing, and if needed any regulatory clearance for
 any of the models produced by this toolbox.**

## Contact

If you have any feature requests, or find issues in the code, please create an
[issue on GitHub](https://github.com/microsoft/hi-ml/issues).

## Contribution Licensing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit <https://cla.opensource.microsoft.com>.

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
