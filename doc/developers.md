# Notes for developers

## Creating a Conda environment

To create a separate Conda environment with all packages that `hi-ml` requires for running and testing,
use the provided `environment.yml` file. Create a Conda environment called `himl` from that via
```shell script
conda env create --file environment.yml
conda activate himl
```

## Using specific versions `hi-ml` in your Python environments 

If you'd like to test specific changes to the `hi-ml` package in your code, you can use two different routes:

* You can clone the `hi-ml` repository on your machine, and use `hi-ml` in your Python environment via a local package
install:
```shell script
pip install -e <your_git_folder>/hi-ml
```
* You can consume an early version of the package from `test.pypi.org` via `pip`:
```shell script
pip install --extra-index-url https://test.pypi.org/simple/ hi-ml==0.1.0.post165
```
* If you are using Conda, you can add an additional parameter for `pip` into the Conda `environment.yml` file like this:
```
name: foo
dependencies:
  - pip=20.1.1
  - python=3.7.3
  - pip:
      - --extra-index-url https://test.pypi.org/simple/
      - hi-ml==0.1.0.post165
```

## Common things to do

The repository contains a makefile with definitions for common operations. 
* `make check`: Run `flake8` and `mypy` on the repository.
* `make test`: Run `flake8` and `mypy` on the repository, then all tests via `pytest`
* `make pip`: Install all packages for running and testing in the current interpreter.
* `make conda`: Update the hi-ml Conda environment and activate it


## Setting up your AzureML workspace

* In the browser, navigate to the AzureML workspace that you want to use for running your tests. 
* In the top right section, there will be a dropdown menu showing the name of your AzureML workspace. Expand that.
* In the panel, there is a link "Download config file". Click that.
* This will download a file `config.json`. Move that file to the root folder of your `hi-ml` repository. The file name
is already present in `.gitignore`, and will hence not be checked in.

## Creating and Deleting Docker Environments in AzureML

* Passing a `docker_base_image` into `submit_to_azure_if_needed` causes a new image to be built and registered in your
workspace (see [docs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-environments) for more
information).
* To remove an environment use the [az ml environment delete](https://docs.microsoft.com/en-us/cli/azure/ml/environment?view=azure-cli-latest#az_ml_environment_delete)
function in the AzureML CLI (note that all the parameters need to be set, none are optional).

## Testing

For all of the tests to work locally you will need to cache your AzureML credentials. One simple way to do this is to
set run the example in `src/health/azure/examples` (`python elevate_this.py --message='Hello World' --azureml`) after
editing `elevate_this.py` to reference your compute cluster.

### WSL

You may need to install the ASP.NET Core Runtime in WSL. Follow the instructions 
[here](https://docs.microsoft.com/en-us/dotnet/core/install/linux-ubuntu#2104-), replacing the latest version number 
when you get to `sudo apt-get install -y aspnetcore-runtime-x.x` with `2.1`.