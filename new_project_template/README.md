# Template

This is a folder with a template set of files to create a new project in hi-ml.

To use it:

* Copy the template folder to the desired project folder
* In the new location, rename the folders `health_multimodal` and `test_multimodal` to be the desired Python namespaces
* Search for all occurences of `multimodal` and replace with the desired project name. Don't use a simple query/replace
  strategy but do check the context!

## Developer setup

Make file commands:

* `make pip_local` to install multimodal package in editable mode. This must happen before running tests.
* `make build` to build the package
* `make mypy` to run `mypy`
* `make check` to run `flake8`, `mypy` and `pyright`
* `make clean` to clean up all temporary files and folders
