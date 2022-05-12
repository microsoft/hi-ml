# Template

This is a folder with a template set of files to create a new project in hi-ml.

To use it:

* Copy the template folder to the desired project folder
* In the new location, rename the folders `health_newproject` and `test_newproject` to be the desired Python namespaces
* Search for all occurences of `newproject` and replace with the desired project name. Don't use a simple query/replace
  strategy but do check the context!
* Update the `requirements_*.txt` files to match your needs.
* Update the Conda environment `environment.yml` to match your needs. Ensure that all packages that are in the
  requirements are also included in the environment.
* Move the workflow definition `new-project-pr.yml` to `.github/workflows` and give it a name that matches the project.
* If you do not want your code to be published as a PyPi package, remove the steps `build-python` and `publish-pypi-pkg`
  from the workflow.
* If you want your package to be published, add a PyPi access token as a secret to the Github project. Add the name of
  the secret in the `publish-pypi-pkg` step of the pipeline.

## Developer setup

Make file commands:

* `make pip_local` to install the package in editable mode. This must happen before running tests.
* `make build` to build the package
* `make mypy` to run `mypy`
* `make check` to run `flake8`, `mypy` and `pyright`
* `make clean` to clean up all temporary files and folders
