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
