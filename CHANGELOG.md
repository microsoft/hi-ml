# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file has sections for all previous releases, and the next one.
For each Pull Request, the affected code parts should be briefly described and added in the section for the upcoming
release. In the first PR after a release has been made, a section for the upcoming release should be added, by copying
the section headers (Added/Changed/...) and incrementing the package version.

## 0.1.12

### Added
- ([#159](https://github.com/microsoft/hi-ml/pull/159)) Add profiling for loading png image files as numpy arrays.

### Changed

### Fixed
- ([#161](https://github.com/microsoft/hi-ml/pull/161)) Empty string as target folder for a dataset creates an invalid mounting path for the dataset in AzureML (fixes #160)

### Removed

### Deprecated

## 0.1.11

### Added
- ([#145](https://github.com/microsoft/hi-ml/pull/145)) Add ability to mount datasets when running locally.
- ([#149](https://github.com/microsoft/hi-ml/pull/149)) Add a k-fold cross validation wrapper around HyperDrive
- ([#132](https://github.com/microsoft/hi-ml/pull/132)) Profile methods for loading png image files.

### Changed

### Fixed
- ([#156](https://github.com/microsoft/hi-ml/pull/156) AzureML Runs should use registered environment after retrieval)

### Removed

### Deprecated

## 0.1.10

### Added
- ([#142](https://github.com/microsoft/hi-ml/pull/142)) Adding AzureML progress bar and diagnostics for batch loading
- ([#138](https://github.com/microsoft/hi-ml/pull/138)) Guidelines and profiling for whole slide images.

### Changed
- ([#129])https://github.com/microsoft/hi-ml/pull/129)) Refactor command line tools' arguments. Refactor health_azure.utils' various get_run functions. Replace
argparsing with parametrized classes.

### Fixed

### Removed

### Deprecated


## 0.1.9 (2021-10-20)

### Added
- ([#133](https://github.com/microsoft/hi-ml/pull/133)) PyTorch Lightning logger for AzureML. Helper functions for consistent logging
- ([#136](https://github.com/microsoft/hi-ml/pull/136)) Documentation for using low priority nodes

### Changed
- ([#133](https://github.com/microsoft/hi-ml/pull/133)) Made _**large breaking changes**_ to module names,
from `health.azure` to `health_azure`.
- ([#144])(https://github.com/microsoft/hi-ml/pull/141) Update changelog for release and increase scope of test_register_environment to ensure that by default environments are registered with a version number

### Fixed
- ([#134](https://github.com/microsoft/hi-ml/pull/134)) Fixed repo references and added pyright to enforce global checking
- ([#139](https://github.com/microsoft/hi-ml/pull/139) Fix register_environment, which was ignoring existing environemnts
previously. Also ensure that the environment is given version 1 by default instead of "autosave")


## 0.1.8 (2021-10-06)

### Added
- ([#123](https://github.com/microsoft/hi-ml/pull/123)) Add helper function to download checkpoint files
- ([#128](https://github.com/microsoft/hi-ml/pull/128)) When downloading files in a distributed PyTorch job, a barrier is used to synchronize the processes.

### Changed
- ([#127](https://github.com/microsoft/hi-ml/pull/127)) The field `is_running_in_azure` of `AzureRunInfo` has been renamed to `is_running_in_azure_ml`

### Fixed
- ([#127](https://github.com/microsoft/hi-ml/pull/127)) Fixing bug #126: get_workspace was assuming it runs in AzureML, when it was running on a plain Azure build agent.


## 0.1.7 (2021-10-04)

### Added
- ([#111](https://github.com/microsoft/hi-ml/pull/111)) Adding changelog. Displaying changelog in sphinx docu. Ensure changelog is updated.

### Changed
- ([#112](https://github.com/microsoft/hi-ml/pull/112)) Update himl_tensorboard to work with files not in 'logs' directory
- ([#106](https://github.com/microsoft/hi-ml/pull/106)) Split into two packages. Most of existing package renamed to hi-ml-azure, remained remains hi-ml.
- ([#113](https://github.com/microsoft/hi-ml/pull/113)) Add helper function to download files from AML Run, tidied up some command line args, and moved some functions from himl.py to azure_util.py
- ([#122](https://github.com/microsoft/hi-ml/pull/122)) Add helper functions to upload to and download from AML Datastores

### Fixed
- ([#117](https://github.com/microsoft/hi-ml/pull/117)) Bug fix: Config.json file was expected to be present, even if workspace was provided explicitly.
- ([#119](https://github.com/microsoft/hi-ml/pull/119)) Bug fix: Code coverage wasn't formatted correctly.


## 0.1.4 (2021-09-15)

- This is the baseline release.
