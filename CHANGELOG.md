# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

For each Pull Request, the affected code parts should be briefly described and added here in the "Upcoming" section.
Once a release is prepared, the "Upcoming" section becomes the release changelog, and a new empty "Upcoming" should be
created.


## Upcoming

### Added

### Changed

### Fixed

### Removed

### Deprecated

## 0.1.7 (2021-10-04)

### Added
- ([#111](https://github.com/microsoft/hi-ml/pull/111)) Adding changelog. Displaying changelog in sphinx docu. Ensure changelog is updated.

### Changed
- ([#112](https://github.com/microsoft/hi-ml/pull/112)) Update himl_tensorboard to work with files not in 'logs' directory
- ([#106](https://github.com/microsoft/hi-ml/pull/106)) Split into two packages. Most of existing package renamed to hi-ml-azure, remained remains hi-ml.
- ([#113](https://github.com/microsoft/hi-ml/pull/113)) Add helper function to download files from AML Run, tidied up some command line args, and moved some functions from himl.py to azure_util.py

### Fixed
- ([#117](https://github.com/microsoft/hi-ml/pull/117)) Bug fix: Config.json file was expected to be present, even if workspace was provided explicitly.
- ([#119](https://github.com/microsoft/hi-ml/pull/119)) Bug fix: Code coverage wasn't formatted correctly.


## 0.1.4 (2021-09-15)

- This is the baseline release.
