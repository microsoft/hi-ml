# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

For each Pull Request, the affected code parts should be briefly described and added here in the "Upcoming" section.
Once a release is prepared, the "Upcoming" section becomes the release changelog, and a new empty "Upcoming" should be
created.


## Upcoming

### Added
- ([#111](https://github.com/microsoft/hi-ml/pull/111)) Adding changelog. Displaying changelog in sphinx docu. Ensure changelog is updated.
- ([#120](https://github.com/microsoft/hi-ml/pull/120)) Added wrapper function to handle failes with run.upload_folder.

### Changed
- ([#112](https://github.com/microsoft/hi-ml/pull/112)) Update himl_tensorboard to work with files not in 'logs' directory
- ([#106](https://github.com/microsoft/hi-ml/pull/106)) Split into two packages. Most of existing package renamed to hi-ml-azure, remained remains hi-ml.

### Fixed
- ([#117](https://github.com/microsoft/hi-ml/pull/117)) Bug fix: Config.json file was expected to be present, even if workspace was provided explicitly.
- ([#119](https://github.com/microsoft/hi-ml/pull/119)) Bug fix: Code coverage wasn't formatted correctly.

### Removed

### Deprecated

## 0.1.4 (2021-09-15)

- This is the baseline release.
