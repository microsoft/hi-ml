# Changelog

Early versions of this toolbox used a manually created changelog. As of March 2022, we have switched to using Github's auto-generated changelog.
If you would like to view the changelog for a particular release, you can do so on the [Releases page](https://github.com/microsoft/hi-ml/releases):
Each release contains a link for "Full Changelog"


## Changelog for Versions before March 2022

## 0.1.14

### Added

- ([#227](https://github.com/microsoft/hi-ml/pull/227)) Add TransformerPooling.
- ([#179](https://github.com/microsoft/hi-ml/pull/179)) Add GaussianBlur and RotationByMultiplesOf90 augmentations. Added torchvision and opencv to
the environment file since it is necessary for the augmentations.
- ([#193](https://github.com/microsoft/hi-ml/pull/193)) Add transformation adaptor to hi-ml-histopathology.
- ([#178](https://github.com/microsoft/hi-ml/pull/178)) Add runner script for running ML experiments.
- ([#181](https://github.com/microsoft/hi-ml/pull/181)) Add computational pathology tools in hi-ml-histopathology folder.
- ([#187](https://github.com/microsoft/hi-ml/pull/187)) Add mean pooling layer for MIL.
- ([#186](https://github.com/microsoft/hi-ml/pull/186)) Add inference to hi-ml runner.
- ([#198](https://github.com/microsoft/hi-ml/pull/198)) Add cross-validation to hi-ml runner.
- ([#198](https://github.com/microsoft/hi-ml/pull/198)) Improved editor setup for VSCode.

### Changed

- ([#227](https://github.com/microsoft/hi-ml/pull/227)) Pooling constructor is outside of DeepMIL and inside of BaseMIL now.
- ([#198](https://github.com/microsoft/hi-ml/pull/198)) Model config loader is now more flexible, can accept fully qualified class name or just top-level module name and class (like histopathology.DeepSMILECrck)
- ([#198](https://github.com/microsoft/hi-ml/pull/198)) Runner raises an error when Conda environment file contains a pip include (-r) statement

- ([#196](https://github.com/microsoft/hi-ml/pull/196)) Show current workspace name in error message.

### Fixed
- ([#267]https://github.com/microsoft/hi-ml/pull/267)) Correct PYTHONPATH for Windows in VS Code settings
- ([#266]https://github.com/microsoft/hi-ml/pull/266)) Pin jinja2 package to avoid 'No attribute Markup' bug in version 3.1.0
- ([#246](https://github.com/microsoft/hi-ml/pull/246)) Added tolerance to `test_attentionlayers.py`.
- ([#198](https://github.com/microsoft/hi-ml/pull/198)) Dependencies for histopathology folder are no longer specified in `test_requirements.txt`, but correctly in the histopathology Conda environment.
- ([#188](https://github.com/microsoft/hi-ml/pull/188)) Updated DeepSMILES models. Now they are uptodate with innereye-dl.
- ([#179](https://github.com/microsoft/hi-ml/pull/179)) HEDJitter was jittering the D channel as well. StainNormalization was relying on skimage.
- ([#195](https://github.com/microsoft/hi-ml/pull/195)) Fix DeepMIL metrics bug whereby hard labels were used instead of probabilities.


### Removed

### Deprecated

## 0.1.13

### Added

- ([#170](https://github.com/microsoft/hi-ml/pull/170)) Add utils including bag sampling, bounding boxes, HEDJitter, StainNormalisation and add attention layers

### Changed

- ([#173](https://github.com/microsoft/hi-ml/pull/173)) Improve report tool: allow lists of tables, option for zipping report folder, option for base64 encoding images

### Fixed

- ([#169](https://github.com/microsoft/hi-ml/pull/169)) Fix a test that was failing occasionally

### Removed

### Deprecated

## 0.1.12

### Added

- ([#159](https://github.com/microsoft/hi-ml/pull/159)) Add profiling for loading png image files as numpy arrays.
- ([#152](https://github.com/microsoft/hi-ml/pull/152)) Add a custom HTML reporting tool
- ([#167](https://github.com/microsoft/hi-ml/pull/167)) Ability to log to an AzureML run when outside of AzureML

### Changed

- ([164](https://github.com/microsoft/hi-ml/pull/164)) Look in more locations for std out from AzureML run.
- ([#167](https://github.com/microsoft/hi-ml/pull/167)) The AzureMLLogger has one mandatory argument now, that controls
  whether it should log to AzureML also when running on a VM.

### Fixed

- ([#161](https://github.com/microsoft/hi-ml/pull/161)) Empty string as target folder for a dataset creates an invalid mounting path for the dataset in AzureML (fixes #160)
- ([#167](https://github.com/microsoft/hi-ml/pull/167)) Fix bugs in logging hyperparameters: logging as name/value
  table, rather than one column per hyperparameter. Use string logging for all hyperparameters
- ([#174](https://github.com/microsoft/hi-ml/pull/174)) Fix bugs in returned local_checkpoint_path when downloading checkpoints from AML run

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

- ([#129])<https://github.com/microsoft/hi-ml/pull/129>)) Refactor command line tools' arguments. Refactor health_azure.utils' various get_run functions. Replace
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
- ([#141])(https://github.com/microsoft/hi-ml/pull/141)) Update changelog for release and increase scope of test_register_environment to ensure that by default environments are registered with a version number

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
