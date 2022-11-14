# Running ML experiments with hi-ml

The hi-ml toolbox is capable of training any PyTorch Lighting (PL) model inside of AzureML, making
use of these features:

- Training on a local GPU machine or inside of AzureML without code changes
- Working with different models in the same codebase, and selecting one by name
- Distributed training in AzureML
- Logging via AzureML's native capabilities

This can be used by invoking the hi-ml runner and providing the name of the container class, like this:
`himl-runner --model=MyContainer`.

There is a fully working example [HelloContainer](https://github.com/microsoft/hi-ml/blob/main/hi-ml/src/health_ml/configs/hello_world.py), that
implements a simple 1-dimensional regression model from data stored in a CSV file. You can run that
from the command line by `himl-runner --model=HelloWorld`.

## Specifying the model to run

The `--model` argument specifies the name of a class that should be used for model training. The class needs to
be a subclass of `LightningContainer`, see below. There are different ways of telling the runner where to find
that class:

- If just providing a single class name, like `--model=HelloWorld`, the class is expected somewhere in the
`health_ml.configs` namespace. It can be in any module/folder inside of that namespace.
- If the class is outside of the `health_ml.configs` (as would be normal if using the `himl-runner` from a package),
you need to provide some "hints" where to start searching. It is enough to provide the start of the namespace string:
for example, `--model health_cpath.PandaImageNetMIL` is effectively telling the runner to search for the
`PandaImageNetMIL` class _anywhere_ in the `health_cpath` namespace. You can think of this as
`health_cpath.*.PandaImageNetMIL`

## Running ML experiments in Azure ML

To train in AzureML, use the flag `--cluster` to specify the name of the cluster
in your Workspace that you want to submit the job to. So the whole command would look like:

```bash
himl-runner --model=HelloWorld --cluster=my_cluster_name
```

You can also specify `--num_nodes` if you wish to distribute the model training.

When starting the runner, you need to do that from a directory that contains all the code that your experiment needs:
The current working directory will be used as the root of all data that will be copied to AzureML to run your experiment.
(the only exception to this rule is if you start the runner from within an enlistment of the HI-ML GitHub repository).

AzureML needs to know which Python/Conda environment it should use. For that, the runner needs a file `environment.yml`
that contains a Conda environment definition. This file needs to be present either in the current working directory or
one of its parents. To specify a Conda environment that is located elsewhere, you can use

```bash
himl-runner --model=HelloWorld --cluster=my_cluster_name --conda_env=/my/folder/to/special_environment.yml
```

## Setup - creating your model config file

In order to use these capabilities, you need to implement a class deriving from
 `health_ml.lightning_container.LightningContainer`. This class encapsulates everything that is needed for training
 with PyTorch Lightning:

 For example:

 ```python
class MyContainer(LightningContainer):
    def __init__(self):
        super().__init__()
        self.azure_datasets = ["folder_name_in_azure_blob_storage"]
        self.local_datasets = [Path("/some/local/path")]
        self.max_epochs = 42

    def create_model(self) -> LightningModule:
        return MyLightningModel()

    def get_data_module(self) -> LightningDataModule:
        return MyDataModule(root_path=self.local_dataset)
```

The `create_model` method needs to return a subclass of PyTorch Lightning's [LightningModule](
 https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html?highlight=lightningmodule
), that has
all the usual PyTorch Lightning methods required for training, like the `training_step` and `forward` methods. E.g:

```python
class MyLightningModel(LightningModule):
    def __init__(self):
        self.layer = ...
    def training_step(self, *args, **kwargs):
        ...
    def forward(self, *args, **kwargs):
        ...
    def configure_optimizers(self):
        ...
    def test_step(self, *args, **kwargs):
        ...
```

The `get_data_module` method of the container needs to return a DataModule (inheriting from a [PyTorch Lightning DataModule](
https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html)) which contains all of the logic for
downloading, preparing and splitting your dataset, as well as methods for wrapping the train, val and test datasets
respectively with [DataLoaders](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader). E.g:

```python
class MyDataModule(LightningDataModule):
    def __init__(self, root_path: Path):
        # All data should be read from the folder given in self.root_path
        self.root_path = root_path
    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        # The data should be read off self.root_path
        train_dataset = ...
        return DataLoader(train_dataset, batch_size=5, num_workers=5)
    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        # The data should be read off self.root_path
        val_dataset = ...
        return DataLoader(val_dataset, batch_size=5, num_workers=5)
    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        # The data should be read off self.root_path
        test_dataset = ...
        return DataLoader(test_dataset, batch_size=5, num_workers=5)
```

So, the **full file** would look like:

```python
from pathlib import Path
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, LightningDataModule
from health_ml.lightning_container import LightningContainer

class MyLightningModel(LightningModule):
    def __init__(self):
        self.layer = ...
    def training_step(self, *args, **kwargs):
        ...
    def forward(self, *args, **kwargs):
        ...
    def configure_optimizers(self):
        ...
    def test_step(self, *args, **kwargs):
        ...

class MyDataModule(LightningDataModule):
    def __init__(self, root_path: Path):
        # All data should be read from the folder given in self.root_path
        self.root_path = root_path
    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        # The data should be read off self.root_path
        train_dataset = ...
        return DataLoader(train_dataset, batch_size=5, num_workers=5)
    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        # The data should be read off self.root_path
        val_dataset = ...
        return DataLoader(val_dataset, batch_size=5, num_workers=5)
    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        # The data should be read off self.root_path
        test_dataset = ...
        return DataLoader(test_dataset, batch_size=5, num_workers=5)

class MyContainer(LightningContainer):
    def __init__(self):
        super().__init__()
        self.azure_datasets = ["folder_name_in_azure_blob_storage"]
        self.local_datasets = [Path("/some/local/path")]
        self.max_epochs = 42

    def create_model(self) -> LightningModule:
        return MyLightningModel()

    def get_data_module(self) -> LightningDataModule:
        return MyDataModule(root_path=self.local_dataset)
```

By default, config files will be looked for in the folder "health_ml.configs". To specify config files
that live elsewhere, use a fully qualified name for the parameter `--model` - e.g. "MyModule.Configs.my_config.py"

## Outputting files during training

The Lightning model returned by `create_model` needs to write its output files to the current working directory.
When running inside of AzureML, the output folders will be directly under the project root. If not running inside
AzureML, a folder with a timestamp will be created for all outputs and logs.

When running in AzureML, the folder structure will be set up such that all files written
to the current working directory are later uploaded to Azure blob storage at the end of the AzureML job. The files
will also be later available via the AzureML UI.

## Trainer arguments

All arguments that control the PyTorch Lightning `Trainer` object are defined in the class `TrainerParams`. A
`LightningContainer` object inherits from this class. The most essential one is the `max_epochs` field, which controls
the `max_epochs` argument of the `Trainer`.

For example:

```python
from pytorch_lightning import LightningModule, LightningDataModule
from health_ml.lightning_container import LightningContainer

class MyContainer(LightningContainer):
    def __init__(self):
        super().__init__()
        self.max_epochs = 42

    def create_model(self) -> LightningModule:
        return MyLightningModel()

    def get_data_module(self) -> LightningDataModule:
        return MyDataModule(root_path=self.local_dataset)
```

### Optimizer and LR scheduler arguments

To the optimizer and LR scheduler: the Lightning model returned by `create_model` should define its own
`configure_optimizers` method, with the same signature as `LightningModule.configure_optimizers`,
and returns a tuple containing the Optimizer and LRScheduler objects

## Run inference with a pretrained model

You can use the hi-ml-runner in inference mode only by switching the `--run_inference_only` flag on and specifying
the model weights by setting `--src_checkpoint` argument that supports three types of checkpoints:

- A local path where the checkpoint is stored `--src_checkpoint=local/path/to/my_checkpoint/model.ckpt`
- A remote URL from where to download the weights `--src_checkpoint=https://my_checkpoint_url.com/model.ckpt`
- An AzureML run id where checkpoints are saved in `outputs/checkpoints`. For this specific use case, you can experiment
  with different checkpoints by setting `--src_checkpoint` according to the format
  `<azureml_run_id>:<optional/custom/path/to/checkpoints/><filename.ckpt>`. If no custom path is provided
  (e.g., `--src_checkpoint=AzureML_run_id:best.ckpt`), we assume the checkpoints to be saved in the default
  checkpoints folder `outputs/checkpoints`. If no filename is provided (e.g., `--src_checkpoint=AzureML_run_id`),
  the last epoch checkpoint `outputs/checkpoints/last.ckpt` will be loaded.

Refer to [Checkpoints Utils](checkpoints.md) for more details on how checkpoints are parsed.

Running the following command line will run inference using `MyContainer` model with weights from the checkpoint saved
in the AzureMl run `MyContainer_XXXX_yyyy` at the best validation loss epoch `/outputs/checkpoints/best_val_loss.ckpt`.

```bash
himl-runner --model=Mycontainer --run_inference_only --src_checkpoint=MyContainer_XXXX_yyyy:best_val_loss.ckpt
```

## Resume training from a given checkpoint

Analogously, one can resume training by setting `--src_checkpoint` and `--resume_training` to train a model longer.
The pytorch lightning trainer will initialize the lightning module from the given checkpoint corresponding to the best
validation loss epoch as set in the following comandline.

```bash
himl-runner --model=Mycontainer --cluster=my_cluster_name --src_checkpoint=MyContainer_XXXX_yyyy:best_val_loss.ckpt --resume_training
```

Warning: When resuming training, one should make sure to set `container.max_epochs` greater than the last epoch of the
specified checkpoint. A misconfiguration exception will be raised otherwise:

```text
pytorch_lightning.utilities.exceptions.MisconfigurationException: You restored a checkpoint with current_epoch=19, but you have set Trainer(max_epochs=4).
```

## Logging to AzureML when running outside AzureML

The runner offers the ability to log metrics to AzureML, even if the present training is not running
inside of AzureML. This adds an additional level of traceability for runs on GPU VMs, where there is otherwise
no record of any past training.

You can trigger this behaviour by specifying the `--log_from_vm` flag. For the `HelloWorld` model, this
will look like:

```bash
himl-runner --model=HelloWorld --log_from_vm
```

For logging to work, you need have a `config.json` file in the current working directory (or one of its
parent folders) that specifies the AzureML workspace itself. When starting the runner, you will be asked
to authenticate to AzureML.

There are two additional flags that can be used to control the logging behaviour:

- The `--experiment` flag sets which AzureML experiment to log to. By default, the experiment name will be
    the name of the model class (`HelloWorld` in the above example).
- The `--tag` flag sets the display name for the AzureML run. You can use that to give your run a memorable name,
    and later easily find it in the AzureML UI.

The following command will log to the experiment `my_experiment`, in a run that is labelled `my_first_run` in the UI:

```bash
himl-runner --model=HelloWorld --log_from_vm --experiment=my_experiment --tag=my_first_run
```

## Starting experiments with different seeds

To assess the variability of metrics, it is often useful to run the same experiment multiple times with different seeds.
There is a built-in functionality of the runner to do this. When adding the commandline flag `--different_seeds=3`, your
experiment will get run 3 times with seeds 0, 1 and 2. This is equivalent to starting the runner with arguments
`--random_seed=0`, `--random_seed=1` and `--random_seed=2`.

These runs will be started in parallel in AzureML via the HyperDrive framework. It is not possible to run with different
seeds on a local machine, other than by manually starting runs with `--random_seed=0` etc.

## Common problems with running in AML

1. `"Your total snapshot size exceeds the limit <SNAPSHOT_LIMIT>"`. Cause: The size of your source directory is larger than
   the limit that AML sets for snapshots. Solution: check for cache files, log files or other files that are not
   necessary for running your experiment and add them to a `.amlignore` file in the root directory. Alternatively, you
   can see Azure ML documentation for instructions on increasing this limit, although it will make your jobs slower.
2. `"FileNotFoundError"`. Possible cause: Symlinked files. Azure ML SDK v2 will resolve the symlink and attempt to upload
the resolved file. Solution: Remove symlinks from any files that should be uploaded to Azure ML.
