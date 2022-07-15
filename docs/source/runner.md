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
`PandaImageNetMIL` class _anywhere_ in the `histopathology` namespace. You can think of this as
`health_cpath.*.PandaImageNetMIL`

## Running ML experiments in Azure ML

To train in AzureML, use the flag `--cluster` to specify the name of the cluster
in your Workspace that you want to submit the job to. So the whole command would look like:

```
himl-runner --model=HelloWorld --cluster=my_cluster_name
```
You can also specify `--num_nodes` if you wish to distribute the model training.

When starting the runner, you need to do that from a directory that contains all the code that your experiment needs:
The current working directory will be used as the root of all data that will be copied to AzureML to run your experiment.
(the only exception to this rule is if you start the runner from within an enlistment of the HI-ML GitHub repository).

AzureML needs to know which Python/Conda environment it should use. For that, the runner needs a file `environment.yml`
that contains a Conda environment definition. This file needs to be present either in the current working directory or
one of its parents. To specify a Conda environment that is located elsewhere, you can use

```shell
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
