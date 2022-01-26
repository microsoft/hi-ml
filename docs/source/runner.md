# Running ML experiments with hi-ml

The hi-ml toolbox is capable of training any PyTorch Lighting (PL) model inside of AzureML, making
use of the features:
- Working with different model in the same codebase, and selecting one by name
- Distributed training in AzureML
- Logging via AzureML's native capabilities
- Training on a local GPU machine or inside of AzureML without code changes
- Supply commandline overrides for model configuration elements, to quickly queue many jobs

This can be used by
- Defining a special container class, that encapsulates the PyTorch Lighting model to train, and the data that should
be used for training and testing.
- Adding essential trainer parameters like number of epochs to that container.
- Invoking the hi-ml runner and providing the name of the container class, like this:
`python health_ml/runner.py --model=MyContainer`. To train in AzureML, just add a `--azureml` flag.

There is a fully working example [HelloContainer](../../hi-ml/src/health-ml/configs/other/HelloContainer.py), that
implements a simple 1-dimensional regression model from data stored in a CSV file. You can run that
from the command line by `python health_ml/runner.py --model=HelloContainer`.

## Setup - creating your model config file

In order to use these capabilities, you need to implement a class deriving from
 `health_ml.lightning_container.LightningContainer`. This class encapsulates everything that is needed for training
 with PyTorch Lightning:

 For example:
 ```python
class MyContainer(LightningContainer):
    def __init__(self):
        super().__init__()
        self.azure_dataset_id = "folder_name_in_azure_blob_storage"
        self.local_dataset = "/some/local/path"
        self.num_epochs = 42

    def get_model(self) -> LightningModule:
        return MyLightningModel()

    def get_data_module(self) -> LightningDataModule:
        return MyDataModule(root_path=self.local_dataset)
```
The `get_model` method needs to return a subclass of PyTorch Lightning's [LightningModule](
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
        self.azure_dataset_id = "folder_name_in_azure_blob_storage"
        self.local_dataset = "/some/local/path"
        self.num_epochs = 42

    def get_model(self) -> LightningModule:
        return MyLightningModel()

    def get_data_module(self) -> LightningDataModule:
        return MyDataModule(root_path=self.local_dataset)
```

Your classes needs to be defined in a Python file in the `health_ml/configs` folder, otherwise it won't be picked up
correctly. If you'd like to have your model defined in a different folder, please specify the Python namespace via
the `--model_configs_namespace` argument. For example, use `--model_configs_namespace=My.Own.configs` if your
model configuration classes reside in folder `My/Own/configs` from the repository root.


### Outputting files during training

The Lightning model returned by `get_model` needs to write its output files to the current working directory.
When running inside of AzureML, the output folders will be directly under the project root. If not running inside
AzureML, a folder with a timestamp will be created for all outputs and logs.

When running in AzureML, the folder structure will be set up such that all files written
to the current working directory are later uploaded to Azure blob storage at the end of the AzureML job. The files
will also be later available via the AzureML UI.

### Trainer arguments
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

    def get_model(self) -> LightningModule:
        return MyLightningModel()

    def get_data_module(self) -> LightningDataModule:
        return MyDataModule(root_path=self.local_dataset)
```
### Optimizer and LR scheduler arguments
To the optimizer and LR scheduler: the Lightning model returned by `get_model` should define its own `configure_optimizers` method, with the same
signature as `LightningModule.configure_optimizers`, and returns a tuple containing the Optimizer and LRScheduler objects

