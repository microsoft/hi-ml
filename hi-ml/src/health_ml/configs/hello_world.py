#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torchmetrics import MeanAbsoluteError
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import StepLR, _LRScheduler
from torch.utils.data import DataLoader, Dataset

from health_ml.lightning_container import LightningContainer


def _create_1d_regression_dataset(n: int = 100, seed: int = 0) -> torch.Tensor:
    """Creates a simple 1-D dataset of a noisy linear function.

    :param n: The number of datapoints to generate, defaults to 100
    :param seed: Random number generator seed, defaults to 0
    :return: A tensor that contains X values in [:, 0] and Y values in [:, 1]
    """
    torch.manual_seed(seed)
    x = torch.rand((n, 1)) * 10
    y = 0.2 * x + 0.1 * torch.randn(x.size())
    xy = torch.cat((x, y), dim=1)
    return xy


def _split_crossval(xy: torch.Tensor, crossval_count: int, crossval_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a split of the given dataset along the first dimension for cross-validation.

    :param xy: The data that should be split. The split will be generated acros dimension 0.
    :param crossval_count: The number of splits in total
    :param crossval_index: The index of the split that should be generated (0 <= crossval_index < crossval_count)
    :return: A tuple of (training data, validation data)
    """
    n = xy.shape[0]
    split_size = n // crossval_count
    val_start = crossval_index * split_size
    val_end = (crossval_index + 1) * split_size
    train1_start = 0 if crossval_index == 0 else (crossval_index - 1) * split_size
    train1_end = 0 if crossval_index == 0 else val_start
    train2_start = val_end if crossval_index < (crossval_count - 1) else 0
    train2_end = n if crossval_index < (crossval_count - 1) else 0
    val = xy[val_start:val_end]
    train = torch.concat([xy[train1_start:train1_end], xy[train2_start:train2_end]])
    return (train, val)


class HelloWorldDataset(Dataset):
    """
    A simple 1dim regression task
    """

    def __init__(self, xy: torch.Tensor) -> None:
        """
        Creates the 1-dim regression dataset.

        :param xy: The raw data, x in the first column, y in the second column
        """
        super().__init__()  # type: ignore
        self.xy = xy

    def __len__(self) -> int:
        return self.xy.shape[0]

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        return {"x": self.xy[item][0:1], "y": self.xy[item][1:2]}


class HelloWorldDataModule(LightningDataModule):
    """
    A data module that gives the training, validation and test data for a simple 1-dim regression task.
    """

    def __init__(self, crossval_count: int, crossval_index: int) -> None:
        super().__init__()
        n_total = 200
        xy = _create_1d_regression_dataset(n=n_total)
        n_test = 40
        n_val = 50
        self.test = HelloWorldDataset(xy=xy[:n_test])
        if crossval_count <= 1:
            self.val = HelloWorldDataset(xy=xy[n_test:(n_test + n_val)])
            self.train = HelloWorldDataset(xy=xy[(n_test + n_val):])
        else:
            # This could be done via a library function like sklearn's KFold function, but we don't want to add
            # scikit-learn as a dependency just for this example.
            train, val = _split_crossval(xy[n_test:], crossval_count=crossval_count, crossval_index=crossval_index)
            self.val = HelloWorldDataset(xy=val)
            self.train = HelloWorldDataset(xy=train)

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        return DataLoader(self.train, batch_size=5)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        return DataLoader(self.val, batch_size=5)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        return DataLoader(self.test, batch_size=5)


class HelloRegression(LightningModule):
    """
    A simple 1-dim regression model.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Linear(in_features=1, out_features=1, bias=True)  # type: ignore
        self.test_mse: List[torch.Tensor] = []
        self.test_mae = MeanAbsoluteError()
        self._on_extra_val_epoch = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """
        This method is part of the standard PyTorch Lightning interface. For an introduction, please see
        https://pytorch-lightning.readthedocs.io/en/stable/starter/converting.html
        It runs a forward pass of a tensor through the model.

        :param x: The input tensor(s)
        :return: The model output.
        """
        return self.model(x)

    def training_step(self, batch: Dict[str, torch.Tensor], *args: Any, **kwargs: Any) -> torch.Tensor:  # type: ignore
        """
        This method is part of the standard PyTorch Lightning interface. For an introduction, please see
        https://pytorch-lightning.readthedocs.io/en/stable/starter/converting.html
        It consumes a minibatch of training data (coming out of the data loader), does forward propagation, and
        computes the loss.

        :param batch: The batch of training data
        :return: The loss value with a computation graph attached.
        """
        loss = self.shared_step(batch)
        self.log("loss", loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(  # type: ignore
        self, batch: Dict[str, torch.Tensor], *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """
        This method is part of the standard PyTorch Lightning interface. For an introduction, please see
        https://pytorch-lightning.readthedocs.io/en/stable/starter/converting.html
        It consumes a minibatch of validation data (coming out of the data loader), does forward propagation, and
        computes the loss.

        :param batch: The batch of validation data
        :return: The loss value on the validation data.
        """
        loss = self.shared_step(batch)
        self.log("val_loss", loss, on_epoch=True, on_step=False)
        return loss

    def shared_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        This is a convenience method to reduce code duplication, because training, validation, and test step share
        large amounts of code.

        :param batch: The batch of data to process, with input data and targets.
        :return: The MSE loss that the model achieved on this batch.
        """
        input = batch["x"]
        target = batch["y"]
        prediction = self.forward(input)
        return torch.nn.functional.mse_loss(prediction, target)  # type: ignore

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        """
        This method is part of the standard PyTorch Lightning interface. For an introduction, please see
        https://pytorch-lightning.readthedocs.io/en/stable/starter/converting.html
        It returns the PyTorch optimizer(s) and learning rate scheduler(s) that should be used for training.
        """
        optimizer = Adam(self.parameters(), lr=1e-1)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
        return [optimizer], [scheduler]

    def on_test_epoch_start(self) -> None:
        """
        This method is part of the standard PyTorch Lightning interface. For an introduction, please see
        https://pytorch-lightning.readthedocs.io/en/stable/starter/converting.html
        In this method, you can prepare data structures that need to be in place before evaluating the model on the
        test set (that is done in the test_step).
        """
        self.test_mse = []
        self.test_mae.reset()

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:  # type: ignore
        """
        This method is part of the standard PyTorch Lightning interface. For an introduction, please see
        https://pytorch-lightning.readthedocs.io/en/stable/starter/converting.html
        It evaluates the model in "inference mode" on data coming from the test set. It could, for example,
        also write each model prediction to disk.

        :param batch: The batch of test data.
        :param batch_idx: The index (0, 1, ...) of the batch when the data loader is enumerated.
        :return: The loss on the test data.
        """
        input = batch["x"]
        target = batch["y"]
        prediction = self.forward(input)
        # This illustrates two ways of computing metrics: Using standard torch
        loss = torch.nn.functional.mse_loss(prediction, target)  # type: ignore
        self.test_mse.append(loss)
        # Metrics computed using PyTorch Lightning objects. Note that these will, by default, attempt
        # to synchronize across GPUs.
        self.test_mae.update(preds=prediction, target=target)
        return loss

    def on_test_epoch_end(self) -> None:
        """
        This method is part of the standard PyTorch Lightning interface. For an introduction, please see
        https://pytorch-lightning.readthedocs.io/en/stable/starter/converting.html
        In this method, you can finish off anything to do with evaluating the model on the test set,
        for example writing aggregate metrics to disk.
        """
        average_mse = torch.mean(torch.stack(self.test_mse))
        Path("test_mse.txt").write_text(str(average_mse.item()))
        Path("test_mae.txt").write_text(str(self.test_mae.compute().item()))
        self.log("test_mse", average_mse, on_epoch=True, on_step=False)

    def on_run_extra_validation_epoch(self) -> None:
        self._on_extra_val_epoch = True


class HelloWorld(LightningContainer):
    """
    An example container for using the hi-ml runner. This container has methods
    to generate the actual Lightning model, and read out the datamodule that will be used for training.
    The number of training epochs is controlled at container level.
    You can train this model by running `python health_ml/runner.py --model=HelloWorld` on the local box,
    or via `python health_ml/runner.py --model=HelloWorld --cluster=<cluster name>` in AzureML
    """

    def __init__(self) -> None:
        super().__init__()
        self.local_dataset_dir = Path(__file__).resolve().parent
        self.max_epochs = 20
        self.save_checkpoint = False

    # This method must be overridden by any subclass of LightningContainer. It returns the model that you wish to
    # train, as a LightningModule
    def create_model(self) -> LightningModule:
        return HelloRegression()

    # This method must be overridden by any subclass of LightningContainer. It returns a data module, which
    # in turn contains 3 data loaders for training, validation, and test set.
    def get_data_module(self) -> LightningDataModule:
        assert self.local_dataset_dir is not None
        # If you would like to use the built-in cross validation functionality that runs training in parallel,
        # you need to provide the crossvalidation parameters in the LightningContainer to the datamodule. The
        # datamodule must carry out appropriate splitting of the data.
        return HelloWorldDataModule(crossval_count=self.crossval_count, crossval_index=self.crossval_index)

    def get_callbacks(self) -> List[Callback]:
        if self.save_checkpoint:
            return [ModelCheckpoint(dirpath=self.checkpoint_folder,
                                    monitor="val_loss",
                                    filename="checkpoint",
                                    auto_insert_metric_name=False,
                                    mode="min"),
                    *super().get_callbacks()]
        else:
            return super().get_callbacks()

    def get_additional_aml_run_tags(self) -> Dict[str, str]:
        return {"max_epochs": str(self.max_epochs)}
