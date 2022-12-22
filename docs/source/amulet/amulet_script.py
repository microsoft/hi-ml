#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os
import sys
import logging
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from pytorch_lightning import Trainer
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.rank_zero import _get_rank

from pathlib import Path

# Add local source folders manually if the code is not installed as a package
repo_root = Path(__file__).parent.parent.parent.parent
for folder in [repo_root / "hi-ml-azure" / "src", repo_root / "hi-ml" / "src"]:
    if folder.is_dir():
        sys.path.append(str(folder))

from health_ml.utils.logging import AzureMLLogger  # noqa: E402
from health_azure.utils import (set_environment_variables_for_multi_node,  # noqa: E402
                                is_local_rank_zero, is_global_rank_zero)
from health_azure.amulet import (ENV_AMLT_PROJECT_NAME, ENV_AMLT_INPUT_OUTPUT,  # noqa: E402
                                 ENV_AMLT_DATAREFERENCE_OUTPUT, is_amulet_job, get_amulet_aml_working_dir,
                                 get_amulet_data_dir, get_amulet_output_dir, prepare_amulet_job)
from health_azure import submit_to_azure_if_needed  # noqa: E402


NUM_FEATURES = 4
DATASET_SIZE = 16


def show_environment() -> None:
    """Show various diagnostic information that are helpful for debugging failing Amulet jobs:
    Environment variables, Amulet input and output folder, etc."""
    print("System setup:")
    print("Full set of environment variables:")
    print(os.environ)

    amulet_project_name = os.environ.get(ENV_AMLT_PROJECT_NAME, None)
    amulet_input_output = os.environ.get(ENV_AMLT_INPUT_OUTPUT, None)
    amulet_output_dir = get_amulet_output_dir()
    amulet_datareference_data = get_amulet_data_dir()
    amulet_datareference_output = os.environ.get(ENV_AMLT_DATAREFERENCE_OUTPUT, None)

    print(f"{amulet_project_name=}")
    print(f"{amulet_input_output=}")
    print(f"{amulet_output_dir=}")
    print(f"{amulet_datareference_data=}")
    print(f"{amulet_datareference_output=}")

    print(f"{is_amulet_job()=}")
    print(f"{get_amulet_aml_working_dir()=}")
    print(f"{Path.cwd()=}")


def write_output_files() -> None:
    """Writes some output files to the output folders, to check that the output folder is correctly mounted.
    There are 2 ways of writing the output files, one that makes the files visible in AzureML, but not in the Amulet
    browser, and one that makes the files visible in the Amulet browser, but not in AzureML."""
    amlt_output_dir = get_amulet_output_dir()
    if amlt_output_dir is None:
        print("No Amulet output directory found")
    else:
        amlt_output_file = amlt_output_dir / "amulet_output.txt"
        print(f"Writing Amulet output file {amlt_output_file}")
        amlt_output_file.write_text("This is a test file written to the Amulet output folder")

    azureml_working_dir = get_amulet_aml_working_dir()
    if azureml_working_dir is None:
        print("No AzureML working directory found")
    else:
        azureml_output_file = azureml_working_dir / "azureml_output.txt"
        print(f"Writing AzureML output file {azureml_output_file}")
        azureml_output_file.write_text("This is a test file written to the AzureML output folder")


class RandomDataset(Dataset):
    """A dummy dataset with random dataset to enable training a simple model."""

    def __init__(self, length: int):
        self.len = length
        self.data = torch.randn(length, NUM_FEATURES)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class BoringModel(LightningModule):
    """A dummy model to illustrate and test training with PyTorch Lightning."""

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(NUM_FEATURES, 2)

    def forward(self, x):
        return self.layer(x)

    def loss(self, batch, prediction):
        return torch.nn.functional.mse_loss(prediction, torch.ones_like(prediction))  # type: ignore

    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(batch, output)
        self.log("loss_step", loss, on_step=True, on_epoch=False)
        self.log("loss_epoch", loss, on_step=False, on_epoch=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(batch, output)
        return {"x": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]


class BoringDataModule(LightningDataModule):
    """A dummy data module to illustrate and test training with PyTorch Lightning."""

    def __init__(self):
        super().__init__()

    def setup(self, stage: Optional[str] = None):
        self.random_full = RandomDataset(DATASET_SIZE * 4)
        if stage == "fit" or stage is None:
            self.random_train = Subset(self.random_full, indices=range(DATASET_SIZE))

        if stage in ("fit", "validate") or stage is None:
            self.random_val = Subset(self.random_full, indices=range(DATASET_SIZE, DATASET_SIZE * 2))

        if stage == "test" or stage is None:
            self.random_test = Subset(self.random_full, indices=range(DATASET_SIZE * 2, DATASET_SIZE * 3))

        if stage == "predict" or stage is None:
            self.random_predict = Subset(self.random_full, indices=range(DATASET_SIZE * 3, DATASET_SIZE * 4))

    def train_dataloader(self):
        return DataLoader(self.random_train)

    def val_dataloader(self):
        return DataLoader(self.random_val)

    def test_dataloader(self):
        return DataLoader(self.random_test)

    def predict_dataloader(self):
        return DataLoader(self.random_predict)


def run_training_loop(logging_folder: Optional[Path] = None) -> None:
    """Runs a simple PyTorch Lightning training loop, with a logger that writes to a given output folder.

    :param logging_folder: The folder to write the logs to. If None, no logging will be performed."""
    num_gpus = torch.cuda.device_count()
    strategy = None
    if num_gpus == 0:
        accelerator = "cpu"
        devices = 1
        message = "CPU"
    else:
        accelerator = "gpu"
        devices = num_gpus
        message = f"{devices} GPU"
        if num_gpus > 1:
            strategy = DDPStrategy()
            message += "s per node with DDP"
    logging.info(f"Using {message}")
    loggers = []
    if logging_folder:
        print(f"Logging via Tensorboard to {logging_folder}")
        loggers.append(TensorBoardLogger(save_dir=str(logging_folder), name="Lightning", version=""))
    else:
        print("Logging disabled")
    # Write all metrics also to AzureML natively, so that they are visible in the AzureML UI
    loggers.append(AzureMLLogger())

    trainer = Trainer(accelerator=accelerator,
                      strategy=strategy,
                      max_epochs=2,
                      logger=loggers,
                      num_nodes=1,
                      devices=devices,
                      # Setting the logging interval to a very small value because we have a tiny dataset
                      log_every_n_steps=1
                      )
    model = BoringModel()
    data_module = BoringDataModule()
    trainer.fit(model, datamodule=data_module)


def show_all_diagnostic_info() -> None:
    """Shows all diagnostic information that is essential to check that Amulet/AzureML are correctly set up."""
    if is_global_rank_zero():
        print("Global rank 0: print environment variables")
        show_environment()
        write_output_files()
    else:
        print("Rank != 0: Skipping environment and output file writing")
    # Pytorch Lightning uses this function to determine the process rank. It is wrongly saying that all processes
    # have rank 0, unless we remove the RANK environment variable (done below in prepare_amulet_job)
    print(f"_get_rank = {_get_rank()}")
    print(f"is_local_rank_zero = {is_local_rank_zero()}")
    print(f"is_global_rank_zero = {is_global_rank_zero()}")
    data_dir = get_amulet_data_dir()
    if data_dir:
        print(f"Data container is mounted at {data_dir}")
        # data_dir is now set to the container mounting point. Append a subfolder to it to get the actual data folder,
        # and use that in your dataloaders.
    else:
        print("No data container mounted - this is probably not an Amulet job")


def main() -> None:
    print("Starting main process")
    show_all_diagnostic_info()
    # Add this part if you want to enable submitting to AzureML from the same codebase
    submit_to_azure_if_needed(compute_cluster_name="litetesting-ds2")
    prepare_amulet_job()
    set_environment_variables_for_multi_node()
    # Run the training loop, and write logs such that the Tensorboard file is accessible via Amulet.
    run_training_loop(logging_folder=get_amulet_output_dir())


if __name__ == '__main__':
    main()
