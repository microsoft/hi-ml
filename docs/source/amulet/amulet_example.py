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

repo_root = Path(__file__).parent.parent.parent.parent
himl_azure_folder = repo_root / "hi-ml-azure" / "src"
assert himl_azure_folder.is_dir()
sys.path.append(str(himl_azure_folder))
himl_folder = repo_root / "hi-ml" / "src"
assert himl_folder.is_dir()
sys.path.append(str(himl_folder))

from health_azure import submit_to_azure_if_needed
from health_azure.utils import (ENV_AMLT_PROJECT_NAME, ENV_AMLT_INPUT_OUTPUT, ENV_AMLT_OUTPUT_DIR,
                                ENV_AMLT_DATAREFERENCE_DATA, ENV_AMLT_DATAREFERENCE_OUTPUT, ENV_RANK,
                                set_environment_variables_for_multi_node, is_amulet_job, get_amlt_aml_working_dir,
                                is_local_rank_zero, is_global_rank_zero)
from health_ml.utils.logging import AzureMLLogger

NUM_FEATURES = 4
DATASET_SIZE = 16


class RandomDataset(Dataset):
    def __init__(self, length: int):
        self.len = length
        self.data = torch.randn(length, NUM_FEATURES)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class BoringModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(NUM_FEATURES, 2)

    def forward(self, x):
        return self.layer(x)

    def loss(self, batch, prediction):
        return torch.nn.functional.mse_loss(prediction, torch.ones_like(prediction))

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


def _path_from_env(env_name: str) -> Optional[Path]:
    """Reads a path from an environment variable, and returns it as a Path object
    if the variable is set. Returns None if the variable is not set or empty.

    :param env_name: The name of the environment variable to read.
    :return: The path read from the environment variable, or None if the variable is not set."""
    path = os.getenv(env_name, None)
    if path is None or path == "":
        return None
    return Path(path)

def get_amulet_output_dir() -> Optional[Path]:
    return _path_from_env(ENV_AMLT_OUTPUT_DIR)

def get_amulet_data_dir() -> Optional[Path]:
    """Gets the directory where the data for the current Amulet job is stored.

    :return: The directory where the data for the current Amulet job is stored.
        Returns None if the current job is not an Amulet job, or no data container is set.
    """
    return _path_from_env(ENV_AMLT_DATAREFERENCE_DATA)

def show_environment() -> None:
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
    print(f"{get_amlt_aml_working_dir()=}")
    print(f"{Path.cwd()=}")


def write_output_files() -> None:

    amlt_output_dir = get_amulet_output_dir()
    if amlt_output_dir is None:
        print("No Amulet output directory found")
    else:
        amlt_output_file = amlt_output_dir / "amulet_output.txt"
        print(f"Writing Amulet output file {amlt_output_file}")
        amlt_output_file.write_text("This is a test file written to the Amulet output folder")

    azureml_working_dir = get_amlt_aml_working_dir()
    if azureml_working_dir is None:
        print("No AzureML working directory found")
    else:
        azureml_output_file = azureml_working_dir / "azureml_output.txt"
        print(f"Writing AzureML output file {amlt_output_file}")
        azureml_output_file.write_text("This is a test file written to the AzureML output folder")


def run_training_loop(logging_folder: Optional[Path] = None) -> None:
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
    loggers.append(AzureMLLogger())

    trainer = Trainer(accelerator=accelerator,
                      strategy=strategy,
                      max_epochs=2,
                      logger=loggers,
                      num_nodes=1,
                      devices=devices
                      )
    model = BoringModel()
    data_module = BoringDataModule()
    trainer.fit(model, datamodule=data_module)


def prepare_amulet_job() -> None:
    # The RANK environment is set by Amulet, but not by AzureML.If set, PyTorch Lightning will think that all
    # processes are running at rank 0 in its `rank_zero_only` decorator, which will cause the logging to fail.
    if ENV_RANK in os.environ:
        del os.environ[ENV_RANK]

def main() -> None:
    if is_global_rank_zero():
        show_environment()
        write_output_files()
    else:
        print("Not rank 0, skipping environment and output file writing")
    prepare_amulet_job()
    data_dir = get_amulet_data_dir()
    logging.error(f"_get_rank = {_get_rank()}")
    logging.error(f"is_local_rank_zero = {is_local_rank_zero()}")
    logging.error(f"is_global_rank_zero = {is_global_rank_zero()}")
    if data_dir:
        print(f"Data container is mounted at {data_dir}")
    else:
        print("No data container mounted - this is probably not an Amulet job")

    submit_to_azure_if_needed(compute_cluster_name="litetesting-ds2")
    set_environment_variables_for_multi_node()
    run_training_loop(logging_folder=get_amulet_output_dir())


if __name__ == '__main__':
    main()
