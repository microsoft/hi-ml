#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import argparse
import os

import torch
from health_azure import get_multi_node_count, set_environment_variables_for_multi_node, submit_to_azure_if_needed
from health_ml.utils import AzureMLLogger, log_on_epoch

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.plugins import DDPPlugin
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


class MNISTModel(LightningModule):
    """
    Sample code taken from
    https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/mnist-hello-world.html
    """
    def __init__(self):  # type: ignore
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):  # type: ignore
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):  # type: ignore
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        log_on_epoch(self, "loss", loss)
        return loss

    def configure_optimizers(self):  # type: ignore
        return torch.optim.Adam(self.parameters(), lr=0.02)

    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self):  # type: ignore
        # download
        MNIST(".", train=True, download=True)

    def setup(self, stage=None):  # type: ignore
        self.mnist_train = MNIST(".", train=True, transform=transforms.ToTensor())

    def train_dataloader(self):  # type: ignore
        return DataLoader(self.mnist_train, batch_size=32)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_nodes', type=int, default=1,
                        help='Number of nodes to train on')

    args, unknown = parser.parse_known_args()

    run_info = submit_to_azure_if_needed(
        compute_cluster_name="testing-nc6",
        ignored_folders=["lightning_logs", "logs", "MNIST", "outputs"],
        num_nodes=args.num_nodes,
        wait_for_completion=True,
        wait_for_completion_show_output=True)

    print("~~~~~~~~~~ Begin Env Vars ~~~~~~~~~~~")
    for k, v in sorted(os.environ.items()):
        print(k + ':', v)
    print("~~~~~~~~~~ End Env Vars ~~~~~~~~~~~")

    set_environment_variables_for_multi_node()

    actual_num_nodes = get_multi_node_count()

    mnist_model = MNISTModel()

    num_gpus = torch.cuda.device_count()
    effective_num_gpus = num_gpus * actual_num_nodes

    print(f"num_nodes: {actual_num_nodes}, num_gpus: {num_gpus}")

    strategy = None
    if effective_num_gpus == 0:
        accelerator = "cpu"
        devices = 1
        message = "CPU"
    else:
        accelerator = "gpu"
        devices = num_gpus
        message = f"{devices} GPU"
        if effective_num_gpus > 1:
            strategy = DDPPlugin(find_unused_parameters=False)
            message += "s per node with DDP"
    print(f"Using {message}")

    azureml_logger = AzureMLLogger(enable_logging_outside_azure_ml=False)

    # Initialize a trainer
    trainer = Trainer(logger=[azureml_logger],
                      default_root_dir=str(run_info.output_folder),
                      accelerator=accelerator,
                      strategy=strategy,
                      max_epochs=1,
                      num_nodes=actual_num_nodes,
                      devices=devices)

    # Train the model ⚡
    trainer.fit(mnist_model)


if __name__ == "__main__":
    main()
