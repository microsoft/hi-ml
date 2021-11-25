#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import argparse
import os
from dataclasses import dataclass

import torch
from health_azure import set_environment_variables_for_multi_node, submit_to_azure_if_needed
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.plugins import DDPPlugin
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


@dataclass
class MultiNodeData:
    """
    Holds all MPI data for this process.
    """
    # the number of processes in this process's MPI_COMM_WORLD
    world_size: int
    # the MPI rank of this process in MPI_COMM_WORLD
    world_rank: int
    # the relative rank of this process on this node within its job. For example, if four processes in a job share a
    # node, they will each be given a local rank ranging from 0 to 3.
    world_local_rank: int
    # the number of process slots allocated to this job. Note that this may be different than the number of
    # processes in the job.
    universe_size: int
    # the number of ranks from this job that are running on this node.
    world_local_size: int
    # the relative rank of this process on this node looking across ALL jobs.
    world_node_rank: int


def get_multi_node_data() -> MultiNodeData:
    """
    Get MPI data, see https://www.open-mpi.org/faq/?category=running#mpi-environmental-variables.

    :return: A MultiNodeData holding values from the MPI environment variables.
    """
    return MultiNodeData(
        world_size=int(os.getenv("OMPI_COMM_WORLD_SIZE", "0")),
        world_rank=int(os.getenv("OMPI_COMM_WORLD_RANK", "0")),
        world_local_rank=int(os.getenv("OMPI_COMM_WORLD_LOCAL_RANK", "0")),
        universe_size=int(os.getenv("OMPI_UNIVERSE_SIZE", "0")),
        world_local_size=int(os.getenv("OMPI_COMM_WORLD_LOCAL_SIZE", "0")),
        world_node_rank=int(os.getenv("OMPI_COMM_WORLD_NODE_RANK", "0"))
    )


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
    parser.add_argument('--min_num_gpus', type=int, default=0,
                        help='Minimum number of gpus to use')

    args, unknown = parser.parse_known_args()

    run_info = submit_to_azure_if_needed(
        compute_cluster_name="training-nd24",
        ignored_folders=["lightning_logs", "logs", "MNIST", "outputs"],
        num_nodes=args.num_nodes,
        wait_for_completion=True,
        wait_for_completion_show_output=True)
    set_environment_variables_for_multi_node()

    az_batch_master_node = os.getenv("AZ_BATCH_MASTER_NODE", "?")
    print(f"az_batch_master_node: {az_batch_master_node}")

    multi_node_data = get_multi_node_data()
    print(f"multi_node_data: {multi_node_data}")

    mnist_model = MNISTModel()

    num_gpus = min(args.min_num_gpus, torch.cuda.device_count()) if torch.cuda.is_available() else 0
    effective_num_gpus = num_gpus * args.num_nodes

    print(f"num_nodes: {args.num_nodes}, num_gpus: {num_gpus}")

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

    # Initialize a trainer
    trainer = Trainer(default_root_dir=str(run_info.output_folder),
                      accelerator=accelerator,
                      strategy=strategy,
                      max_epochs=1,
                      num_nodes=args.num_nodes,
                      devices=devices)

    # Train the model ⚡
    trainer.fit(mnist_model)


if __name__ == "__main__":
    main()
