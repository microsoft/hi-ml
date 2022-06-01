#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from health_azure import DatasetConfig
from health_azure.utils import get_workspace


def mount_dataset(dataset_id: str) -> str:
    ws = get_workspace()
    target_folder = "/tmp/datasets/" + dataset_id
    dataset = DatasetConfig(
        name=dataset_id, target_folder=target_folder, use_mounting=True)
    dataset_mount_folder, mount_ctx = dataset.to_input_dataset_local(ws)
    assert mount_ctx is not None  # for mypy
    mount_ctx.start()
    return str(dataset_mount_folder)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # Run this script as "python mount_azure_dataset.py --dataset_id TCGA-CRCk"
    parser.add_argument('--dataset_id', type=str,
                        help='Name of the Azure dataset e.g. PANDA or TCGA-CRCk')
    args = parser.parse_args()
    mount_dataset(args.dataset_id)
