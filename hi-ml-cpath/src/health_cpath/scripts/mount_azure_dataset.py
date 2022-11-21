#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
import sys
import time
from typing import Any, Optional
from azureml.core import Workspace

himl_histo_root_dir = Path(__file__).parent.parent.parent
himl_root = himl_histo_root_dir.parent.parent
himl_azure_package_root = himl_root / "hi-ml-azure" / "src"
sys.path.insert(0, str(himl_azure_package_root))

from health_azure import DatasetConfig  # noqa: E402
from health_azure.utils import get_workspace  # noqa: E402


def mount_dataset(dataset_id: str, tmp_root: str = "/tmp/datasets", aml_workspace: Optional[Workspace] = None) -> Any:
    ws = get_workspace(aml_workspace)
    target_folder = "/".join([tmp_root, dataset_id])
    dataset = DatasetConfig(name=dataset_id, target_folder=target_folder, use_mounting=True)
    _, mount_ctx = dataset.to_input_dataset_local(strictly_aml_v1=True, workspace=ws)
    assert mount_ctx is not None  # for mypy
    mount_ctx.start()
    return mount_ctx


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # Run this script as "python mount_azure_dataset.py --dataset_id TCGA-CRCk"
    parser.add_argument('--dataset_id', type=str,
                        help='Name of the Azure dataset e.g. PANDA or TCGA-CRCk')
    args = parser.parse_args()
    # It is essential that the mount context is returned from the mounting function and referenced here.
    # If not, mounting will be stopped, and the files are no longer available.
    _ = mount_dataset(args.dataset_id)
    print("The mounted dataset will only be available while this script is running. Press Ctrl-C to terminate it.`")
    while True:
        time.sleep(60)
