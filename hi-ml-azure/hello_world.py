#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
Simple 'hello world' script to elevate to AML using our `submit_to_azure_if_needed` function.

Invoke like this:
    python hello_world.py --cluster <name_of_compute_cluster>

"""
import sys
from argparse import ArgumentParser

from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add hi-ml packages to sys.path so that AML can find them
himl_azure_root = Path(__file__).resolve().parent
folders_to_add = [himl_azure_root / "src"]
for folder in folders_to_add:
    if folder.is_dir():
        sys.path.insert(0, str(folder))

from health_azure import submit_to_azure_if_needed
from health_azure.logging import logging_to_stdout


def main() -> None:
    """
    Write out the given message, in an AzureML 'experiment' if required.

    First call submit_to_azure_if_needed.
    """
    parser = ArgumentParser()
    parser.add_argument("-c", "--cluster", type=str, required=True, help="The name of the compute cluster to run on")
    args = parser.parse_args()
    logging_to_stdout
    _ = submit_to_azure_if_needed(
        compute_cluster_name=args.cluster,
        strictly_aml_v1=True,
        submit_to_azureml=True,
        workspace_config_file=himl_azure_root/"config.json",
        snapshot_root_directory=himl_azure_root,
    )
    print("Hello Chris!")


if __name__ == "__main__":
    main()
