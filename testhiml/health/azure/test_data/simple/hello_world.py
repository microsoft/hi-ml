#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

"""
THIS FILE IS AUTO GENERATED. DO NOT EDIT THIS, EDIT THE TEMPLATE.

Simple 'hello world' script to elevate to AML using our `submit_to_azure_if_needed` function.
"""
import logging
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

from health.azure.himl import submit_to_azure_if_needed

logger = logging.getLogger('test.health.azure.test_data')
logger.setLevel(logging.DEBUG)

here = Path(__file__).parent.resolve()


def main() -> None:
    """
    Write out the given message, in an AzureML 'experiment' if required.
    """
    _ = submit_to_azure_if_needed(
        entry_script=Path(sys.argv[0]),
        compute_cluster_name=os.getenv("COMPUTE_CLUSTER_NAME", ""),
        conda_environment_file=here / "environment.yml",
        aml_workspace=None,
        workspace_config_path=None,
        snapshot_root_directory=here,
        environment_variables=None,
        wait_for_completion=True,
        wait_for_completion_show_output=True)

    parser = ArgumentParser()
    parser.add_argument("-m", "--message", type=str, required=True, help="The message to print out")
    args = parser.parse_args()

    print(args.message)


if __name__ == "__main__":
    main()
