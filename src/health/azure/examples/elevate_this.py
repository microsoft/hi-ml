#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
Simple 'hello world' script to elevate to AML using our `submit_to_azure_if_needed` function.

Invoke like this:
    python elevate_this.py -m 'Hello World' --azureml
or:
    python elevate_this.py --message='Hello World' --azureml

N.B. The --azureml flag mus match the constant AZUREML_COMMANDLINE_FLAG in health.azure.himl
"""
from argparse import ArgumentParser
from pathlib import Path

from health.azure.himl import submit_to_azure_if_needed


def main() -> None:
    """
    Write out the given message, in an AzureML 'experiment' if required.

    First call submit_to_azure_if_needed.
    """
    _ = submit_to_azure_if_needed(
        compute_cluster_name="lite-testing-ds2",
        workspace_config_file=Path("config.json").absolute(),
        snapshot_root_directory=Path.cwd().parent.parent.parent,
        entry_script=Path(__file__).absolute(),
        conda_environment_file=Path("environment.yml").absolute(),
        wait_for_completion=True,
        wait_for_completion_show_output=True)

    parser = ArgumentParser()
    parser.add_argument("-m", "--message", type=str, required=True, help="The message to print out")
    args = parser.parse_args()

    print(f"The message was: {args.message}")


if __name__ == "__main__":
    main()
