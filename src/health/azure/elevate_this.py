#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
Simple 'hello world' script to elevate to AML using our `submit_to_azure_if_needed` function.
"""
from argparse import ArgumentParser

from aml import submit_to_azure_if_needed


def main() -> None:
    """
    Write out the given message, in an AzureML 'experiment' if required.
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--message", type=str, required=True, help="The message to print out")
    parser.add_argument("-p", "--workspace_config_path", type=str, required=True, help="AzureML workspace config file")
    args = parser.parse_args()

    submit_to_azure_if_needed(
        None,
        args.workspace_config_path)
    print(args.message)


if __name__ == "__main__":
    main()
