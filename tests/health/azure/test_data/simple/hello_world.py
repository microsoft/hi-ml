#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

"""
Simple 'hello world' script to elevate to AML using our `submit_to_azure_if_needed` function.
"""
import logging
from argparse import ArgumentParser

try:
    from health.azure.aml import submit_to_azure_if_needed  # type: ignore
except ImportError:
    logging.info("using local src")
    from src.health.azure.aml import submit_to_azure_if_needed  # type: ignore

logger = logging.getLogger('test.health.azure.test_data')
logger.setLevel(logging.DEBUG)

submit_to_azure_if_needed(
    workspace_config=None,
    workspace_config_path=None)


def main() -> None:
    """
    Write out the given message, in an AzureML 'experiment' if required.
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--message", type=str, required=True, help="The message to print out")
    args = parser.parse_args()

    print(args.message)


if __name__ == "__main__":
    main()
