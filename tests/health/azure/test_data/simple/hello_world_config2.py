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
from argparse import ArgumentParser

try:
    from health.azure.himl import submit_to_azure_if_needed, WorkspaceConfig  # type: ignore
except ImportError:
    logging.info("using local src")
    from src.health.azure.himl import submit_to_azure_if_needed, WorkspaceConfig  # type: ignore

logger = logging.getLogger('test.health.azure.test_data')
logger.setLevel(logging.DEBUG)

submit_to_azure_if_needed(
    workspace_config=WorkspaceConfig(
        os.getenv("TEST_WORKSPACE_NAME", ""),
        os.getenv("TEST_SUBSCRIPTION_ID", ""),
        os.getenv("TEST_RESOURCE_GROUP", "")),
    workspace_config_path=None,
    environment_variables=None)


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
