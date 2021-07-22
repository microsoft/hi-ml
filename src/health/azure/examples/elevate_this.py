#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
Simple 'hello world' script to elevate to AML using our `submit_to_azure_if_needed` function.

Invoke like this:
    python elevate_this.py -m 'Hello World' --azureml -w=config.json -c=lite-testing-ds2 -e=environment.yml
"""
from argparse import ArgumentParser
from pathlib import Path

from src.health.azure.himl import submit_to_azure_if_needed


def main() -> None:
    """
    Write out the given message, in an AzureML 'experiment' if required.
    """
    parser = ArgumentParser()
    parser.add_argument("--azureml", action="store_true", required=False,
                        help="Flag to say whether to elevate script to AzureML")
    parser.add_argument("-m", "--message", type=str, required=True, help="The message to print out")
    parser.add_argument("-w", "--workspace_config_path", type=str, required=True, help="AzureML workspace config file")
    parser.add_argument("-c", "--compute_cluster_name", type=str, required=True, help="AzureML compute cluster to use")
    parser.add_argument("-e", "--conda_env", type=str, required=True, help="Conda environment YAML file")
    args = parser.parse_args()

    # N.B. submit_to_azure_if_needed reads the --azureml flag from sys.argv and so it is not passed in as a parameter.
    submit_to_azure_if_needed(
        None,
        args.workspace_config_path,
        args.compute_cluster_name,
        snapshot_root_directory=Path.cwd(),
        entry_script=Path(__file__),
        script_params=[f"--message='{args.message}"],
        conda_environment_file=args.conda_env)
    print(args.message)


if __name__ == "__main__":
    main()
