#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
Simple 'hello world' script to elevate to AML using our `submit_to_azure_if_needed` function.

Invoke like this:
    python elevate_this.py -m 'Hello World' --azureml -w=config.json -c=lite-testing-ds2 -e=environment.yml
or:
    python elevate_this.py --message='Hello World :-)' --workspace_config_path=config.json\
         --compute_cluster_name=lite-testing-ds2 --conda_env=environment.yml --azureml
"""
from argparse import ArgumentParser
from pathlib import Path

from health.azure.himl import submit_to_azure_if_needed


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

    snapshot_root_directory = Path.cwd().parent.parent.parent
    workspace_config_path = Path(args.workspace_config_path).absolute()
    entry_script = Path(__file__).absolute()
    conda_environment_file = Path(args.conda_env).absolute()

    script_params = [
        f"--message='{args.message}",
        f"--workspace_config_path={args.workspace_config_path}",
        f"--compute_cluster_name={args.compute_cluster_name}",
        f"--conda_env={args.conda_env}",
    ]

    # N.B. submit_to_azure_if_needed reads the --azureml flag from sys.argv and so it is not passed in as a parameter.
    _ = submit_to_azure_if_needed(
        workspace_config=None,
        workspace_config_path=workspace_config_path,
        compute_cluster_name=args.compute_cluster_name,
        snapshot_root_directory=snapshot_root_directory,
        entry_script=entry_script,
        script_params=script_params,
        conda_environment_file=conda_environment_file)

    if not args.azureml:
        print(args.message)


if __name__ == "__main__":
    main()
