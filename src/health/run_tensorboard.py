#!/usr/bin/env python3
import json
import os

from argparse import ArgumentParser
from pathlib import Path

from azureml.core import Workspace
from azureml.tensorboard import Tensorboard

from azure_helpers.azure_util import get_most_recent_run
import sys
print(f"Sys path: {sys.path}")

def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--latest_run_path",
        type=str,
        default="most_recent_run.txt",
        required=False,
        help="Path to most_recent_run.txt where details on latest run are stored"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="config.json",
        required=False,
        help="Path to config.json where Workspace name is defined"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6006,
        required=False,
        help="The port to run Tensorboard on"
    )
    parser.add_argument(
        "--run_logs_dir",
        type=str,
        default="tensorboard_logs",
        required=False,
        help="Path to directory in which to store Tensorboard logs"
    )

    args = parser.parse_args()
    config_path = Path(args.config_path)
    latest_run_path = Path(args.latest_run_path)
    run_logs_dir = Path(args.run_logs_dir)

    # Load config and retrieve AML Workspace
    if config_path.is_file:
        with open(config_path, 'r') as f_path:
            config = json.load(f_path)
            subscription_id = config.get('subscription_id')
            resource_group = config.get('resource_group')
            workspace_name = config.get('workspace_name')
    else:
        subscription_id = os.getenv('subscription_id')
        resource_group = os.getenv('resource_group')
        workspace_name = os.getenv('workspace_name')

    # print(f"Subscription id: {subscription_id}\nresource group: {resource_group}\nworkspace name: {workspace_name}")
    workspace = Workspace(subscription_id, resource_group, workspace_name)

    # get the most recent run
    # TODO: option for multiple runs
    run = get_most_recent_run(latest_run_path, workspace)

    # start Tensorboard
    ts = Tensorboard(runs=run, local_root=run_logs_dir, port=args.port)

    ts.start()
    print("==============================================================================\n\n")
    input("Press Enter to close TensorBoard...")
    ts.stop()


if __name__ == "__main__":
    main()
