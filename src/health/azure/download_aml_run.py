from argparse import ArgumentParser, Namespace
from pathlib import Path

from azureml.core import workspace

from health.azure.azure_util import get_aml_runs

from health.azure.himl import get_workspace
from health.azure.run_tensorboard import determine_run_id_source


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type="str",
        defult="outputs",
        required=False,
        help="Path to directory to store  files downloaded from Run"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="config.json",
        required=False,
        help="Path to config.json where Workspace name is defined"
    )
    parser.add_argument(
        "--latest_run_path",
        type=str,
        required=False,
        help="Optional path to most_recent_run.txt where details on latest run are stored"
    )
    parser.add_argument(
        "--run_id",
        type=str,
        required=False,
        help="Optional Run ID that you wish to download files for"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    config_path = Path(args.config_path)
    if not config_path.is_file():
        raise ValueError(
            "You must provide a config.json file in the root folder to connect"
            "to an AML workspace. This can be downloaded from your AML workspace (see README.md)"
            )

    workspace = get_workspace(aml_workspace=None, workspace_config_path=config_path)

    run_id_source = determine_run_id_source(args)
    run = get_aml_runs(args, workspace, run_id_source)[0]

    # TODO: extend to multiple runs?
    try:
        run.download_files(output_directory=str(output_dir))
        print(f"Downloading files to {args.output_dir} ")
    except Exception as e:
        raise ValueError(f"Couldn't download files from run {args.run_id}: {e}")


if __name__ == "__main__":
    main()