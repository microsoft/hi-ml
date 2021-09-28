#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import argparse
from pathlib import Path

from azureml.core import Run

from health.azure.azure_util import download_run_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)

    args = parser.parse_args()
    output_path = Path(args.output_path)
    output_path.mkdir(exist_ok=True)
    print(f"Created output directory: {output_path}")

    run_ctx = Run.get_context()
    available_files = run_ctx.get_file_names()
    first_file_name = available_files[0]
    output_file_path = output_path / first_file_name

    download_run_files(run_ctx, output_path)

    run_ctx.download_file(first_file_name, output_file_path=output_file_path)
    print(f"Downloaded file {first_file_name} to location {output_file_path}")


if __name__ == "__main__":
    main()
