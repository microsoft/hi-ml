#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path

from health_azure import get_most_recent_run, get_workspace
from health_azure.himl import RUN_RECOVERY_FILE


def main() -> None:
    path = Path(__file__).parent.resolve()

    workspace = get_workspace()

    run = get_most_recent_run(run_recovery_file=path / RUN_RECOVERY_FILE,
                              workspace=workspace)

    # Either download all outputs:
    # run.download_files(prefix="outputs", output_directory=str(path))
    # Or download 1:
    outputs_root = path / "outputs"
    outputs_root.mkdir(exist_ok=False)
    run.download_file("outputs/primes.txt", str(outputs_root))


if __name__ == "__main__":
    main()
