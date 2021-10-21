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
    log_root = path / "logs"
    log_root.mkdir(exist_ok=False)
    run.get_all_logs(destination=str(log_root))
    driver_log = log_root / "azureml-logs" / "70_driver_log.txt"
    log_text = driver_log.read_text()
    print(log_text)


if __name__ == "__main__":
    main()
