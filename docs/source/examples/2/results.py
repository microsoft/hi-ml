#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path

import health_azure.himl as himl
from health_azure.utils import WORKSPACE_CONFIG_JSON, get_most_recent_run


def main() -> None:
    path = Path(__file__).parent.resolve()

    workspace = himl.get_workspace(aml_workspace=None,
                                   workspace_config_path=path / WORKSPACE_CONFIG_JSON)

    run = get_most_recent_run(run_recovery_file=path / himl.RUN_RECOVERY_FILE,
                              workspace=workspace)
    log_root = path / "logs"
    log_root.mkdir(exist_ok=False)
    run.get_all_logs(destination=log_root)
    driver_log = log_root / "azureml-logs" / "70_driver_log.txt"
    log_text = driver_log.read_text()
    print(log_text)


if __name__ == "__main__":
    main()
