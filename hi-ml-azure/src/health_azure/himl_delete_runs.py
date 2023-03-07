#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

"""
Script to delete runs from an AzureML workspace.
The workspace is specified in the config.json file.
You can specify the run to delete with the --run argument. To delete whole experiments, use the --experiment argument.
With experiments, you can use wildcards to delete multiple experiments at once.
In addition, you can specify an exclusion list via a file. The exclusion list file should contain
one run ID per line.
"""

import fnmatch
from pathlib import Path
from typing import Optional
import sys

import param
from azureml.core import Workspace

src_folder = Path(__file__).parents[1]
sys.path.append(str(src_folder))
from health_azure.utils import get_workspace


class DeleteRunConfig(param.Parameterized):
    run: Optional[str] = param.String(default=None, allow_None=True, doc="Run ID to delete")
    experiment: Optional[str] = param.String(default=None, allow_None=True, doc="Experiment name to delete. This can contain * and ? as wildcards")
    exclusion_list: Optional[str] = param.String(default=None, allow_None=True, doc="Path to file containing runs to exclude")


def main(config: DeleteRunConfig) -> None:
    workspace: Workspace = get_workspace()
    exclusion_list = []
    if config.exclusion_list is not None:
        lines = Path(config.exclusion_list).read_text().splitlines()
        exclusion_list = [line.strip() for line in lines if line.strip() != ""]
    for name, experiment in workspace.experiments.items():
        if fnmatch.fnmatch(name, config.experiment):
            for run in experiment.get_runs():
                if run.id in exclusion_list:
                    print(f"Skipping run {run.id} because it is in the exclusion list")
                else:
                    print(f"Deleting run {run.id}")
                    # run.delete()


if __name__ == "__main__":
    config = DeleteRunConfig()
    config.experiment = "refs_pull_819_merge"
    main(config)
