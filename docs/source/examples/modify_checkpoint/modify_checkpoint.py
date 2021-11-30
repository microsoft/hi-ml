#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

# This script shows how to download files from a run, modify a checkpoint, and upload to a new run.
# From that new run, the modified checkpoint can be easily consumed in other training runs, both inside and
# outside AzureML.

from pathlib import Path

import torch

from health_azure import download_files_from_run_id, create_aml_run_object

if __name__ == "__main__":

    root_folder = Path.cwd()

    # This is the run from which we want to download checkpoints
    experiment_name = "SSLCheckpoint"
    old_run = f"{experiment_name}:SSLCheckpoint_1629396871_2263a0ec"

    # Specify where your AML workspace config.json file lives. If you set that to None, the code will try to find a file
    # called config.json in the current folder
    workspace_config_json = root_folder / "myworkspace.config.json"

    download_folder = Path(root_folder / "old_run")
    download_folder.mkdir(exist_ok=True)

    # Download all checkpoints in the run
    checkpoint_folder = "outputs/checkpoints"
    download_files_from_run_id(run_id=old_run, workspace_config_path=workspace_config_json,
                               output_folder=download_folder, prefix=checkpoint_folder)

    for file in download_folder.rglob("*.ckpt"):
        checkpoint = torch.load(file)
        state_dict = checkpoint['state_dict']
        # Here we modify the checkpoints: They reference weights from an older version of the code, delete any
        # such weights
        linear_head_states = [name for name in state_dict.keys() if name.startswith("non_linear_evaluator")]
        print(linear_head_states)
        if linear_head_states:
            print(f"Removing linear head from {file}")
            for state in linear_head_states:
                del checkpoint['state_dict'][state]
            torch.save(checkpoint, file)

    # Create a new AzureML run in the same experiment. The run will get a new unique ID
    new_run = create_aml_run_object(experiment_name=experiment_name, workspace_config_path=workspace_config_json)
    new_run.upload_folder(name=checkpoint_folder, path=str(download_folder / checkpoint_folder))
    new_run.complete()

    print(f"Uploaded the modified checkpoints to this run: {new_run.get_portal_url()}")
    print(f"Use this RunID to download the modified checkpoints: {new_run.id}")
