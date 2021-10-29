#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path

from azureml.core import Environment, ScriptRunConfig
from azureml.core.compute import ComputeTarget

from health_azure import get_workspace, submit_run
from health_azure.utils import WORKSPACE_CONFIG_JSON


# A compute instance, with a GPU, required for running cuCIM.
GPU_TESTING_INSTANCE_NAME = "testing-standard-nc6"

here = Path(__file__).parent.resolve()

workspace = get_workspace(aml_workspace=None,
                          workspace_config_path=here / WORKSPACE_CONFIG_JSON)

environment = Environment.from_dockerfile(name='image_load_env',
                                          dockerfile='./Dockerfile',
                                          conda_specification='./environment.yml')

compute_target = ComputeTarget(workspace=workspace, name=GPU_TESTING_INSTANCE_NAME)

config = ScriptRunConfig(source_directory='./src',
                         script='profile_load_image.py',
                         compute_target=compute_target,
                         environment=environment)

run = submit_run(workspace=workspace,
                 experiment_name='image_load_exp',
                 script_run_config=config,
                 wait_for_completion=True,
                 wait_for_completion_show_output=True)
