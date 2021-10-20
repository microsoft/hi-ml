#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from azureml.core import Environment, Experiment, ScriptRunConfig, Workspace
from azureml.core.compute import ComputeTarget


workspace = Workspace.from_config()

environment = Environment.from_dockerfile(name='image_load_env',
                                          dockerfile='./Dockerfile',
                                          conda_specification='./environment.yml')

compute_target = ComputeTarget(workspace=workspace, name='jontri1')

config = ScriptRunConfig(source_directory='./src',
                         script='profile_load_image.py',
                         compute_target=compute_target,
                         environment=environment)

experiment = Experiment(workspace=workspace,
                        name='image_load_exp')

run = experiment.submit(config)
print(run.get_portal_url())
run.wait_for_completion(show_output=True)
