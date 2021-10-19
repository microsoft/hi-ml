#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.compute import ComputeTarget

ws = Workspace.from_config()

env = Environment.from_dockerfile(name="jontri_image_load",
                                  dockerfile="./Dockerfile",
                                  conda_specification="./environment.yml")

compute_target = ComputeTarget(workspace=ws, name='jontri1')

command = 'kernprof -l profile_load_image.py && ' \
          'python -m line_profiler profile_load_image.py.lprof > outputs/profile.txt'.split()

config = ScriptRunConfig(source_directory='./src',
                         command=command,
                         compute_target=compute_target,
                         environment=env)

experiment = Experiment(workspace=ws,
                        name='image_load_exp')

run = experiment.submit(config)
print(run.get_portal_url())
run.wait_for_completion(show_output=True)
