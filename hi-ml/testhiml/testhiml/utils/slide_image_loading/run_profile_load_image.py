from pathlib import Path

from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.compute import ComputeTarget

ws = Workspace.from_config()
print(ws)

env = Environment.from_dockerfile(name="jontri_image_load",
                                  dockerfile="./Dockerfile",
                                  conda_specification="./environment.yml")
print(env)

compute_target = ComputeTarget(workspace=ws, name='jontri1')

print(compute_target.get_status().serialize())

config = ScriptRunConfig(source_directory='./src',
                         script='profile_load_image.py',
                         compute_target=compute_target,
                         environment=env)

experiment = Experiment(workspace=ws,
                        name='day1-experiment-hello')

run = experiment.submit(config)
aml_url = run.get_portal_url()
print(aml_url)
