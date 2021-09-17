#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from azureml.core import Environment, Experiment, ScriptRunConfig, Workspace


def main() -> None:
    ws = Workspace.from_config("config.json")
    experiment = Experiment(ws, "tensorboard_test")
    config = ScriptRunConfig(
        source_directory='.',
        script="pytorch_sample.py",
        compute_target="<name of compute target>"
    )
    env = Environment.from_conda_specification("TensorboardTestEnv", "tensorboard_env.yml")
    config.run_config.environment = env

    run = experiment.submit(config)
    run.wait_for_completion()


if __name__ == "__main__":
    main()
