# Hyperparameter Search via Hyperdrive (AML SDK v1)

[HyperDrive runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)
can start multiple AzureML jobs in parallel. This can be used for tuning hyperparameters, or executing multiple
training runs for cross validation. To use that with the `hi-ml` package, simply supply a HyperDrive configuration
object as an additional argument. Note that this object needs to be created with an empty `run_config` argument (this
will later be replaced with the correct `run_config` that submits your script.)

The example below shows a hyperparameter search that aims to minimize the validation loss `val_loss`, by choosing
one of three possible values for the learning rate commandline argument `learning_rate`.
```python
from azureml.core import ScriptRunConfig
from azureml.train.hyperdrive import GridParameterSampling, HyperDriveConfig, PrimaryMetricGoal, choice
from health_azure import submit_to_azure_if_needed
hyperdrive_config = HyperDriveConfig(
            run_config=ScriptRunConfig(source_directory=""),
            hyperparameter_sampling=GridParameterSampling(
                parameter_space={
                    "learning_rate": choice([0.1, 0.01, 0.001])
                }),
            primary_metric_name="val_loss",
            primary_metric_goal=PrimaryMetricGoal.MINIMIZE,
            max_total_runs=5
        )
submit_to_azure_if_needed(..., hyperdrive_config=hyperdrive_config)
```

For further examples, please check the [example scripts here](examples.md), and the
[HyperDrive documentation](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters).


# Hyperparameter Search in AML SDK v2

There is no concept of a HyperDriveConfig in AML SDK v2. Instead, hyperparameter search arguments are passed into a
command, and then the 'sweep' method is called [AML
docs](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters). To specify a hyperparameter
search job you must specify the method `get_parameter_tuning_args` in your Container. This should return a dictionary of
the arguments to be passed in to the command. For example:

```python
    def get_parameter_tuning_args(self) -> Dict[str, Any]:
        from azure.ai.ml.entities import Choice
        from health_azure.himl import (MAX_TOTAL_TRIALS_ARG, PARAM_SAMPLING_ARG, SAMPLING_ALGORITHM_ARG,
                                       PRIMARY_METRIC_ARG, GOAL_ARG)

        values = [0.1, 0.5, 0.9]
        argument_name = "learning_rate"
        param_sampling = {argument_name: Choice(values)}
        metric_name = "val/loss"

        hparam_args = {
            MAX_TOTAL_TRIALS_ARG: len(values),
            PARAM_SAMPLING_ARG: param_sampling,
            SAMPLING_ALGORITHM_ARG: "grid",
            PRIMARY_METRIC_ARG: metric_name,
            GOAL_ARG: "Minimize"

        }
        return hparam_args
```
Additional parameters, sampling strategies, limits etc. are described in the link above. Note that each job that is
created will receive an additional command line argument `<argument_name>` and it is your job to update the script to be
able to parse and use this argument.
