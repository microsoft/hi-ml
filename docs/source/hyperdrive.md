# Hyperparameter Search via Hyperdrive

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
