# Examples

## Controlling when to submit to AzureML and when not

By default, the `hi-ml` package assumes that you supply a commandline argument `--azureml` (that can be anywhere on 
the commandline) to trigger a submission of the present script to AzureML. If you wish to control it via a different
flag, coming out of your own argument parser, use the `submit_to_azureml` argument of the function
`health.azure.himl.submit_to_azure_if_needed`. 

## Hyperdrive

[HyperDrive runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)
can start multiple AzureML jobs in parallel. This can be used for tuning hyperparameters, or executing multiple
training runs for cross validation. To use that with the `hi-ml` package, simply supply a HyperDrive configuration
object as an additional argument. Note that this object needs to be created with an empty `run_config` argument (this
will later be replaced with the correct `run_config` that submits your script.)

```python
from azureml.core import ScriptRunConfig
from azureml.train.hyperdrive import GridParameterSampling, HyperDriveConfig, PrimaryMetricGoal, choice
from health.azure.himl import submit_to_azure_if_needed
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

## Run TensorBoard

From the command line, run the command

```run-tensorboard```

specifying one of 
`[--experiment_name] [--latest_run_path] [--run_recovery_ids]` 

This will start a TensorBoard session, by default running on port 6006. To use an alternative port, specify this with `--port`.

If `--experiment_name` is provided, the most recent Run from this experiment will be visualised.
If `--latest_run_path` is provided, the script will expect to find a RunId in this file.
Alternatively you can specify the Runs to visualise via  `--run_recovery_ids` or `--run_ids`.
You can specify the location where TensorBoard logs will be stored, using the `--run_logs_dir` argument.

If you choose to specify `--experiment_name`, you can also specify `--num_runs` to view and/or `--tags` to filter by.

If your AML config path is not ROOT_DIR/config.json, you must also specify `--config_path`.
<br><br>
## Download files from AML Runs

From the command line, run the command 

```download-aml-run```

specifying one of 
`[--experiment_name] [--latest_run_path] [--run_recovery_ids] [--run_ids]` 

If `--experiment_name` is provided, the most recent Run from this experiment will be downloaded.
If `--latest_run_path` is provided, the script will expect to find a RunId in this file.
Alternatively you can specify the Runs to download via  `--run_recovery_ids` or `--run_ids`.

The files associated with your Run(s) will be downloaded to the location specified with `--output_dir` (by default ROOT_DIR/outputs)

If you choose to specify `--experiment_name`, you can also specify `--num_runs` to view and/or `--tags` to filter by.

If your AML config path is not ROOT_DIR/config.json, you must also specify `--config_path`.

