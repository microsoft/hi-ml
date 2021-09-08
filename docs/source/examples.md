# Examples

## Basic integration

The first sample [examples/1/sample.py](examples/1/sample.py) is a script that prints all the prime numbers up to (but not including) a target. It is simply intended to demonstrate a long running operation, that we want to run in Azure. It takes an optional command line argument of the target value and prints the primes to the console, using e.g.

```bash
cd examples/1
python sample.py -n 103
```

The second sample [examples/2/sample.py](examples/2/sample.py) shows the minimal modifications to run this in AzureML. Firstly create an AzureML workspace and download the config file, as explained [here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-environment). The config file should be placed in the same folder as the sample script. A sample [Conda environment file](examples/2/environment.yml) is supplied. Import the [hi-ml package](https://pypi.org/project/hi-ml/) into the current environment. Finally add the following to the sample script:

```python
from health.azure.himl import submit_to_azure_if_needed, WORKSPACE_CONFIG_JSON
```

and add the following at the beginning of main:

```python
    _ = submit_to_azure_if_needed(
        compute_cluster_name="lite-testing-ds2",
        workspace_config_path=WORKSPACE_CONFIG_JSON,
        conda_environment_file=Path("environment.yml"),
        wait_for_completion=True,
        wait_for_completion_show_output=True)
```

If this script is invoked as the first sample, e.g.

```bash
cd examples/2
python sample.py -n 103
```

then the output will be exactly the same. But if the script is invoked as follows:

```bash
cd examples/2
python sample.py -n 103 --azureml
```

then the function `submit_to_azure_if_needed` will do all the required actions to run this script in AzureML and exit. Note that:

* code after `submit_to_azure_if_needed` is not run.
* the print statement prints to the AzureML console output and is available in the `Output + logs` tab of the experiment in the `70_driver_log.txt` file.
* the command line arguments are passed through (apart from --azureml) when running in AzureML.
* a new file: `most_recent_run.txt` will be created containing an identifier of this AzureML run.

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

