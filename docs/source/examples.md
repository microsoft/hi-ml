# Examples

## Basic integration

The sample [examples/1/sample.py](examples/1/sample.py) is a script that prints all the prime numbers up to (but not including) a target. It is simply intended to demonstrate a long running operation, that we want to run in Azure. It takes an optional command line argument of the target value and prints the primes to the console, using e.g.

```bash
cd examples/1
python sample.py -n 103
```

The sample [examples/2/sample.py](examples/2/sample.py) shows the minimal modifications to run this in AzureML. Firstly create an AzureML workspace and download the config file, as explained [here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-environment). The config file should be placed in the same folder as the sample script. A sample [Conda environment file](examples/2/environment.yml) is supplied. Import the [hi-ml package](https://pypi.org/project/hi-ml/) into the current environment. Finally add the following to the sample script:

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
* the print statement prints to the AzureML console output and is available in the `Output + logs` tab of the experiment in the `70_driver_log.txt` file, and can be downloaded from there.
* the command line arguments are passed through (apart from --azureml) when running in AzureML.
* a new file: `most_recent_run.txt` will be created containing an identifier of this AzureML run.

A sample script [examples/2/results.py](examples/2/results.py) demonstrates how to programmatically download the driver log file.

## Output files

The sample [examples/3/sample.py](examples/3/sample.py) demonstrates output file handling when running on AzureML. Because each run is performed in a separate VM or cluster any file output is not generally preserved. In order to keep the output it should be written to the `outputs` folder when running in AzureML. The AzureML infrastructure will preserve this and it will be available for download from the `outputs` folder in the `Output + logs` tab.

Make the following additions:

```python
    run_info = submit_to_azure_if_needed(

    ...

    parser.add_argument("-o", "--output", type=str, default="primes.txt", required=False, help="Output file name")

    ...

    output = run_info.output_folder / args.output
    output.write_text("\n".join(map(str, primes)))
```

When running locally this will create a subfolder called `outputs` and write the output to a file there. When running in AzureML the output will be available in a file in the Experiment.

A sample script [examples/3/results.py](examples/3/results.py) demonstrates how to programmatically download the output file.

## Output datasets

The sample [examples/4/sample.py](examples/4/sample.py) demonstrates output dataset handling when running on AzureML.

In this case, the following parameters are added to `submit_to_azure_if_needed`:

```python
        default_datastore="himldatasets",
        output_datasets=["himl_sample4_output"],
```

The `default_datastore` is required if using the simplest configuration for an output dataset, to just use the blob container name. There is an alternative that doesn't require the `default_datastore` and allows a different datastore for each dataset:

```python
from health.azure.datasets import DatasetConfig

    ...

        output_datasets=[DatasetConfig(name="himl_sample4_output", datastore="himldatasets")]
```

Now the output folder is constructed as follows:

```python
    output_folder = run_info.output_datasets[0] or Path("outputs") / "himl_sample4_output"
    output_folder.mkdir(parents=True, exist_ok=True)
    output = output_folder / args.output
```

When running in AzureML `run_info.output_datasets[0]` will be populated using the new parameter and the output will be written to that blob storage. When running locally `run_info.output_datasets[0]` will be None and a local folder will be created and used.

A sample script [examples/4/results.py](examples/4/results.py) demonstrates how to programmatically download the output dataset file.

## Input datasets

The sample [examples/5/sample.py](examples/5/sample.py) is modified from [https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/ml-frameworks/scikit-learn/train-hyperparameter-tune-deploy-with-sklearn/train_iris.py](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/ml-frameworks/scikit-learn/train-hyperparameter-tune-deploy-with-sklearn/train_iris.py) to work with input csv files.

A sample script [examples/5/inputs.py](examples/5/inputs.py) is provided to prepare the csv files. Run the script:

```bash
cd examples/5
python inputs.py
```

this will download the Iris dataset and create two csv files.

Because the csv files will be in the snapshot, this script can run in AzureML with only minimal modification as above, see the sample [examples/6/sample.py](examples/6/sample.py). It is not ideal to have the sample csv files in the snapshot, it is better to put them into blob storage and use input datasets.

A sample script [examples/7/inputs.py](examples/7/inputs.py) is provided to prepare the csv files and upload them to blob storage. Run the script:

```bash
cd examples/7
python inputs.py
```

In this case, the following parameters are added to `submit_to_azure_if_needed`:

```python
        default_datastore="himldatasets",
        input_datasets=["himl_sample7_input"],
```

The `default_datastore` is required if using the simplest configuration for an input dataset, to just use the blob container name. There are alternatives that do not require the `default_datastore` and allows a different datastore for each dataset, for example:

```python
from health.azure.datasets import DatasetConfig

    ...

        input_datasets=[DatasetConfig(name="himl_sample7_input", datastore="himldatasets"],
```

Now the input folder is constructed as follows:

```python
    input_folder = run_info.input_datasets[0] or Path("inputs")
```

When running in AzureML `run_info.input_datasets[0]` will be populated using the new parameter and the input will be mounted from blob storage. When running locally `run_info.input_datasets[0]` will be None and a local folder should be populated and used.


## Hyperdrive

The sample [examples/8/sample.py](examples/8/sample.py) demonstrates adding hyperparameter tuning. This shows the same hyperparameter search as in the [AzureML sample](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/ml-frameworks/scikit-learn/train-hyperparameter-tune-deploy-with-sklearn/train-hyperparameter-tune-deploy-with-sklearn.ipynb).

Make the following additions:

```python
from azureml.core import ScriptRunConfig
from azureml.train.hyperdrive import HyperDriveConfig, PrimaryMetricGoal, choice
from azureml.train.hyperdrive.sampling import RandomParameterSampling

    ...

def main() -> None:
    param_sampling = RandomParameterSampling({
        "--kernel": choice('linear', 'rbf', 'poly', 'sigmoid'),
        "--penalty": choice(0.5, 1, 1.5)
    })

    hyperdrive_config = HyperDriveConfig(
        run_config=ScriptRunConfig(source_directory=""),
        hyperparameter_sampling=param_sampling,
        primary_metric_name='Accuracy',
        primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
        max_total_runs=12,
        max_concurrent_runs=4)

    run_info = submit_to_azure_if_needed(
        ...
        hyperdrive_config=hyperdrive_config)
```

Note that this does not make sense to run locally, it should always be run in AzureML. When invoked with:

```bash
cd examples/8
python sample.py --azureml
```

this will perform a Hyperdrive run in AzureML, i.e. there will be 12 child runs, each randomly drawing from the parameter sample space. AzureML can plot the metrics from the child runs, but to do that, some small modifications are required.

Add in:

```python
    run = run_info.run

    ...

    args = parser.parse_args()
    run.log('Kernel type', np.str(args.kernel))
    run.log('Penalty', np.float(args.penalty))

    ...

    print('Accuracy of SVM classifier on test set: {:.2f}'.format(accuracy))
    run.log('Accuracy', np.float(accuracy))
```

and these metrics will be displayed on the child runs tab in the Experiment page on AzureML.

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

