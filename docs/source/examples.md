# Examples

**Note**: All examples below contain links to sample scripts that are also included in the repository.
The experience is **optimized for use on readthedocs**. When navigating to the sample scripts on the github UI,
you will only see the `.rst` file that links to the `.py` file. To access the `.py` file, go to the folder that
contains the respective `.rst` file.

## Basic integration

The sample [examples/1/sample.py](examples/1/sample.rst) is a script that takes an optional command line argument of a target value and prints all the prime numbers up to (but not including) this target. It is simply intended to demonstrate a long running operation that we want to run in Azure. Run it using e.g.

```bash
cd examples/1
python sample.py -n 103
```

The sample [examples/2/sample.py](examples/2/sample.rst) shows the minimal modifications to run this in AzureML. Firstly create an AzureML workspace and download the config file, as explained [here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-environment). The config file should be placed in the same folder as the sample script. A sample [Conda environment file](examples/2/environment.rst) is supplied. Import the [hi-ml package](https://pypi.org/project/hi-ml/) into the current environment. Finally add the following to the sample script:

```python
from health_azure import submit_to_azure_if_needed
    ...
def main() -> None:
    _ = submit_to_azure_if_needed(
        compute_cluster_name="lite-testing-ds2",
        wait_for_completion=True,
        wait_for_completion_show_output=True)
```

Replace `lite-testing-ds2` with the name of a compute cluster created within the AzureML workspace.
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

then the function `submit_to_azure_if_needed` will perform all the required actions to run this script in AzureML and exit. Note that:

* code after `submit_to_azure_if_needed` is not run locally, but it is run in AzureML.
* the print statement prints to the AzureML console output and is available in the `Output + logs` tab of the experiment in the `70_driver_log.txt` if using the old AzureML runtime, or `std_log.txt` if using the new AzureML runtime, and can be downloaded from there.
* the command line arguments are passed through (apart from --azureml) when running in AzureML.
* a new file: `most_recent_run.txt` will be created containing an identifier of this AzureML run.

A sample script [examples/2/results.py](examples/2/results.rst) demonstrates how to programmatically download the driver log file.

## Output files

The sample [examples/3/sample.py](examples/3/sample.rst) demonstrates output file handling when running on AzureML. Because each run is performed in a separate VM or cluster then any file output is not generally preserved. In order to keep the output it should be written to the `outputs` folder when running in AzureML. The AzureML infrastructure will preserve this and it will be available for download from the `outputs` folder in the `Output + logs` tab.

Make the following additions:

```python
    from health_azure import submit_to_azure_if_needed
    run_info = submit_to_azure_if_needed(
    ...
    parser.add_argument("-o", "--output", type=str, default="primes.txt", required=False, help="Output file name")
    ...
    output = run_info.output_folder / args.output
    output.write_text("\n".join(map(str, primes)))
```

When running locally `submit_to_azure_if_needed` will create a subfolder called `outputs` and then the output can be written to the file `args.output` there. When running in AzureML the output will be available in the file `args.output` in the Experiment.

A sample script [examples/3/results.py](examples/3/results.rst) demonstrates how to programmatically download the output file.

## Output datasets

The sample [examples/4/sample.py](examples/4/sample.rst) demonstrates output dataset handling when running on AzureML.

In this case, the following parameters are added to `submit_to_azure_if_needed`:

```python
    from health_azure import submit_to_azure_if_needed
    run_info = submit_to_azure_if_needed(
        ...
        default_datastore="himldatasets",
        output_datasets=["himl_sample4_output"],
```

The `default_datastore` is required if using the simplest configuration for an output dataset, to just use the blob container name. There is an alternative that doesn't require the `default_datastore` and allows a different datastore for each dataset:

```python
from health_azure import DatasetConfig, submit_to_azure_if_needed
    ...
    run_info = submit_to_azure_if_needed(
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

A sample script [examples/4/results.py](examples/4/results.rst) demonstrates how to programmatically download the output dataset file.

For more details about datasets, see [here](datasets.md)

## Input datasets

This example trains a simple classifier on a toy dataset, first creating the dataset files and then in a second script
training the classifier.

The script [examples/5/inputs.py](examples/5/inputs.rst) is provided to prepare the csv files. Run the script to
download the Iris dataset and create two CSV files:
```bash
cd examples/5
python inputs.py
```

The training script [examples/5/sample.py](examples/5/sample.rst) is modified from
[https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/ml-frameworks/scikit-learn/train-hyperparameter-tune-deploy-with-sklearn/train_iris.py](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/ml-frameworks/scikit-learn/train-hyperparameter-tune-deploy-with-sklearn/train_iris.py) to work with input csv files.
Start it to train the actual classifier, based on the data files that were just written:
```bash
cd examples/5
python sample.py
```

### Including input files in the snapshot
When using very small datafiles (in the order of few MB), the easiest way to get the input data to Azure is to include
them in the set of (source) files that are uploaded to Azure. You can run the dataset creation script on your local
machine, writing the resulting two files to the same folder where your training script is located, and then submit
the training script to AzureML. Because the dataset files are in the same folder, they will automatically be uploaded
to AzureML.

However, it is not ideal to have the input files in the snapshot: The size of the snapshot is limited to 25 MB.
It is better to put the data files into blob storage and use input datasets.

### Creating the dataset in AzureML
The suggested way of creating a dataset is to run a script in AzureML that writes an output dataset. This is
particularly important for large datasets, to avoid the usually low bandwith from a local machine to the cloud.

This is shown in [examples/6/inputs.py](examples/6/inputs.rst):
This script prepares the CSV files in an AzureML run, and writes them to an output dataset called `himl_sample6_input`.
The relevant code parts are:
```python
run_info = submit_to_azure_if_needed(
    compute_cluster_name="lite-testing-ds2",
    default_datastore="himldatasets",
    output_datasets=["himl_sample6_input"])
# The dataset files should be written into this folder:
dataset = run_info.output_datasets[0] or Path("dataset")
```
Run the script:
```bash
cd examples/6
python inputs.py --azureml
```

You can now modify the training script [examples/6/sample.py](examples/6/sample.rst) to use the newly created dataset
`himl_sample6_input` as an input. To do that, the following parameters are added to `submit_to_azure_if_needed`:
```python
run_info = submit_to_azure_if_needed(
    compute_cluster_name="lite-testing-ds2",
    default_datastore="himldatasets",
    input_datasets=["himl_sample6_input"])
```
When running in AzureML, the dataset will be downloaded before running the job. You can access the temporary folder
where the dataset is available like this:
```python
input_folder = run_info.input_datasets[0] or Path("dataset")
```
The part behind the `or` statement is only necessary to keep a reasonable behaviour when running outside of AzureML:
When running in AzureML `run_info.input_datasets[0]` will be populated using input dataset specified in the call to
`submit_to_azure_if_needed`, and the input will be downloaded from blob storage. When running locally
`run_info.input_datasets[0]` will be `None` and a local folder should be populated and used.

The `default_datastore` is required if using the simplest configuration for an input dataset. There are
alternatives that do not require the `default_datastore` and allows a different datastore for each dataset, for example:

```python
from health_azure import DatasetConfig, submit_to_azure_if_needed
    ...
    run_info = submit_to_azure_if_needed(
        ...
        input_datasets=[DatasetConfig(name="himl_sample7_input", datastore="himldatasets"],
```

For more details about datasets, see [here](datasets.md)

### Uploading the input files manually
An alternative to writing the dataset in AzureML (as suggested above) is to create them on the local machine, and
upload them manually directly to Azure blob storage.

This is shown in [examples/7/inputs.py](examples/7/inputs.rst): This script prepares the CSV files
and uploads them to blob storage, in a folder called `himl_sample7_input`. Run the script:
```bash
cd examples/7
python inputs_via_upload.py
```

As in the above example, you can now modify the training script [examples/7/sample.py](examples/7/sample.rst) to use
an input dataset that has the same name as the folder where the files just got uploaded. In this case, the following
parameters are added to `submit_to_azure_if_needed`:

```python
    run_info = submit_to_azure_if_needed(
        ...
        default_datastore="himldatasets",
        input_datasets=["himl_sample7_input"],
```

## Hyperdrive

The sample [examples/8/sample.py](examples/8/sample.rst) demonstrates adding hyperparameter tuning. This shows the
same hyperparameter search as in the
[AzureML sample](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/ml-frameworks/scikit-learn/train-hyperparameter-tune-deploy-with-sklearn/train-hyperparameter-tune-deploy-with-sklearn.ipynb).

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

this will perform a Hyperdrive run in AzureML, i.e. there will be 12 child runs, each randomly drawing from the
parameter sample space. AzureML can plot the metrics from the child runs, but to do that, some small modifications are required.

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

## Training with k-fold cross validation in Azure ML

It is possible to create a parent run on Azure ML that is associated with one or more child runs (see [here](
https://docs.microsoft.com/en-us/azure/machine-learning/how-to-track-monitor-analyze-runs?tabs=python#create-child-runs)
for further information.) This is useful in circumstances such as k-fold cross-validation, where individual child run
perform validation on a different data split. When a [HyperDriveRun](
https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriverun?view=azure-ml-py)
is created in Azure ML, it follows this same principle and generates multiple child runs, associated with one parent.

To train with k-fold cross validation using `submit_to_azure_if_needed`, you must do two things.

1. Call the helper function `create_crossval_hyperdrive_config`
to create an AML HyperDriveConfig object representing your parent run. It will have one child run for each of the k-fold
splits you request, as follows

    ```python
    from health_azure import create_crossval_hyperdrive_config

     hyperdrive_config = create_crossval_hyperdrive_config(num_splits,
                                                           cross_val_index_arg_name=cross_val_index_arg_name,
                                                           metric_name=metric_name)
    ```
    where:
    - `num_splits` is the number of k-fold cross validation splits you require
    - `cross_val_index_arg_name` is the name of the argument given to each child run, whose value denotes which split
      that child represents (this parameter defaults to 'cross_validation_split_index', in which case, supposing you
      specified 2 cross validation splits, one would  receive the arguments ['--cross_validation_split_index' '0']
      and the other would receive ['--cross_validation_split_index' '1']]. It is up to you to then use these args
      to retrieve the correct split from your data.
    - `metrics_name` represents the name of a metric that you will compare your child runs by. **NOTE** the
    run will expect to find this metric, otherwise it will fail [as described here](
    https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters#log-metrics-for-hyperparameter-tuning
    )
    You can log this metric in your training script as follows:
     ```python
    from azureml.core import Run

    # Example of logging a metric called <metric_name> to an AML Run.
    loss = <my_loss_calc>
    run_log = Run.get_context()
    run_log.log(metric_name, loss)
    ```
   See the [documentation here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-log-view-metrics) for
   further explanation.

2. The hyperdrive_config returned above must be passed into the function `submit_to_azure_if_needed` as follows:

    ```python
     run_info = submit_to_azure_if_needed(
            ...
            hyperdrive_config=hyperdrive_config
    )
    ```
    This will create a parent (HyperDrive) Run, with `num_cross_validation_split` children - each one associated with a different data split.

## Retrieving the aggregated results of a cross validation/ HyperDrive run

You can retrieve a Pandas DataFrame of the aggregated results from your cross validation run as follows:

```python
from health_azure import aggregate_hyperdrive_metrics

df = aggregate_hyperdrive_metrics(run_id, child_run_arg_name)
```
where:
 - `run_id` is a string representing the id of your HyperDriveRun. Note that this **must** be an instance of an
AML HyperDriveRun.
- `child_run_arg_name` is a string representing the name of the argument given to each child run to denote its position
 relative to other child runs (e.g. this arg could equal 'child_run_index' - then each of your child runs should expect
 to receive the arg '--child_run_index' with a value <= the total number of child runs)


If your HyperDrive run has 2 children, each logging the metrics epoch, accuracy and loss, the result would look like this:

    |              | 0               | 1                  |
    |--------------|-----------------|--------------------|
    | epoch        | [1, 2, 3]       | [1, 2, 3]          |
    | accuracy     | [0.7, 0.8, 0.9] | [0.71, 0.82, 0.91] |
    | loss         | [0.5, 0.4, 0.3] | [0.45, 0.37, 0.29] |

 here each column is one of the splits/ child runs, and each row is one of the metrics you have logged to the run.

It is possible to log rows and tables in Azure ML by calling run.log_table and run.log_row respectively.
In this case, the DataFrame will contain a Dictionary entry instead of a list, where the keys are the
table columns (or keywords provided to log_row), and the values are the table values. e.g.

    |                | 0                                        | 1                                         |
    |----------------|------------------------------------------|-------------------------------------------|
    | accuracy_table |{'epoch': [1, 2], 'accuracy': [0.7, 0.8]} | {'epoch': [1, 2], 'accuracy': [0.8, 0.9]} |

It is also posisble to log plots in Azure ML by calling run.log_image and passing in a matplotlib plot. In
this case, the DataFrame will contain a string representing the path to the artifact that is generated by AML
(the saved plot in the Logs & Outputs pane of your run on the AML portal). E.g.

    |                | 0                                       | 1                                     |
    |----------------|-----------------------------------------|---------------------------------------|
    | accuracy_plot  | aml://artifactId/ExperimentRun/dcid.... | aml://artifactId/ExperimentRun/dcid...|


## Modifying checkpoints stored in an AzureML run

The script in [examples/modify_checkpoint/modify_checkpoint.py](examples/modify_checkpoint/modify_checkpoint.rst)
shows how checkpoints can be downloaded from an AzureML run, modified, and the uploaded back to a newly created run.

This can be helpful for example if networks architecture changed, but you do not want to re-train the stored models
with the new code.

The essential bits are:
* Download files from a run via `download_files_from_run_id`
* Modify the checkpoints
* Create a new run via `create_aml_run_object`
* Then use `Run.upload_folder` to upload all modified checkpoints to that new run. From there, they can be consumed
  in a follow-up training run again via `download_files_from_run_id`
