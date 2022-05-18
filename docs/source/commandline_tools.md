# Commandline tools

## Run TensorBoard

From the command line, run the command

```himl-tb```

specifying one of
`[--experiment] [--latest_run_file] [--run]`

This will start a TensorBoard session, by default running on port 6006. To use an alternative port, specify this with `--port`.

If `--experiment` is provided, the most recent Run from this experiment will be visualised.
If `--latest_run_file` is provided, the script will expect to find a RunId in this file.
Alternatively you can specify the Runs to visualise via  `--run`. This can be a single run id, or multiple ids separated by commas. This argument also accepts one or more run recovery ids, although these are not recommended since it is no longer necessary to provide an experiment name in order to recovery an AML Run.

By default, this tool expects that your TensorBoard logs live in a folder named 'logs' and will create a similarly named folder in your root directory. If your TensorBoard logs are stored elsewhere, you can specify this with the `--log_dir` argument.

If you choose to specify `--experiment`, you can also specify `--num_runs` to view and/or `--tags` to filter by.

If your AML config path is not ROOT_DIR/config.json, you must also specify `--config_file`.

To see an example of how to create TensorBoard logs using PyTorch on AML, see the
[AML submitting script](examples/9/aml_sample.rst) which submits the following [pytorch sample script](examples/9/pytorch_sample.rst). Note that to run this, you'll need to create an environment with pytorch and tensorboard as dependencies, as a minimum. See an [example conda environemnt](examples/9/tensorboard_env.rst). This will create an experiment named 'tensorboard_test' on your Workspace, with a single run. Go to outputs + logs -> outputs to see the tensorboard events file.
## Download files from AML Runs

From the command line, run the command

```himl-download```

specifying one of
`[--experiment] [--latest_run_file] [--run]`

If `--experiment` is provided, the most recent Run from this experiment will be downloaded.
If `--latest_run_file` is provided, the script will expect to find a RunId in this file.
Alternatively you can specify the Run to download via `--run`. This can be a single run id, or multiple ids separated by commas. This argument also accepts one or more run recovery ids, although these are not recommended since it is no longer necessary to provide an experiment name in order to recovery an AML Run.

The files associated with your Run will be downloaded to the location specified with `--output_dir` (by default ROOT_DIR/outputs)

If you choose to specify `--experiment`, you can also specify `--tags` to filter by.

If your AML config path is not `ROOT_DIR/config.json`, you must also specify `--config_file`.


## Creating your own command line tools

When creating your own command line tools that interact with the Azure ML ecosystem, you may wish to use the
 `AmlRunScriptConfig` class for argument parsing. This gives you a quickstart way for accepting command line arguments to
 specify the following

  - experiment: a string representing the name of an Experiment, from which to retrieve AML runs
  - tags: to filter the runs within the given experiment
  - num_runs: to define the number of most recent runs to return from the experiment
  - run: to instead define one or more run ids from which to retrieve runs (also supports the older format of run recovery ideas although these are obsolete now)
  - latest_run_file: to instead provide a path to a file containing the id of your latest run, for retrieval.
  - config_path: to specify a config.json file in which your workspace settings are defined

You can extend this list of arguments by creating a child class that inherits from AMLRunScriptConfig.

### Defining your own argument types

Additional arguments can have any of the following types: `bool`, `integer`, `float`, `string`, `list`, `class/class instance`
with no additional work required. You can also define your own custom type, by providing a custom class in your code that
inherits from `CustomTypeParam`. It must define 2 methods:
1. `_validate(self, x: Any)`: which should raise a `ValueError` if x is not of the type you expect, and should also make a call
`super()._validate(val)`
2. `from_string(self, y: string)` which takes in the command line arg as a string (`y`) and returns an instance of the type
that you want. For example, if your custom type is a tuple, this method should create a tuple from the input string and return that.
An example of a custom type can be seen in our own custom type: `RunIdOrListParam`, which accepts a string representing one or more
run ids (or run recovery ids) and returns either a List or a single RunId object (or RunRecoveryId object if appropriate)

### Example:

```python
class EvenNumberParam(util.CustomTypeParam):
    """ Our custom type param for even numbers """

    def _validate(self, val: Any) -> None:
        if (not self.allow_None) and val is None:
            raise ValueError("Value must not be None")
        if val % 2 != 0:
            raise ValueError(f"{val} is not an even number")
        super()._validate(val)  # type: ignore

    def from_string(self, x: str) -> int:
        return int(x)


class MyScriptConfig(util.AmlRunScriptConfig):
    # example of a generic param
    simple_string: str = param.String(default="")
    # example of a custom param
    even_number = EvenNumberParam(2, doc="your choice of even number")

```
