# Debugging and Profiling

While using the hi-ml toolbox modules, you might encounter some errors that require running the code step by step in
order to track down their sources. Here's some guidelines to help you debug and/or profile training pipelines.

## Debugging within VS Code

VS code has a great [Debugging Support](https://code.visualstudio.com/docs/editor/debugging) for many programming
languages. Make sure to install and enable the [Python Extension for VS Code](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
to debug hi-ml toolbox modules built in Python.

### Debugging configs

The hi-ml repository is organised as
[Multi-root workspaces](https://code.visualstudio.com/docs/editor/workspaces#_multiroot-workspaces) to account
for environment differences among the modules and offer flexibility to configure each module seperatly.

We provide a set of custom debugging configs for each of hi-ml modules:
[hi-ml](https://github.com/microsoft/hi-ml/tree/main/hi-ml/.vscode/lanch.json),
[hi-ml-azure](https://github.com/microsoft/hi-ml/tree/main/hi-ml-azure/.vscode/lanch.json),
[hi-ml-histopathology](https://github.com/microsoft/hi-ml/tree/main/hi-ml-histopathology/.vscode/lanch.json), and
[multimodal](https://github.com/microsoft/hi-ml/tree/main/multimodal/.vscode/lanch.json)

Vs Code restricts debugging to user-written code only by default. If you want to step through external code and
standard libraries functions, set `"justMyCode": false` inside the debugging config block in the `launch.json` file.
In particular, if you would like to debug the current file while breaking through `pytorch` code, navigate to
`himl-projects.code-workspace` in the repo root and edit the "launch" block as follow:

```json
"launch": {
    "configurations": [
      {
        "name": "Python: Current File",
        "type": "python",
        "request": "launch",
        "program": "${file}",
        "console": "integratedTerminal",
        "justMyCode": false
      },
    ],
}
```

### Pytorch Lightning flags for debugging and quick runs

The hi-ml toolbox is built upon [Pytorch Lightning (PL)](https://www.pytorchlightning.ai/) to help you build scalable
deep learning models for healtcare and life sciences. Refer to [Running ML Experiments with hi-ml](runner.md) for
detailed instructions on how to build scalable pipelines within hi-ml.

Whether you're building a brand new model, or extending an existing one, you might want to make sure that your
code runs as expected locally before submitting a job to AzureML. hi-ml supports a set of debugging flags that triggers
[Pytorch Lightning Trainer](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer) arguments to
help you detect any potential errors or bugs at early stage.

These are available as part of the
[TrainerParams](https://github.com/microsoft/hi-ml/blob/96b8ba60ebf84416f5c9b13e2df15ee07a13e6bb/hi-ml/src/health_ml/deep_learning_config.py#L357)
and can be used as extra command line arguments with the [hi-ml
runner](https://github.com/microsoft/hi-ml/blob/746c8b58c1af71f71eeaaac2a8584be1d9a5386f/hi-ml/src/health_ml/runner.py#L107).

* `pl-fast-dev-run`: If set to `n`, runs the pipeline for only `n` batch(es) of train, val and test for only a single
  epoch. Additionally [this flag](https://pytorch-lightning.readthedocs.io/en/stable/common/debugging.html#fast-dev-run)
  disables all callbacks and hyperparameters serialization which makes the debugging process very quick. This must be
  used for debugging purposes only.
* `pl-limit-train-batches`: Limits the training dataset to the given number of batches `n`.
* `pl-limit-val-batches`: Limits the validation dataset to the given number of batches `n`.
* `pl-limit-train-batches`: Limits the test dataset to the given number of batches `n`.

In general, it is very useful to run the following two steps as part of the developement cycle. Let's take the example
of `SlidesPandaImageNetMIL` model:

1. Make sure all training, validation and test loops complete properly:

```shell
conda activate HimlHisto
python ../hi-ml/src/health_ml/runner.py --model histopathology.SlidesPandaImageNetMIL --crossval-count=0 --bach-size=2 --pl-fast-dev-run=4
```

2. Make sure the whole pipeline runs properly, including checkpoints callbacks and hyperparameter serialization:

```shell
conda activate HimlHisto
python ../hi-ml/src/health_ml/runner.py --model histopathology.SlidesPandaImageNetMIL --crossval-count=0 --bach-size=2 --pl-limit-train-batches=4 --pl-limit-val-batches=4 --pl-limit-test-batches=4 --max_epochs=4
```

Note: Under the hood, setting `pl-fast-dev-run=n` overrides
`pl-limit-train-batches=n`, `pl-limit-val-batches=n`, `pl-limit-train-batches=n`, `max_epochs=1` and disables all
callbacks. Please keep in mind that all the above is useful for efficient and quick debugging purposes only and is in no
way a performance indicator.

## Profiling Machine Learning Pipelines

Pytorch Lightning supports a set of built-in
[profilers](https://pytorch-lightning.readthedocs.io/en/stable/advanced/profiler.html) that help you identify
bottlenecks in your code during training, testing and inference. You can trigger code profiling through the command line
argument `--pl_profiler` that you can set to either
[`simple`](https://pytorch-lightning.readthedocs.io/en/stable/advanced/profiler.html#simple-profiler),
[`advanced`](https://pytorch-lightning.readthedocs.io/en/stable/advanced/profiler.html#simple-profiler), or
[`pytorch`](https://pytorch-lightning.readthedocs.io/en/stable/advanced/profiler.html#pytorch-profiler).

The profiler outputs will be saved in a subfolder `profiler` inside the outputs folder of the run.

```shell
conda activate HimlHisto
python ../hi-ml/src/health_ml/runner.py --model histopathology.SlidesPandaImageNetMIL --crossval-count=0 --bach-size=2 --pl-limit-train-batches=4 --pl-limit-val-batches=4 --pl-limit-test-batches=4 --max_epochs=4 --pl-profiler=pytorch
```

### Interpret Pytorch Profiling outputs via Tensorboard

[Pytorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) can effectively be interpreted via
the TensorBoard dashbord interface that is integrated in [VS
Code](https://code.visualstudio.com/docs/datascience/pytorch-support#_tensorboard-integration) as part of the Python
extension. Once you have the outputs of the Pytorch Profiler in `outputs/YYYY-MM-DDTHHmmssZ_YourContainerName`, you can
open the TensorBoard Profiler plugin by launching the Command Palette using the keyboard shortcut CTRL + SHIFT + P (CMD
+ SHIFT + P on a Mac) and typing the “Launch TensorBoard” command.



### Memory profiling with PytorchProfiler

In some scenarios

### Advanced profiling arguments

You can specify additional profiling arguments t
