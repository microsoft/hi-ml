# Logging metrics when training models in AzureML

This section describes the basics of logging to AzureML, and how this can be simplified when using PyTorch Lightning.
It also describes helper functions to make logging more consistent across your code.

## Basics
The mechanics of writing metrics to an ML training run inside of AzureML are described 
[here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-log-view-metrics).

Using the `hi-ml-azure` toolbox, you can simplify that like this:
```python
from health_azure import RUN_CONTEXT
...
RUN_CONTEXT.log(name="name_of_the_metric", value=my_tensor.item())
```
Similarly you can log strings (via the `log_text` method) or figures (via the `log_image` method), see the 
[documentation](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-log-view-metrics).

## Using PyTorch Lightning
The `hi-ml` toolbox relies on `pytorch-lightning` for a lot of its functionality. 
Logging of metrics is described in detail 
[here](https://pytorch-lightning.readthedocs.io/en/latest/extensions/logging.html)

`hi-ml` provides a Lightning-ready logger object to use with AzureML. You can add that to your trainer as you would
add a Tensorboard logger, and afterwards see all metrics in both your Tensorboard files and in the AzureML UI. 
This logger can be added to the `Trainer` object as follows:
```python
from health_ml.utils import AzureMLLogger
from pytorch_lightning.loggers import TensorBoardLogger
tb_logger = TensorBoardLogger("logs/")
azureml_logger = AzureMLLogger()
trainer = Trainer(logger=[tb_logger, azureml_logger])
```
You do not need to make any changes to your logging code to write to both loggers at the same time. This means
that, if your code correctly writes to Tensorboard in a local run, you can expect the metrics to come out correctly
in the AzureML UI as well after adding the `AzureMLLogger`.

## Making logging consistent when training with PyTorch Lightning

A common problem of training scripts is that the calls to the logging methods tend to run out of sync.
The `.log` method of a `LightningModule` has a lot of arguments, some of which need to be set correctly when running
on multiple GPUs. 

To simplify that, there is a function `log_on_epoch` that turns synchronization across nodes on/off depending on the
number of GPUs, and always forces the metrics to be logged upon epoch completion. Use as follows:

```python
from health_ml.utils import log_on_epoch
from pytorch_lightning import LightningModule

class MyModule(LightningModule):
    def training_step(self, *args, **kwargs):
        ...
        loss = my_loss(y_pred, y)
        log_on_epoch(self, loss)
        return loss
```

### Logging learning rates

Logging learning rates is important for monitoring training, but again this can add overhead. To log learning rates
easily and consistently, we suggest either of two options:
* Add a `LearningRateMonitor` callback to your trainer, as described 
[here](https://pytorch-lightning.readthedocs.io/en/latest/extensions/generated/pytorch_lightning.callbacks.LearningRateMonitor.html#pytorch_lightning.callbacks.LearningRateMonitor)
* Use the `hi-ml` function `log_learning_rate`

The `log_learning_rate` function can be used at any point the training code, like this:
```python
from health_ml.utils import log_learning_rate
from pytorch_lightning import LightningModule

class MyModule(LightningModule):
    def training_step(self, *args, **kwargs):
        ...
        log_learning_rate(self, "learning_rate")
        loss = my_loss(y_pred, y)
        return loss
```
`log_learning_rate` will log values from all learning rate schedulers, and all learning rates if a scheduler
returns multiple values. In this example, the logged metric will be `learning_rate` if there is a single scheduler
that outputs a single LR, or `learning_rate/1/0` to indicate the value coming from scheduler index 1, value index 0. 
