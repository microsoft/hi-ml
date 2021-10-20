# Performance Diagnostics

The `hi-ml` toolbox offers several components to integrate with PyTorch Lightning based training workflows:

* The `AzureMLProgressBar` is a replacement for the default progress bar that the Lightning Trainer uses. Its output is
  more suitable for display in an offline setup like AzureML.
* The `BatchTimeCallback` can be added to the trainer to detect performance issues with data loading.

## `AzureMLProgressBar`

The standard PyTorch Lightning is well suited for interactive training sessions on a GPU machine, but its output can get
confusing when run inside AzureML. The `AzureMLProgressBar` class can replace the standard progress bar, and optionally
adds timestamps to each progress event. This makes it easier to later correlate training progress with, for example, low
GPU utilization showing in AzureML's GPU monitoring.

Here's a code snippet to add the progress bar to a PyTorch Lightning Trainer object:

```python
from health_ml.utils import AzureMLProgressBar
from pytorch_lightning import Trainer

progress = AzureMLProgressBar(refresh_rate=100, print_timestamp=True)
trainer = Trainer(callbacks=[progress])
```

This produces progress information like this:
```
2021-10-20T06:06:07Z Training epoch 18 (step 94):    5/5 (100%) completed. 00:00 elapsed, total epoch time ~ 00:00
2021-10-20T06:06:07Z Validation epoch 18:    2/2 (100%) completed. 00:00 elapsed, total epoch time ~ 00:00
2021-10-20T06:06:07Z Training epoch 19 (step 99):    5/5 (100%) completed. 00:00 elapsed, total epoch time ~ 00:00
...
```


## `BatchTimeCallback`

This callback can help diagnose issues with low performance of data loading. It captures the time between the end of a
training or validation step, and the start of the next step. This is often indicative of the time it takes to retrieve
the next batch of data: When the data loaders are not performant enough, this time increases.

The `BatchTimeCallback` will detect minibatches where the estimated data loading time is too high, and print alerts.
These alerts will be printed at most 5 times per epoch, for a maximum of 3 epochs, to avoid cluttering the output.

Note that it is common for the first minibatch of data in an epoch to take a long time to load, because data loader
processes need to spin up.

The callback will log a set of metrics:

* `timing/train/batch_time [sec] avg` and `timing/train/batch_time [sec] max`: Average and maximum time that it takes
  for batches to train/validate
* `timing/train/batch_loading_over_threshold [sec]` is the total time wasted per epoch in waiting for the next batch of
  data. This is computed by looking at all batches where the batch loading time was over the threshold
  `max_batch_load_time_seconds` (that is set in the constructor of the callback), and totalling the batch loading time
  for those batches.
* `timing/train/epoch_time [sec]` is the time for an epoch to complete.

### Caveats

* In distributed training, the performance metrics will be collected at rank 0 only.
* The time between the end of a batch and the start of the next batch is also impacted by other callbacks. If you have
  callbacks that are particularly expensive to run, for example because they actually have their own model training, the
  results of the `BatchTimeCallback` may be misleading.

### Usage example

```python
from health_ml.utils import BatchTimeCallback
from pytorch_lightning import Trainer

batchtime = BatchTimeCallback(max_batch_load_time_seconds=0.5)
trainer = Trainer(callbacks=[batchtime])
```

This would produce output like this:
```
Epoch 18 training: Loaded the first minibatch of data in 0.00 sec.
Epoch 18 validation: Loaded the first minibatch of data in 0.00 sec.
Epoch 18 training took 0.02sec, of which waiting for data took 0.01 sec total.
Epoch 18 validation took 0.00sec, of which waiting for data took 0.00 sec total.
```
