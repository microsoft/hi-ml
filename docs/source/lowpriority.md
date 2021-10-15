# Using Cheap Low Priority VMs

By using Low Priority machines in AzureML, we can run training at greatly reduced costs (around 20% of the original
price). This comes with the risk, though, of having the job interrupted and later re-started. This document describes
the inner workings of Low Priority compute, and how to best make use of it.

Because the jobs can get interrupted, low priority machines are not suitable for production workload where time is
critical. They do offer a lot of benefits though for long-running training jobs, that would otherwise be expensive to
carry out.

## Setting up the Compute Cluster

Jobs in Azure Machine Learning run in a "compute cluster". When creating a compute cluster, we can specify the size of
the VM, the type and number of GPUs, etc. Doing this via the AzureML UI is described
[here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-attach-compute-studio#amlcompute). Doing it
programmatically is described
[here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-attach-compute-cluster?tabs=python)

One of the setting to tweak when creating the compute cluster is whether the machines are "Dedicated" or "Low Priority":

* Dedicated machines will be permanently allocated to your compute cluster. The VMs in a dedicated cluster will be
  always available, unless the cluster is set up in a way that it removes idle machine. Jobs will not be interrupted.
* Low priority machines effectively make use of spare capacity in the data centers, you can think of them as
  "dedicated machines that are presently idle". They are available at a much lower price (around 20% of the price of a
  dedicated machine). These machines are made available to you until they are needed as dedicated machines somewhere
  else.

In order to get a compute cluster that operates at the lowest price point, choose
* Low priority machines
* Set "Minimum number of nodes" to 0, so that the cluster removes all idle machines if no jobs are running.

## Behaviour of Low Priority VMs

Jobs can be interrupted at any point, this is called "low priority preemption". When interrupted, the job stops - there
is no signal that we can make use of to do cleanup or something. All the files that the job has produced up to that
point will be saved to the cloud.

At some later point, the job will be assigned a virtual machine again. When re-started, all the files that the job had
produced in its previous run will be available on disk.

Note that all AzureML-internal log files that the job produced in a previous run will be overwritten (this behaviour may
change in the future). The metrics that were written to AzureML (via `Run.log`, for example) will be available when the
job restarts. The re-started job will append to the metrics written in the previous run. This typically leads to sudden
jumps in metrics, as illustrated here:
![lowpriority_interrupted_lr.png](lowpriority_interrupted_lr.png)

How do you verify that your job got interrupted? Usually, you would see a warning displayed on the job page in the
AzureML UI, that says something along the lines of "Low priority compute preemption warning: a node has been preempted."
. You can use kinks in metrics as another indicator that your job got preempted: Sudden jumps in metrics after which the
metric follows a shape similar to the one at job start usually indicates low priority preemption.

Note that a job can be interrupted more than one time.

## Best Practice Guide for Your Jobs

In order to make best use of low priority compute, your code needs to be made resilient to restarts. Essentially, this
means that it should write regular checkpoints, and try to use those checkpoint files if they already exist. Examples of
how to best do that are given below.

In addition, you need to bear in mind that the job can be interrupted at any moment, for example when it is busy
uploading huge checkpoint files to Azure. When trying to upload again after restart, there can be resource collisions.

### Writing and Using Recovery Checkpoints

When using PyTorch Lightning, you can add a checkpoint callback to your trainer, that ensures that you save the model
and optimizer to disk in regular intervals. This callback needs to be added to your `Trainer` object. Note that these
recovery checkpoints need to be written to the "outputs" folder, because only files in this folder get saved to Azure
automatically when the job gets interrupted.

When starting training, your code needs to check if there is already a recovery checkpoint present on disk. If so,
training should resume from that point.

Here is a code snippet that illustrates all that:

```python
import re
from pathlib import Path
import numpy as np
from health_ml.utils import AzureMLLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

RECOVERY_CHECKPOINT_FILE_NAME = "recovery_"
CHECKPOINT_FOLDER = "outputs/checkpoints"


class RecoveryCheckpointCallback(ModelCheckpoint):
    """
    This callback is used to save recovery checkpoints every 10 epochs. It ensures that there is a logged 
    quantity to monitor if we want to log more then one recent checkpoint.
    """

    def __init__(self):
        super().__init__(dirpath=CHECKPOINT_FOLDER,
                         monitor="epoch_started",
                         filename=RECOVERY_CHECKPOINT_FILE_NAME + "{epoch}",
                         period=10,
                         save_top_k=1,
                         mode="max",
                         save_last=False)

    def on_train_epoch_start(self, trainer, pl_module, unused: bool = None) -> None:
        # The metric to monitor must be logged on all ranks in distributed training
        pl_module.log("epoch_started", trainer.current_epoch, on_epoch=True, on_step=False, sync_dist=False)


def get_latest_recovery_checkpoint():
    all_recovery_files = [f for f in Path(CHECKPOINT_FOLDER).glob(RECOVERY_CHECKPOINT_FILE_NAME + "*")]
    if len(all_recovery_files) == 0:
        return None
    recovery_epochs = [int(re.findall(r"[\d]+", f.stem)[0]) for f in all_recovery_files]
    idx_max_epoch = int(np.argmax(recovery_epochs))
    return str(all_recovery_files[idx_max_epoch])


trainer = Trainer(default_root_dir="outputs",
                  callbacks=[RecoveryCheckpointCallback()],
                  logger=[AzureMLLogger()],
                  resume_from_checkpoint=get_latest_recovery_checkpoint())
```
