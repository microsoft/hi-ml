# Distributed GPU Training

Calling `submit_to_azure_if_needed` with `num_nodes>1` will prepare the AzureML run_configuration for distributed training jobs using IntelMpi ([https://docs.microsoft.com/en-us/azure/machine-learning/how-to-train-distributed-gpu](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-train-distributed-gpu)) with `process_count_per_node=1` (the default, for per-node launch) and `node_count=num_nodes`.

After calling `submit_to_azure_if_needed` with `num_nodes>1`, then call: `set_environment_variables_for_multi_node()` to ensure the environment variables for Mpi are set correctly.

## PyTorch Lightning

If training with [PyTorch Lightning](https://www.pytorchlightning.ai/) then the [DDPPlugin](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.plugins.training_type.DDPPlugin.html) needs to be created with the same number of nodes before passing to the [Trainer](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html), for example:

```python
    num_gpus = torch.cuda.device_count()
    effective_num_gpus = num_gpus * num_nodes

    if effective_num_gpus > 1:
        accelerator: Optional[str] = "ddp"
        plugins = [DDPPlugin(num_nodes=num_nodes,
                             sync_batchnorm=True,
                             find_unused_parameters=False)]
    else:
        accelerator = None
        plugins = []

    trainer = Trainer(accelerator=accelerator,
                      plugins=plugins,
                      num_nodes=num_nodes,
                      gpus=num_gpus,
                      ...)
```
