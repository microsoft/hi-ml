# Checkpoint Utils

Hi-ml toolbox offers different utilities to parse and download pretrained checkpoints that help you abstract checkpoint
downloading from different sources. Refer to
[CheckpointParser](https://github.com/microsoft/hi-ml/blob/main/hi-ml/src/health_ml/utils/checkpoint_utils.py#L238) for
more details on the supported checkpoints format. Here's how you can use the checkpoint parser depending on the source:

- For a local path, simply pass it as shown below. The parser will further check if the provided path exists:

 ```python
from health_ml.utils.checpoint_utils import CheckpointParser

download_dir = 'outputs/checkpoints'
checkpoint_parser = CheckpointParser(checkpoint='local/path/to/my_checkpoint/model.ckpt')
print('Checkpoint', checkpoint_parser.checkpoint, 'is a local file', checkpoint_parser.is_local_file)
local_file = parser.get_or_download_checkpoint(download_dir)
```

- To download a checkpoint from a URL:

```python
from health_ml.utils.checpoint_utils import CheckpointParser, MODEL_WEIGHTS_DIR_NAME

download_dir = 'outputs/checkpoints'
checkpoint_parser = CheckpointParser('https://my_checkpoint_url.com/model.ckpt')
print('Checkpoint', checkpoint_parser.checkpoint, 'is a URL', checkpoint_parser.is_url)
# will dowload the checkpoint to download_dir/MODEL_WEIGHTS_DIR_NAME
path_to_ckpt = checkpoint_parser.get_or_download_checkpoint(download_dir)
```

- Finally checkpoints from an Azure ML runs can be reused by providing an id in this format
  `<AzureML_run_id>:<optional/custom/path/to/checkpoints/><filename.ckpt>`. If no custom path is provided (e.g.,
  `<AzureML_run_id>:<filename.ckpt>`) the checkpoint will be downloaded from the default checkpoint folder
  (e.g., `outputs/checkpoints`) If no filename is provided, (e.g., `src_checkpoint=<AzureML_run_id>`) the latest
  checkpoint will be downloaded (e.g., `last.ckpt`).

```python
from health_ml.utils.checpoint_utils import CheckpointParser

checkpoint_parser = CheckpointParser('AzureML_run_id:best.ckpt')
print('Checkpoint', checkpoint_parser.checkpoint, 'is a AML run', checkpoint_parser.is_aml_run_id)
path_azure_ml_ckpt = checkpoint_parser.get_or_download_checkpoint(download_dir)
```

If the Azure ML run is in a different workspace, a temporary SAS URL to download the checkpoint can be generated as follow:

```bash
cd hi-ml-cpath
python src/health_cpath/scripts/generate_checkpoint_url.py --run_id=AzureML_run_id:best_val_loss.ckpt --expiry_days=10
```

N.B: config.json should correspond to the original workspace where the AML run lives.

## Use cases

CheckpointParser is used to specify a `src_checkpoint` to [resume training from a given
checkpoint](https://github.com/microsoft/hi-ml/blob/main/docs/source/runner.md#L238),
or [run inference with a pretrained model](https://github.com/microsoft/hi-ml/blob/main/docs/source/runner.md#L215),
as well as
[ssl_checkpoint](https://github.com/microsoft/hi-ml/blob/main/hi-ml-cpath/src/health_cpath/utils/deepmil_utils.py#L62)
for computation pathology self supervised pretrained encoders.
