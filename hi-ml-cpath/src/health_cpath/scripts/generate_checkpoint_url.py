from azure.storage.blob import generate_blob_sas, BlobSasPermissions
from azureml.core import Workspace
from datetime import datetime, timedelta
from health_azure import get_workspace
from health_ml.utils.common_utils import DEFAULT_AML_CHECKPOINT_DIR
from pathlib import Path
from typing import Optional


def get_checkpoint_url_from_aml_run(
    run_id: str,
    checkpoint_filename: str,
    account_key: str,
    expiry_days: int = 1,
    workspace_config_path: Optional[Path] = None,
    aml_workspace: Optional[Workspace] = None,
    sas_token: Optional[str] = None,
) -> str:
    """Generate a SAS URL for the checkpoint file in the given run.

    :param run_id: The run ID of the checkpoint.
    :param checkpoint_filename: The filename of the checkpoint.
    :param account_key: The Azure Storage account key to use for the SAS token.
    :param expiry_days: The number of days the SAS URL is valid for, defaults to 30.
    :param workspace_config_path: The path to the workspace config file, defaults to None.
    :param aml_workspace: The Azure ML workspace to use, defaults to the default workspace.
    :param sas_token: The SAS token to use, defaults to None.
    :return: The SAS URL for the checkpoint.
    """
    workspace = get_workspace(aml_workspace=aml_workspace, workspace_config_path=workspace_config_path)
    account_name = workspace.get_details()['storageAccount'].split('/')[-1]
    print(f"Workspace {workspace.name} stores its run results in storage account {account_name}.")
    container_name = 'azureml'
    blob_name = f'ExperimentRun/dcid.{run_id}/{DEFAULT_AML_CHECKPOINT_DIR}/{checkpoint_filename}'
    if not sas_token:
        sas_token = generate_blob_sas(
            account_name=account_name,
            container_name=container_name,
            blob_name=blob_name,
            account_key=account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(days=expiry_days),
        )

    return f'https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas_token}'


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=str, help='The run id of the model checkpoint')
    parser.add_argument('--workspace_config', type=str, help='The path to the workspace config file.')
    parser.add_argument(
        '--checkpoint_filename',
        type=str,
        default='last.ckpt',
        help='The filename of the model checkpoint. Default: last.ckpt',
    )
    parser.add_argument(
        '--expiry_days',
        type=int,
        default=30,
        help='The number of hours for which the SAS token is valid. Default: 30 for 1 month',
    )
    parser.add_argument(
        '--account_key', default='', type=str, help='The Azure Storage account key to use for the SAS token.'
    )
    args = parser.parse_args()
    workspace_config_path = Path(args.workspace_config) if args.workspace_config else None
    url = get_checkpoint_url_from_aml_run(
        run_id=args.run_id,
        checkpoint_filename=args.checkpoint_filename,
        expiry_days=args.expiry_days,
        workspace_config_path=workspace_config_path,
        account_key=args.account_key,
    )
    print(f'Checkpoint URL: {url}')
