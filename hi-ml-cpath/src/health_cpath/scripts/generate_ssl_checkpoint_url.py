from azure.storage.blob import generate_blob_sas, BlobSasPermissions
from azureml.core import Workspace
from datetime import datetime, timedelta
from health_azure import get_workspace
from health_ml.utils.common_utils import DEFAULT_AML_CHECKPOINT_DIR
from typing import Optional


def get_ssl_checkpoint_url(
    run_id: str,
    checkpoint_filename: str,
    expiry_hours: int = 1,
    aml_workspace: Optional[Workspace] = None,
    sas_token: Optional[str] = None,
) -> str:
    """Generate a SAS URL for the SSL checkpoint file in the given run.

    :param run_id: The run ID of the SSL checkpoint.
    :param checkpoint_filename: The filename of the SSL checkpoint.
    :param expiry_hours: The number of hours the SAS URL is valid for, defaults to 1.
    :param aml_workspace: The Azure ML workspace to use, defaults to the default workspace.
    :param sas_token: The SAS token to use, defaults to None.
    :return: The SAS URL for the SSL checkpoint.
    """
    datastore = get_workspace(aml_workspace=aml_workspace).get_default_datastore()
    account_name = datastore.account_name
    container_name = 'azureml'
    blob_name = f'ExperimentRun/dcid.{run_id}/{DEFAULT_AML_CHECKPOINT_DIR}/{checkpoint_filename}'

    if not sas_token:
        sas_token = generate_blob_sas(account_name=datastore.account_name,
                                      container_name=container_name,
                                      blob_name=blob_name,
                                      account_key=datastore.account_key,
                                      permission=BlobSasPermissions(read=True),
                                      expiry=datetime.utcnow() + timedelta(hours=expiry_hours))

    return f'https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas_token}'


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=str, help='The run id of the SSL model checkpoint')
    parser.add_argument('--checkpoint_filename', type=str, default='last.ckpt',
                        help='The filename of the SSL model checkpoint. Default: last.ckpt')
    parser.add_argument('--expiry_hours', type=int, default=168,
                        help='The number of hours for which the SAS token is valid. Default: 168 for 1 week')
    args = parser.parse_args()
    args.run_id = '1f391509-f0a7-41d9-bef5-06713739fb0b'
    ssl_url = get_ssl_checkpoint_url(args.run_id, args.checkpoint_filename, args.expiry_hours)
    print(f'SSL checkpoint URL: {ssl_url}')
