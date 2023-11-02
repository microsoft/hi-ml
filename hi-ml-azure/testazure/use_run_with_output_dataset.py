import os
from pathlib import Path
import sys

from azure.ai.ml import MLClient
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential

src_root = Path(__file__).parents[1] / "src"
sys.path.append(str(src_root))

from health_azure.himl import submit_to_azure_if_needed
from health_azure.utils import get_ml_client, get_workspace, get_credential
from azure.storage.blob import BlobServiceClient


def main() -> None:
    # Check out if we can get the credential
    credential = AzureMLOnBehalfOfCredential()
    try:
        credential.get_token("https://management.azure.com/.default")
    except Exception:
        print("Failed to get the credential")
    uri = os.environ["MLFLOW_TRACKING_URI"]
    uri_segments = uri.split("/")
    subscription_id = uri_segments[uri_segments.index("subscriptions") + 1]
    resource_group_name = uri_segments[uri_segments.index("resourceGroups") + 1]
    workspace_name = uri_segments[uri_segments.index("workspaces") + 1]
    credential = AzureMLOnBehalfOfCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group_name,
        workspace_name=workspace_name,
    )
    print("Got the client")

    run_id = "sincere_yacht_xjz95gwvq8"
    workspace = get_workspace()
    run = workspace.get_run(run_id)
    if hasattr(run, "output_datasets"):
        print(run.output_datasets)
    else:
        print("No output datasets")

    job = ml_client.jobs.get(run_id)
    output_dataset = job.outputs["OUTPUT_0"]

    datastore = ml_client.datastores.get("workspaceblobstore")
    print(datastore.account_name)
    print(datastore.container_name)
    account_url = f"{datastore.protocol}://{datastore.account_name}.blob.{datastore.endpoint}"
    print(f"{output_dataset.path}")

    blob_client = BlobServiceClient(account_url=account_url, credential=credential)
    container_client = blob_client.get_container_client(datastore.container_name)

    # List all blobs (files) inside a specific folder (prefix)
    paths_parts = output_dataset.path.split("/paths/")
    assert len(paths_parts) == 2
    folder_name = paths_parts[1]
    blob_list = [blob.name for blob in container_client.list_blobs(name_starts_with=folder_name)]
    print(f"Files in {folder_name}:")
    for blob_name in blob_list:
        print(blob_name)

    # Get the client without further authentication.
    ml_client2 = get_ml_client()


if __name__ == "__main__":
    submit_to_azure_if_needed(
        snapshot_root_directory=Path(__file__).parents[2],
        compute_cluster_name="lite-testing-ds2",
        strictly_aml_v1=True,
        submit_to_azureml=True,
    )
    main()
