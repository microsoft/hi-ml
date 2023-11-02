from pathlib import Path
import sys

src_root = Path(__file__).parents[1] / "src"
sys.path.append(str(src_root))

from health_azure.utils import get_ml_client, get_workspace

run_id = "sincere_yacht_xjz95gwvq8"
workspace = get_workspace()
run = workspace.get_run(run_id)


ml_client = get_ml_client()
job = ml_client.jobs.get(run_id)
output_dataset = job.outputs["OUTPUT_0"]

from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

data_type = AssetTypes.URI_FILE

data = Data(path=output_dataset.path)
# data.mount(ml_client)

datastore = ml_client.datastores.get("workspaceblobstore")
print(datastore.account_name)
print(datastore.container_name)
account_url = f"{datastore.protocol}://{datastore.account_name}.blob.{datastore.endpoint}"
print(f"{output_dataset.path}")
"azureml://subscriptions/a85ceddd-892e-4637-ae4b-67d15ddf5f2b/resourcegroups/health-ml/workspaces/hi-ml/datastores/workspaceblobstore/paths/output_dataset/"

from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential

blob_client = BlobServiceClient(account_url=account_url, credential=DefaultAzureCredential())
container_client = blob_client.get_container_client(datastore.container_name)

# List all blobs (files) inside a specific folder (prefix)
paths_parts = output_dataset.path.split("/paths/")
assert len(paths_parts) == 2
folder_name = paths_parts[1]
blob_list = [blob.name for blob in container_client.list_blobs(name_starts_with=folder_name)]
