#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path

from azureml.core import Datastore

from health_azure.datasets import get_datastore
from health_azure import get_workspace


def main() -> None:
    path = Path(__file__).parent.resolve()

    workspace = get_workspace()

    datastore: Datastore = get_datastore(workspace=workspace,
                                         datastore_name="himldatasets")

    # Either download all outputs:
    # run.download_files(prefix="outputs", output_directory=str(path))
    # Or download 1:
    outputs_root = path / "outputs"
    outputs_root.mkdir(exist_ok=False)

    downloaded = datastore.download(
        target_path=str(outputs_root),
        prefix="himl_sample4_output/primes.txt",
        overwrite=True,
        show_progress=True)
    assert downloaded == 1


if __name__ == "__main__":
    main()
