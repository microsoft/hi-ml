from pathlib import Path

from health.azure.himl import submit_to_azure_if_needed
from torchvision.datasets import MNIST


def main() -> None:
    # Data on local disk: We expect that the data lives under the root folder
    run_info = submit_to_azure_if_needed(root_folder=".")
    file_contents = Path("dataset.csv").read_text()
    assert file_contents == "some_contents"

    # Create an example where people download the MNIST data
    run_info = submit_to_azure_if_needed(root_folder=".")
    mnist_folder = MNIST(download=True)  # noqa: F841 assigned to but never used
    mnist_bytes = Path("mnist.tar.gz").read_bytes()
    assert mnist_bytes is not None

    # Data lives in Azure blob storage already, needs to be mounted or downloaded when running in AML.
    run_info = submit_to_azure_if_needed(input_datasets="foo")
    # For this to work, run_info must always be provided, even in local execution
    dataset_folder = run_info.input_datasets[0] or Path("Z:/datasets/foo")

    file_contents = (dataset_folder / "dataset.csv").read_text()
    assert file_contents == "some_contents"

    # Data lives in Azure blob storage, specified via
    input_datasets = ["foo",
                      DatasetConfig(name="bar", datastore="some_store", use_mounting=True)]
    run_info = submit_to_azure_if_needed(input_datasets=input_datasets)
    # For this to work, run_info must always be provided, even in local execution
    dataset_folder = run_info.input_datasets[0] or Path("Z:/datasets/foo")


if __name__ == '__main__':
    main()
