from pathlib import Path

from torchvision.datasets import MNIST

from health.azure.aml import submit_to_azure_if_needed


def main() -> None:
    # Data on local disk: We expect that the data lives under the root folder
    run_info = submit_to_azure_if_needed(root_folder=".")
    file_contents = Path("dataset.csv").read_text()
    assert file_contents == "some_contents"

    # Create an example where people download the MNIST data
    run_info = submit_to_azure_if_needed(root_folder=".")
    mnist_folder = MNIST(download=True)
    file_contents = Path("mnist.tar.gz").binary()
    assert file_contents is not None

    # Data lives in Azure blob storage already, needs to be mounted or downloaded when running in AML.
    run_info = submit_to_azure_if_needed(input_datasets="foo")
    # For this to work, run_info must always be provided, even in local execution
    dataset_folder = run_info.input_datasets[0] or "Z:/datasets/foo"

    file_contents = (dataset_folder / "dataset.csv").read_text()
    assert file_contents == "some_contents"


if __name__ == '__main__':
    main()
