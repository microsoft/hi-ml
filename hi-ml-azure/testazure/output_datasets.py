from datetime import datetime
from pathlib import Path
import sys


src_root = Path(__file__).parents[1] / "src"
sys.path.append(str(src_root))

from health_azure import submit_to_azure_if_needed, DatasetConfig


def main():
    # Define the output dataset
    output_dataset = DatasetConfig(
        name='output_dataset',
        datastore='workspaceblobstore',
        # path=f"outputs_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}" # Plus a random part
    )

    # Submit the script to Azure if needed
    run_info = submit_to_azure_if_needed(
        snapshot_root_directory=Path(__file__).parents[1],
        output_datasets=[output_dataset],
        compute_cluster_name="lite-testing-ds2",
        submit_to_azureml=True,
        strictly_aml_v1=True,
    )

    output_folder = run_info.output_datasets[0]
    print(f"Output folder: {output_folder}")
    output_file = output_folder / "output.txt"
    output_file.write_text('Hello, world!')

    print("Done!")


if __name__ == "__main__":
    main()
