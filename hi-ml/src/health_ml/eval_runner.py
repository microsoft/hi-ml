from pytorch_lightning import LightningDataModule

from health_azure.logging import logging_section
from health_ml.runner_base import RunnerBase


class EvalRunner(RunnerBase):
    """A class to run the evaluation of a model on a new dataset. The initialization logic is taken from the base
    class `RunnerBase`.
    """

    def validate(self) -> None:
        """Checks if the fields of the class are set up correctly."""
        if self.container.src_checkpoint is None or self.container.src_checkpoint.checkpoint == "":
            raise ValueError(
                "To use model evaluation, you need to provide a checkpoint to use, via the --src_checkpoint argument."
            )

    def run(self) -> None:
        """Start the core workflow that the class implements: Initialize a PL Trainer object and use that to run
        inference on the inference dataset."""
        self.container.outputs_folder.mkdir(exist_ok=True, parents=True)
        self.init_inference()
        with logging_section("Model inference"):
            self.run_inference()

    def get_data_module(self) -> LightningDataModule:
        """Reads the evaluation data module from the underlying container."""
        return self.container.get_eval_data_module()
