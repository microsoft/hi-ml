from health_azure.logging import logging_section
from pytorch_lightning import LightningDataModule
from health_ml.ml_runner import RunnerBase


class EvalRunner(RunnerBase):
    def validate(self) -> None:
        if self.container.src_checkpoint is None or self.container.src_checkpoint.checkpoint == "":
            raise ValueError(
                "To use model evaluation, you need to provide a checkpoint to use, via the --src_checkpoint argument."
            )

    def run(self) -> None:
        self.container.outputs_folder.mkdir(exist_ok=True, parents=True)
        self.init_inference()
        with logging_section("Model inference"):
            self.run_inference()

    def get_data_module(self) -> LightningDataModule:
        return self.container.get_eval_data_module()
