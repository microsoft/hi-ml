from health_azure.logging import logging_section
from pytorch_lightning import LightningDataModule
from health_ml.ml_runner import RunnerBase


class EvalRunner(RunnerBase):
    def run(self) -> None:
        self.init_inference()

        with logging_section("Model inference"):
            self.run_inference()

    def get_data_module(self) -> LightningDataModule:
        return self.container.get_eval_data_module()
