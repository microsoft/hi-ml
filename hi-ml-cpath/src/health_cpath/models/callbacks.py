import os
from pathlib import Path
import torch
import numpy as np
import pandas as pd

from typing import Dict, List
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback

from health_cpath.models.deepmil import BaseDeepMILModule
from health_cpath.utils.naming import ResultsKey
from health_cpath.utils.tiles_selection_utils import TilesSelector

TILES_JOIN_TOKEN = "$"
LOSS_VALUES_SUBFOLDER = "loss_values"
LOSS_VALUES_FRILENAME = "epoch_{}.csv"


class LossValuesAnalysisCallback(Callback):
    def __init__(self, patience: int = 0, save_every_n_epoch: int = 1) -> None:
        self.patience = patience
        self.loss_cache = self.reset_loss_cache()
        self.save_every_n_epoch = save_every_n_epoch

    def reset_loss_cache(self) -> Dict[str, List]:
        return {ResultsKey.LOSS: [], ResultsKey.SLIDE_ID: [], ResultsKey.TILE_ID: []}

    def cache_loss_values(self, trainer: Trainer) -> bool:
        return (trainer.current_epoch - self.patience) % self.save_every_n_epoch == 0

    def dump_loss_cache(self, outputs_root: Path, current_epoch: int) -> None:
        loss_cache_df = pd.DataFrame(self.loss_cache)
        loss_cache_df = loss_cache_df.sort_values(by=ResultsKey.LOSS, ascending=False)
        loss_val_analysis_path = outputs_root / LOSS_VALUES_SUBFOLDER
        os.makedirs(loss_val_analysis_path, exist_ok=True)
        loss_cache_df.to_csv(loss_val_analysis_path / LOSS_VALUES_FRILENAME.format(current_epoch), index=False)

    def gather_loss_cache(self) -> None:
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            if world_size > 1:
                self.loss_cache = TilesSelector._gather_dictionaries(world_size, self.loss_cache)

    @torch.no_grad()
    def on_train_batch_end(  # type: ignore
        self,
        trainer: Trainer,
        pl_module: BaseDeepMILModule,
        outputs: torch.Tensor,
        batch: Dict,
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        if self.cache_loss_values(trainer):
            bag_logits, bag_labels, _ = pl_module.compute_bag_labels_logits_and_attn_maps(batch)
            if pl_module.n_classes > 1:
                loss = pl_module.loss_fn_no_reduction(bag_logits, bag_labels.long())
            else:
                loss = pl_module.loss_fn_no_reduction(bag_logits.squeeze(1), bag_labels.float())
            self.loss_cache[ResultsKey.LOSS].extend(loss.tolist())
            self.loss_cache[ResultsKey.SLIDE_ID].extend(np.array(batch[ResultsKey.SLIDE_ID])[:, 0])
            self.loss_cache[ResultsKey.TILE_ID].extend(
                [TILES_JOIN_TOKEN.join(tiles) for tiles in batch[ResultsKey.TILE_ID]]
            )

    def on_train_epoch_end(self, trainer: Trainer, pl_module: BaseDeepMILModule) -> None:  # type: ignore
        if self.cache_loss_values(trainer):
            assert pl_module.outputs_handler is not None
            self.gather_loss_cache()
            if pl_module.global_rank == 0:
                self.dump_loss_cache(pl_module.outputs_handler.outputs_root, trainer.current_epoch)
        self.loss_cache = self.reset_loss_cache()

    def on_train_end(self, trainer: Trainer, pl_module: BaseDeepMILModule) -> None:  # type: ignore
        assert pl_module.outputs_handler is not None
        loss_val_analysis_path = pl_module.outputs_handler.outputs_root / LOSS_VALUES_SUBFOLDER
        epochs = list(range(self.patience, trainer.max_epochs + 1, self.save_every_n_epoch))
        for epoch in epochs:
            loss_values = pd.read_csv(loss_val_analysis_path / LOSS_VALUES_FRILENAME.format(epoch))
