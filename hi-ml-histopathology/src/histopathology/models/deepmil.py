#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging
from typing import Callable, Dict, List, Optional, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import Tensor, argmax, mode, nn, optim, round, set_grad_enabled
from torchmetrics import AUROC, F1, Accuracy, ConfusionMatrix, Precision, Recall

from health_azure.utils import is_global_rank_zero
from health_ml.utils import log_on_epoch
from histopathology.datasets.base_dataset import TilesDataset
from histopathology.models.encoders import TileEncoder
from histopathology.utils.naming import MetricsKey, ResultsKey
from histopathology.utils.output_utils import (BatchResultsType, DeepMILOutputsHandler, EpochResultsType,
                                               validate_class_names)

RESULTS_COLS = [ResultsKey.SLIDE_ID, ResultsKey.TILE_ID, ResultsKey.IMAGE_PATH, ResultsKey.PROB,
                ResultsKey.CLASS_PROBS, ResultsKey.PRED_LABEL, ResultsKey.TRUE_LABEL, ResultsKey.BAG_ATTN]


def _format_cuda_memory_stats() -> str:
    return (f"GPU {torch.cuda.current_device()} memory: "
            f"{torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB allocated, "
            f"{torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB reserved")


class BaseDeepMILModule(LightningModule):
    """Base class for deep multiple-instance learning"""

    def __init__(self,
                 label_column: str,
                 n_classes: int,
                 encoder: TileEncoder,
                 pooling_layer: Callable[[Tensor], Tuple[Tensor, Tensor]],
                 num_features: int,
                 dropout_rate: Optional[float] = None,
                 class_weights: Optional[Tensor] = None,
                 l_rate: float = 5e-4,
                 weight_decay: float = 1e-4,
                 adam_betas: Tuple[float, float] = (0.9, 0.99),
                 verbose: bool = False,
                 class_names: Optional[List[str]] = None,
                 is_finetune: bool = False,
                 outputs_handler: Optional[DeepMILOutputsHandler] = None) -> None:
        """
        :param label_column: Label key for input batch dictionary.
        :param n_classes: Number of output classes for MIL prediction. For binary classification, n_classes should be
         set to 1.
        :param encoder: The tile encoder to use for feature extraction. If no encoding is needed,
        you should use `IdentityEncoder`.
        :param pooling_layer: A pooling layer nn.module
        :param num_features: Dimensions of the input encoding features * attention dim outputs
        :param dropout_rate: Rate of pre-classifier dropout (0-1). `None` for no dropout (default).
        :param class_weights: Tensor containing class weights (default=None).
        :param l_rate: Optimiser learning rate.
        :param weight_decay: Weight decay parameter for L2 regularisation.
        :param adam_betas: Beta parameters for Adam optimiser.
        :param verbose: if True statements about memory usage are output at each step.
        :param class_names: The names of the classes if available (default=None).
        :param is_finetune: Boolean value to enable/disable finetuning (default=False).
        :param outputs_handler: A configured :py:class:`DeepMILOutputsHandler` object to save outputs for the best
            validation epoch and test stage. If omitted (default), no outputs will be saved to disk (aside from usual
            metrics logging).
        """
        super().__init__()

        # Dataset specific attributes
        self.label_column = label_column
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate
        self.class_weights = class_weights
        self.encoder = encoder
        self.aggregation_fn = pooling_layer
        self.num_pooling = num_features

        self.class_names = validate_class_names(class_names, self.n_classes)

        # Optimiser hyperparameters
        self.l_rate = l_rate
        self.weight_decay = weight_decay
        self.adam_betas = adam_betas

        self.save_hyperparameters()

        self.verbose = verbose

        # Finetuning attributes
        self.is_finetune = is_finetune

        self.outputs_handler = outputs_handler

        self.classifier_fn = self.get_classifier()
        self.loss_fn = self.get_loss()
        self.activation_fn = self.get_activation()

        # Metrics Objects
        self.train_metrics = self.get_metrics()
        self.val_metrics = self.get_metrics()
        self.test_metrics = self.get_metrics()

    def get_classifier(self) -> Callable:
        classifier_layer = nn.Linear(in_features=self.num_pooling,
                                     out_features=self.n_classes)
        if self.dropout_rate is None:
            return classifier_layer
        elif 0 <= self.dropout_rate < 1:
            return nn.Sequential(nn.Dropout(self.dropout_rate), classifier_layer)
        else:
            raise ValueError(f"Dropout rate should be in [0, 1), got {self.dropout_rate}")

    def get_loss(self) -> Callable:
        if self.n_classes > 1:
            if self.class_weights is None:
                return nn.CrossEntropyLoss()
            else:
                class_weights = self.class_weights.float()
                return nn.CrossEntropyLoss(weight=class_weights)
        else:
            pos_weight = None
            if self.class_weights is not None:
                pos_weight = Tensor([self.class_weights[1] / (self.class_weights[0] + 1e-5)])
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def get_activation(self) -> Callable:
        if self.n_classes > 1:
            return nn.Softmax()
        else:
            return nn.Sigmoid()

    @staticmethod
    def get_bag_label(labels: Tensor) -> Tensor:
        # Get bag (batch) labels as majority vote
        bag_label = mode(labels).values
        return bag_label.view(1)

    def get_metrics(self) -> nn.ModuleDict:
        if self.n_classes > 1:
            return nn.ModuleDict({MetricsKey.ACC: Accuracy(num_classes=self.n_classes),
                                  MetricsKey.ACC_MACRO: Accuracy(num_classes=self.n_classes, average='macro'),
                                  MetricsKey.ACC_WEIGHTED: Accuracy(num_classes=self.n_classes, average='weighted'),
                                  MetricsKey.AUROC: AUROC(num_classes=self.n_classes),
                                  MetricsKey.CONF_MATRIX: ConfusionMatrix(num_classes=self.n_classes)})
        else:
            threshold = 0.5
            return nn.ModuleDict({MetricsKey.ACC: Accuracy(threshold=threshold),
                                  MetricsKey.AUROC: AUROC(num_classes=self.n_classes),
                                  MetricsKey.PRECISION: Precision(threshold=threshold),
                                  MetricsKey.RECALL: Recall(threshold=threshold),
                                  MetricsKey.F1: F1(threshold=threshold),
                                  MetricsKey.CONF_MATRIX: ConfusionMatrix(num_classes=2, threshold=threshold)})

    def log_metrics(self,
                    stage: str) -> None:
        valid_stages = ['train', 'test', 'val']
        if stage not in valid_stages:
            raise Exception(f"Invalid stage. Chose one of {valid_stages}")
        for metric_name, metric_object in self.get_metrics_dict(stage).items():
            if metric_name == MetricsKey.CONF_MATRIX:
                metric_value = metric_object.compute()
                metric_value_n = metric_value / metric_value.sum(axis=1, keepdims=True)
                for i in range(metric_value_n.shape[0]):
                    log_on_epoch(self, f'{stage}/{self.class_names[i]}', metric_value_n[i, i])
            else:
                log_on_epoch(self, f'{stage}/{metric_name}', metric_object)

    def forward(self, instances: Tensor) -> Tuple[Tensor, Tensor]:  # type: ignore
        with set_grad_enabled(self.is_finetune):
            instance_features = self.encoder(instances)                    # N X L x 1 x 1
        attentions, bag_features = self.aggregation_fn(instance_features)  # K x N | K x L
        bag_features = bag_features.view(1, -1)
        bag_logit = self.classifier_fn(bag_features)
        return bag_logit, attentions

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.parameters(), lr=self.l_rate, weight_decay=self.weight_decay,
                          betas=self.adam_betas)

    def get_metrics_dict(self, stage: str) -> nn.ModuleDict:
        return getattr(self, f'{stage}_metrics')

    def _compute_bag_labels_logits_and_attn_maps(self, batch: Dict) -> Tuple[Tensor, Tensor]:
        """Compute bag labels, logits and attention maps depending on the nature of batch.
        Subclasses (TilesDeepMILModule and SlidesDeepMILModule) should specify how this is handled depending on if
        the tiles are fixed or generated on the fly from whole slide images.
        """
        raise NotImplementedError

    def _shared_step(self, batch: Dict, batch_idx: int, stage: str) -> BatchResultsType:
        bag_logits, bag_labels, bag_attn_list = self._compute_bag_labels_logits_and_attn_maps(batch)

        if self.n_classes > 1:
            loss = self.loss_fn(bag_logits, bag_labels.long())
        else:
            loss = self.loss_fn(bag_logits.squeeze(1), bag_labels.float())

        predicted_probs = self.activation_fn(bag_logits)
        if self.n_classes > 1:
            predicted_labels = argmax(predicted_probs, dim=1)
            probs_perclass = predicted_probs
        else:
            predicted_labels = round(predicted_probs)
            probs_perclass = Tensor([[1.0 - predicted_probs[i][0].item(), predicted_probs[i][0].item()]
                                     for i in range(len(predicted_probs))])

        loss = loss.view(-1, 1)
        predicted_labels = predicted_labels.view(-1, 1)
        batch_size = predicted_labels.shape[0]

        if self.n_classes == 1:
            predicted_probs = predicted_probs.squeeze(dim=1)

        bag_labels = bag_labels.view(-1, 1)

        results = dict()
        for metric_object in self.get_metrics_dict(stage).values():
            metric_object.update(predicted_probs, bag_labels.view(batch_size,))
        results.update({ResultsKey.SLIDE_ID: batch[TilesDataset.SLIDE_ID_COLUMN],
                        ResultsKey.TILE_ID: batch[TilesDataset.TILE_ID_COLUMN],
                        ResultsKey.IMAGE_PATH: batch[TilesDataset.PATH_COLUMN],
                        ResultsKey.LOSS: loss,
                        ResultsKey.PROB: predicted_probs,
                        ResultsKey.CLASS_PROBS: probs_perclass,
                        ResultsKey.PRED_LABEL: predicted_labels,
                        ResultsKey.TRUE_LABEL: bag_labels,
                        ResultsKey.BAG_ATTN: bag_attn_list,
                        ResultsKey.IMAGE: batch[TilesDataset.IMAGE_COLUMN]})

        if (TilesDataset.TILE_X_COLUMN in batch.keys()) and (TilesDataset.TILE_Y_COLUMN in batch.keys()):
            results.update({ResultsKey.TILE_X: batch[TilesDataset.TILE_X_COLUMN],
                           ResultsKey.TILE_Y: batch[TilesDataset.TILE_Y_COLUMN]}
                           )
        else:
            if is_global_rank_zero():
                logging.warning("Coordinates not found in batch. If this is not expected check your"
                                "input tiles dataset.")
        return results

    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:  # type: ignore
        train_result = self._shared_step(batch, batch_idx, 'train')
        self.log('train/loss', train_result[ResultsKey.LOSS], on_epoch=True, on_step=True, logger=True,
                 sync_dist=True)
        if self.verbose:
            print(f"After loading images batch {batch_idx} -", _format_cuda_memory_stats())
        self.log_metrics('train')
        return train_result[ResultsKey.LOSS]

    def validation_step(self, batch: Dict, batch_idx: int) -> BatchResultsType:  # type: ignore
        val_result = self._shared_step(batch, batch_idx, 'val')
        self.log('val/loss', val_result[ResultsKey.LOSS], on_epoch=True, on_step=True, logger=True,
                 sync_dist=True)
        self.log_metrics('val')
        return val_result

    def test_step(self, batch: Dict, batch_idx: int) -> BatchResultsType:   # type: ignore
        test_result = self._shared_step(batch, batch_idx, 'test')
        self.log('test/loss', test_result[ResultsKey.LOSS], on_epoch=True, on_step=True, logger=True,
                 sync_dist=True)
        self.log_metrics('test')
        return test_result

    def validation_epoch_end(self, epoch_results: EpochResultsType) -> None:
        if self.outputs_handler:
            self.outputs_handler.save_validation_outputs(
                epoch_results=epoch_results,
                metrics_dict=self.get_metrics_dict('val'),
                epoch=self.current_epoch,
                is_global_rank_zero=self.global_rank == 0
            )

    def test_epoch_end(self, epoch_results: EpochResultsType) -> None:
        if self.outputs_handler:
            self.outputs_handler.save_test_outputs(
                epoch_results=epoch_results,
                is_global_rank_zero=self.global_rank == 0
            )


class TilesDeepMILModule(BaseDeepMILModule):
    """Base class for Tiles based deep multiple-instance learning."""

    def _compute_bag_labels_logits_and_attn_maps(self, batch: Dict) -> Tuple[Tensor, Tensor, List]:
        # The batch dict contains lists of tensors of different sizes, for all bags in the batch.
        # This means we can't stack them along a new axis without padding to the same length.
        # We could alternatively concatenate them, but this would require other changes (e.g. in
        # the attention layers) to correctly split the tensors by bag/slide ID.
        bag_labels_list = []
        bag_logits_list = []
        bag_attn_list = []
        for bag_idx in range(len(batch[self.label_column])):
            images = batch[TilesDataset.IMAGE_COLUMN][bag_idx]
            labels = batch[self.label_column][bag_idx]
            bag_labels_list.append(self.get_bag_label(labels))
            logit, attn = self(images)
            bag_logits_list.append(logit.view(-1))
            bag_attn_list.append(attn)
        bag_logits = torch.stack(bag_logits_list)
        bag_labels = torch.stack(bag_labels_list).view(-1)
        return bag_logits, bag_labels, bag_attn_list


class SlidesDeepMILModule(BaseDeepMILModule):
    """Base class for Slides based deep multiple-instance learning."""

    def _compute_bag_labels_logits_and_attn_maps(self, batch: Dict) -> Tuple[Tensor, Tensor, List]:
        bag_labels_list = []
        bag_logits_list = []
        bag_attn_list = []
        for bag_idx in range(len(batch[self.label_column])):
            images = batch[TilesDataset.IMAGE_COLUMN][bag_idx]
            label = batch[self.label_column][bag_idx]
            bag_labels_list.append(label)
            logit, attn = self(images)
            bag_logits_list.append(logit.view(-1))
            bag_attn_list.append(attn)
        bag_logits = torch.stack(bag_logits_list)
        bag_labels = torch.stack(bag_labels_list).view(-1)
        return bag_logits, bag_labels, bag_attn_list


