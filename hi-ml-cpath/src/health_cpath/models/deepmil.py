#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from pytorch_lightning.utilities.warnings import rank_zero_warn
from pathlib import Path

from pytorch_lightning import LightningModule
from torch import Tensor, argmax, mode, nn, optim, round, set_grad_enabled
from torchmetrics import AUROC, F1, Accuracy, ConfusionMatrix, Precision, Recall, CohenKappa

from health_ml.utils import log_on_epoch
from health_ml.deep_learning_config import OptimizerParams
from health_cpath.models.encoders import IdentityEncoder
from health_cpath.utils.deepmil_utils import EncoderParams, PoolingParams

from health_cpath.datasets.base_dataset import TilesDataset
from health_cpath.utils.naming import MetricsKey, ResultsKey, SlideKey, ModelKey, TileKey
from health_cpath.utils.output_utils import (BatchResultsType, DeepMILOutputsHandler, EpochResultsType,
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
                 class_weights: Optional[Tensor] = None,
                 class_names: Optional[Sequence[str]] = None,
                 dropout_rate: Optional[float] = None,
                 verbose: bool = False,
                 ssl_ckpt_run_id: Optional[str] = None,
                 outputs_folder: Optional[Path] = None,
                 encoder_params: EncoderParams = EncoderParams(),
                 pooling_params: PoolingParams = PoolingParams(),
                 optimizer_params: OptimizerParams = OptimizerParams(),
                 outputs_handler: Optional[DeepMILOutputsHandler] = None,
                 pretrained_checkpoint_path: Optional[Path] = None,
                 use_pretrained_classifier: bool = False) -> None:
        """
        :param label_column: Label key for input batch dictionary.
        :param n_classes: Number of output classes for MIL prediction. For binary classification, n_classes should be
         set to 1.
        :param class_weights: Tensor containing class weights (default=None).
        :param class_names: The names of the classes if available (default=None).
        :param dropout_rate: Rate of pre-classifier dropout (0-1). `None` for no dropout (default).
        :param verbose: if True statements about memory usage are output at each step.
        :param ssl_ckpt_run_id: Optional parameter to provide the AML run id from where to download the checkpoint
            if using `SSLEncoder`.
        :param outputs_folder: Path to output folder where encoder checkpoint is downloaded.
        :param encoder_params: Encoder parameters that specify all encoder specific attributes.
        :param pooling_params: Pooling layer parameters that specify all encoder specific attributes.
        :param optimizer_params: Optimizer parameters that specify all specific attributes to be used for oprimization.
        :param outputs_handler: A configured :py:class:`DeepMILOutputsHandler` object to save outputs for the best
            validation epoch and test stage. If omitted (default), no outputs will be saved to disk (aside from usual
            metrics logging).
        """
        super().__init__()

        # Dataset specific attributes
        self.label_column = label_column
        self.n_classes = n_classes
        self.class_weights = class_weights
        self.class_names = validate_class_names(class_names, self.n_classes)

        self.dropout_rate = dropout_rate
        self.encoder_params = encoder_params
        self.pooling_params = pooling_params
        self.optimizer_params = optimizer_params

        self.save_hyperparameters()
        self.verbose = verbose
        self.outputs_handler = outputs_handler
        self.use_pretrained_classifier = use_pretrained_classifier

        # This flag can be switched on before invoking trainer.validate() to enable saving additional time/memory
        # consuming validation outputs
        self.run_extra_val_epoch = False

        # Model components
        self.encoder = encoder_params.get_encoder(ssl_ckpt_run_id, outputs_folder)
        self.aggregation_fn, self.num_pooling = pooling_params.get_pooling_layer(self.encoder.num_encoding)
        self.classifier_fn = self.get_classifier()

        self.transfer_weights(pretrained_checkpoint_path)

        self.activation_fn = self.get_activation()
        self.loss_fn = self.get_loss()

        # Metrics Objects
        self.train_metrics = self.get_metrics()
        self.val_metrics = self.get_metrics()
        self.test_metrics = self.get_metrics()

    def transfer_weights(self, pretrained_checkpoint_path: Optional[Path]) -> None:
        """Transfer weights from pretrained checkpoint if provided."""

        if pretrained_checkpoint_path:
            pretrained_model = self.load_from_checkpoint(checkpoint_path=str(pretrained_checkpoint_path))

            pretrained_model.encoder_params.use_pretrained_encoder = False
            pretrained_model.pooling_params.use_pretrained_pooling = False
            pretrained_model.use_pretrained_classifier = False

            if self.encoder_params.use_pretrained_encoder:
                for param, pretrained_param in zip(self.encoder.parameters(), pretrained_model.encoder.parameters()):
                    param.data.copy_(pretrained_param.data)

            if self.pooling_params.use_pretrained_pooling:
                for param, pretrained_param in zip(
                    self.aggregation_fn.parameters(), pretrained_model.aggregation_fn.parameters()
                ):
                    param.data.copy_(pretrained_param.data)

            if (
                self.use_pretrained_classifier
                and pretrained_model.classifier_fn.output_dim == self.classifier_fn.output_dim
            ):
                for param, pretrained_param in zip(
                    self.classifier_fn.parameters(), pretrained_model.classifier_fn.parameters()
                ):
                    param.data.copy_(pretrained_param.data)

    def get_classifier(self) -> nn.Module:
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
        raise NotImplementedError

    def get_metrics(self) -> nn.ModuleDict:
        if self.n_classes > 1:
            return nn.ModuleDict({MetricsKey.ACC: Accuracy(num_classes=self.n_classes),
                                  MetricsKey.ACC_MACRO: Accuracy(num_classes=self.n_classes, average='macro'),
                                  MetricsKey.ACC_WEIGHTED: Accuracy(num_classes=self.n_classes, average='weighted'),
                                  MetricsKey.AUROC: AUROC(num_classes=self.n_classes),
                                  # Quadratic Weighted Kappa (QWK) used in PANDA challenge
                                  # is calculated using Cohen's Kappa with quadratic weights
                                  # https://www.kaggle.com/code/reighns/understanding-the-quadratic-weighted-kappa/
                                  MetricsKey.COHENKAPPA: CohenKappa(num_classes=self.n_classes, weights='quadratic'),
                                  MetricsKey.CONF_MATRIX: ConfusionMatrix(num_classes=self.n_classes)})
        else:
            threshold = 0.5
            return nn.ModuleDict({MetricsKey.ACC: Accuracy(threshold=threshold),
                                  MetricsKey.AUROC: AUROC(num_classes=self.n_classes),
                                  MetricsKey.PRECISION: Precision(threshold=threshold),
                                  MetricsKey.RECALL: Recall(threshold=threshold),
                                  MetricsKey.F1: F1(threshold=threshold),
                                  MetricsKey.CONF_MATRIX: ConfusionMatrix(num_classes=2, threshold=threshold)})

    def log_metrics(self, stage: str) -> None:
        valid_stages = [stage for stage in ModelKey]
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

    def get_instance_features(self, instances: Tensor) -> Tensor:
        should_enable_encoder_grad = torch.is_grad_enabled() and self.encoder_params.tune_encoder
        if not self.encoder_params.tune_encoder:
            self.encoder.eval()
        with set_grad_enabled(should_enable_encoder_grad):
            if self.encoder_params.encoding_chunk_size > 0:
                embeddings = []
                chunks = torch.split(instances, self.encoder_params.encoding_chunk_size)
                for chunk in chunks:
                    chunk_embeddings = self.encoder(chunk)
                    embeddings.append(chunk_embeddings)
                instance_features = torch.cat(embeddings)
            else:
                instance_features = self.encoder(instances)  # N X L x 1 x 1
        return instance_features

    def get_attentions_and_bag_features(self, instance_features: Tensor) -> Tuple[Tensor, Tensor]:
        if not self.pooling_params.tune_pooling:
            self.aggregation_fn.eval()
        should_enable_pooling_grad = torch.is_grad_enabled() and self.pooling_params.tune_pooling
        with set_grad_enabled(should_enable_pooling_grad):
            attentions, bag_features = self.aggregation_fn(instance_features)  # K x N | K x L
        bag_features = bag_features.view(1, -1)
        return attentions, bag_features

    def forward(self, instances: Tensor) -> Tuple[Tensor, Tensor]:  # type: ignore
        instance_features = self.get_instance_features(instances)
        attentions, bag_features = self.get_attentions_and_bag_features(instance_features)
        bag_logit = self.classifier_fn(bag_features)
        return bag_logit, attentions

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.parameters(), lr=self.optimizer_params.l_rate,
                          weight_decay=self.optimizer_params.weight_decay,
                          betas=self.optimizer_params.adam_betas)

    def get_metrics_dict(self, stage: str) -> nn.ModuleDict:
        return getattr(self, f'{stage}_metrics')

    def compute_bag_labels_logits_and_attn_maps(self, batch: Dict) -> Tuple[Tensor, Tensor, List]:
        # The batch dict contains lists of tensors of different sizes, for all bags in the batch.
        # This means we can't stack them along a new axis without padding to the same length.
        # We could alternatively concatenate them, but this would require other changes (e.g. in
        # the attention layers) to correctly split the tensors by bag/slide ID.
        bag_labels_list = []
        bag_logits_list = []
        bag_attn_list = []
        for bag_idx in range(len(batch[self.label_column])):
            images = batch[TilesDataset.IMAGE_COLUMN][bag_idx].float()
            labels = batch[self.label_column][bag_idx]
            bag_labels_list.append(self.get_bag_label(labels))
            logit, attn = self(images)
            bag_logits_list.append(logit.view(-1))
            bag_attn_list.append(attn)
        bag_logits = torch.stack(bag_logits_list)
        bag_labels = torch.stack(bag_labels_list).view(-1)
        return bag_logits, bag_labels, bag_attn_list

    def update_results_with_data_specific_info(self, batch: Dict, results: Dict) -> None:
        """Update training results with data specific info. This can be either tiles or slides related metadata."""
        raise NotImplementedError

    def _shared_step(self, batch: Dict, batch_idx: int, stage: str) -> BatchResultsType:

        bag_logits, bag_labels, bag_attn_list = self.compute_bag_labels_logits_and_attn_maps(batch)

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
            metric_object.update(predicted_probs, bag_labels.view(batch_size,).int())
        results.update({ResultsKey.LOSS: loss,
                        ResultsKey.PROB: predicted_probs,
                        ResultsKey.CLASS_PROBS: probs_perclass,
                        ResultsKey.PRED_LABEL: predicted_labels,
                        ResultsKey.TRUE_LABEL: bag_labels,
                        ResultsKey.BAG_ATTN: bag_attn_list
                        })
        self.update_results_with_data_specific_info(batch=batch, results=results)
        if (
            (stage == ModelKey.TEST or (stage == ModelKey.VAL and self.run_extra_val_epoch))
            and self.outputs_handler
            and self.outputs_handler.tiles_selector
        ):
            self.outputs_handler.tiles_selector.update_slides_selection(batch, results)
        return results

    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:  # type: ignore
        train_result = self._shared_step(batch, batch_idx, ModelKey.TRAIN)
        self.log('train/loss', train_result[ResultsKey.LOSS], on_epoch=True, on_step=True, logger=True,
                 sync_dist=True)
        if self.verbose:
            print(f"After loading images batch {batch_idx} -", _format_cuda_memory_stats())
        return train_result[ResultsKey.LOSS]

    def validation_step(self, batch: Dict, batch_idx: int) -> BatchResultsType:  # type: ignore
        val_result = self._shared_step(batch, batch_idx, ModelKey.VAL)
        self.log('val/loss', val_result[ResultsKey.LOSS], on_epoch=True, on_step=True, logger=True,
                 sync_dist=True)
        return val_result

    def test_step(self, batch: Dict, batch_idx: int) -> BatchResultsType:  # type: ignore
        test_result = self._shared_step(batch, batch_idx, ModelKey.TEST)
        self.log('test/loss', test_result[ResultsKey.LOSS], on_epoch=True, on_step=True, logger=True,
                 sync_dist=True)
        return test_result

    def training_epoch_end(self, outputs: EpochResultsType) -> None:  # type: ignore
        self.log_metrics(ModelKey.TRAIN)

    def validation_epoch_end(self, epoch_results: EpochResultsType) -> None:  # type: ignore
        self.log_metrics(ModelKey.VAL)
        if self.outputs_handler:
            if self.run_extra_val_epoch:
                self.outputs_handler.val_plots_handler.plot_options = (
                    self.outputs_handler.test_plots_handler.plot_options
                )
            self.outputs_handler.save_validation_outputs(
                epoch_results=epoch_results,
                metrics_dict=self.get_metrics_dict(ModelKey.VAL),  # type: ignore
                epoch=self.current_epoch,
                is_global_rank_zero=self.global_rank == 0,
                run_extra_val_epoch=self.run_extra_val_epoch
            )

            # Reset the top and bottom slides heaps
            if self.outputs_handler.tiles_selector is not None:
                self.outputs_handler.tiles_selector._clear_cached_slides_heaps()

    def test_epoch_end(self, epoch_results: EpochResultsType) -> None:  # type: ignore
        self.log_metrics(ModelKey.TEST)
        if self.outputs_handler:
            self.outputs_handler.save_test_outputs(
                epoch_results=epoch_results,
                is_global_rank_zero=self.global_rank == 0
            )


class TilesDeepMILModule(BaseDeepMILModule):
    """Base class for Tiles based deep multiple-instance learning."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.encoder_params.is_caching:
            # Encoding is done in the datamodule, so here we provide instead a dummy
            # no-op IdentityEncoder to be used inside the model
            self.encoder = IdentityEncoder(input_dim=(self.encoder.num_encoding,))

    @staticmethod
    def get_bag_label(labels: Tensor) -> Tensor:
        # Get bag (batch) labels as majority vote
        bag_label = mode(labels).values
        return bag_label.view(1)

    def update_results_with_data_specific_info(self, batch: Dict, results: Dict) -> None:
        results.update({ResultsKey.SLIDE_ID: batch[TilesDataset.SLIDE_ID_COLUMN],
                        ResultsKey.TILE_ID: batch[TilesDataset.TILE_ID_COLUMN],
                        ResultsKey.IMAGE_PATH: batch[TilesDataset.PATH_COLUMN]})

        if all(key in batch.keys() for key in [TileKey.TILE_TOP, TileKey.TILE_LEFT,
                                               TileKey.TILE_RIGHT, TileKey.TILE_BOTTOM]):
            results.update({ResultsKey.TILE_TOP: batch[TileKey.TILE_TOP],
                            ResultsKey.TILE_LEFT: batch[TileKey.TILE_LEFT],
                            ResultsKey.TILE_RIGHT: batch[TileKey.TILE_RIGHT],
                            ResultsKey.TILE_BOTTOM: batch[TileKey.TILE_BOTTOM]})
        # the condition below ensures compatibility with older tile datasets (without LEFT, TOP, RIGHT, BOTTOM)
        elif (TilesDataset.TILE_X_COLUMN in batch.keys()) and (TilesDataset.TILE_Y_COLUMN in batch.keys()):
            results.update({ResultsKey.TILE_LEFT: batch[TilesDataset.TILE_X_COLUMN],
                           ResultsKey.TILE_TOP: batch[TilesDataset.TILE_Y_COLUMN]})
        else:
            rank_zero_warn(message="Coordinates not found in batch. If this is not expected check your"
                           "input tiles dataset.")


class SlidesDeepMILModule(BaseDeepMILModule):
    """Base class for slides based deep multiple-instance learning."""

    @staticmethod
    def get_bag_label(labels: Tensor) -> Tensor:
        # SlidesDataModule attributes a single label to a bag of tiles already no need to do majority voting
        return labels

    def update_results_with_data_specific_info(self, batch: Dict, results: Dict) -> None:
        # WARNING: This is a dummy input until we figure out tiles coordinates retrieval in the next iteration.
        bag_sizes = [tiles.shape[0] for tiles in batch[SlideKey.IMAGE]]
        results.update(
            {
                ResultsKey.SLIDE_ID: [
                    [slide_id] * bag_sizes[i] for i, slide_id in enumerate(batch[SlideKey.SLIDE_ID])
                ],
                ResultsKey.TILE_ID: [
                    [f"{slide_id}_{tile_id}" for tile_id in range(bag_sizes[i])]
                    for i, slide_id in enumerate(batch[SlideKey.SLIDE_ID])
                ],
                ResultsKey.IMAGE_PATH: [
                    [img_path] * bag_sizes[i] for i, img_path in enumerate(batch[SlideKey.IMAGE_PATH])
                ],
            }
        )
