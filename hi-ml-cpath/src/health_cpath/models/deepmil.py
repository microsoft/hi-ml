#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from pathlib import Path
from pytorch_lightning import LightningModule
from torch import Tensor, argmax, mode, nn, optim, round
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from torchmetrics.classification import (MulticlassAUROC, MulticlassAccuracy, MulticlassConfusionMatrix,
                                         MulticlassCohenKappa, MulticlassAveragePrecision, BinaryConfusionMatrix,
                                         BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryCohenKappa,
                                         BinaryAUROC, BinarySpecificity, BinaryAveragePrecision)
from health_ml.utils import log_on_epoch
from health_ml.deep_learning_config import OptimizerParams
from health_cpath.models.encoders import IdentityEncoder
from health_cpath.utils.deepmil_utils import ClassifierParams, EncoderParams, PoolingParams
from health_cpath.datasets.base_dataset import TilesDataset
from health_cpath.utils.naming import DeepMILSubmodules, MetricsKey, ResultsKey, SlideKey, ModelKey, TileKey
from health_cpath.utils.output_utils import (BatchResultsType, DeepMILOutputsHandler, EpochResultsType,
                                             validate_class_names, EXTRA_PREFIX)


RESULTS_COLS = [ResultsKey.SLIDE_ID, ResultsKey.TILE_ID, ResultsKey.IMAGE_PATH, ResultsKey.PROB,
                ResultsKey.CLASS_PROBS, ResultsKey.PRED_LABEL, ResultsKey.TRUE_LABEL, ResultsKey.BAG_ATTN]


def _format_cuda_memory_stats() -> str:
    return (f"GPU {torch.cuda.current_device()} memory: "
            f"{torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB allocated, "
            f"{torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB reserved")


class DeepMILModule(LightningModule):
    """Base class for deep multiple-instance learning"""

    def __init__(self,
                 label_column: str,
                 n_classes: int,
                 class_weights: Optional[Tensor] = None,
                 class_names: Optional[Sequence[str]] = None,
                 encoder_params: EncoderParams = EncoderParams(),
                 pooling_params: PoolingParams = PoolingParams(),
                 classifier_params: ClassifierParams = ClassifierParams(),
                 optimizer_params: OptimizerParams = OptimizerParams(),
                 outputs_folder: Optional[Path] = None,
                 outputs_handler: Optional[DeepMILOutputsHandler] = None,
                 analyse_loss: Optional[bool] = False,
                 verbose: bool = False,
                 ) -> None:
        """
        :param label_column: Label key for input batch dictionary.
        :param n_classes: Number of output classes for MIL prediction. For binary classification, n_classes should be
         set to 1.
        :param class_weights: Tensor containing class weights (default=None).
        :param class_names: The names of the classes if available (default=None).
        :param encoder_params: Encoder parameters that specify all encoder specific attributes.
        :param pooling_params: Pooling layer parameters that specify all encoder specific attributes.
        :param classifier_params: Classifier parameters that specify all classifier specific attributes.
        :param optimizer_params: Optimizer parameters that specify all specific attributes to be used for oprimization.
        :param outputs_folder: Path to output folder where encoder checkpoint is downloaded.
        :param outputs_handler: A configured :py:class:`DeepMILOutputsHandler` object to save outputs for the best
            validation epoch and test stage. If omitted (default), no outputs will be saved to disk (aside from usual
            metrics logging).
        :param analyse_loss: If True, the loss is analysed per sample and analysed with LossAnalysisCallback.
        :param verbose: if True statements about memory usage are output at each step.
        """
        super().__init__()

        # Dataset specific attributes
        self.label_column = label_column
        self.n_classes = n_classes
        self.class_weights = class_weights
        self.class_names = validate_class_names(class_names, self.n_classes)

        self.encoder_params = encoder_params
        self.pooling_params = pooling_params
        self.classifier_params = classifier_params
        self.optimizer_params = optimizer_params

        self.save_hyperparameters()
        self.verbose = verbose
        self.outputs_handler = outputs_handler

        # This flag can be switched on before invoking trainer.validate() to enable saving additional time/memory
        # consuming validation outputs via calling self.on_run_extra_validation_epoch()
        self._on_extra_val_epoch = False

        # Model components
        self.encoder = encoder_params.get_encoder(outputs_folder)
        self.aggregation_fn, self.num_pooling = pooling_params.get_pooling_layer(self.encoder.num_encoding)
        self.classifier_fn = classifier_params.get_classifier(self.num_pooling, self.n_classes)
        self.activation_fn = self.get_activation()

        # Loss function
        self.loss_fn = self.get_loss(reduction="mean")
        self.loss_fn_no_reduction = self.get_loss(reduction="none")
        self.analyse_loss = analyse_loss

        # Metrics Objects
        self.train_metrics = self.get_metrics()
        self.val_metrics = self.get_metrics()
        self.test_metrics = self.get_metrics()

        self.reset_encoder_to_identity_if_caching()

    def reset_encoder_to_identity_if_caching(self) -> None:
        """If caching is enabled, the encoder is replaced with an identity encoder."""
        if self.encoder_params.is_caching:
            # Encoding is done in the datamodule, so here we provide instead a dummy
            # no-op IdentityEncoder to be used inside the model
            self.encoder = IdentityEncoder(input_dim=(self.encoder.num_encoding,))

    @staticmethod
    def copy_weights(
        current_submodule: nn.Module, pretrained_submodule: nn.Module, submodule_name: DeepMILSubmodules
    ) -> None:
        """Copy weights from pretrained submodule to current submodule.

        :param current_submodule: Submodule to copy weights to.
        :param pretrained_submodule: Submodule to copy weights from.
        :param submodule_name: Name of the submodule.
        """

        def _total_params(submodule: nn.Module) -> int:
            return sum(p.numel() for p in submodule.parameters())

        pre_total_params = _total_params(pretrained_submodule)
        cur_total_params = _total_params(current_submodule)

        if pre_total_params != cur_total_params:
            raise ValueError(f"Submodule {submodule_name} has different number of parameters "
                             f"({cur_total_params} vs {pre_total_params}) from pretrained model.")

        for param, pretrained_param in zip(
            current_submodule.state_dict().values(), pretrained_submodule.state_dict().values()
        ):
            try:
                param.data.copy_(pretrained_param.data)
            except Exception as e:
                raise ValueError(f"Failed to copy weights for {submodule_name} because of the following exception: {e}")

    def transfer_weights(self, pretrained_checkpoint_path: Optional[Path]) -> None:
        """Transfer weights from pretrained checkpoint if provided."""

        if pretrained_checkpoint_path:
            pretrained_model = self.load_from_checkpoint(checkpoint_path=str(pretrained_checkpoint_path))

            if self.encoder_params.pretrained_encoder:
                self.copy_weights(self.encoder, pretrained_model.encoder, DeepMILSubmodules.ENCODER)

            if self.pooling_params.pretrained_pooling:
                self.copy_weights(self.aggregation_fn, pretrained_model.aggregation_fn, DeepMILSubmodules.POOLING)

            if self.classifier_params.pretrained_classifier:
                if pretrained_model.n_classes != self.n_classes:
                    raise ValueError(f"Number of classes in pretrained model {pretrained_model.n_classes} "
                                     f"does not match number of classes in current model {self.n_classes}.")
                self.copy_weights(self.classifier_fn, pretrained_model.classifier_fn, DeepMILSubmodules.CLASSIFIER)

    def get_loss(self, reduction: str = "mean") -> Callable:
        if self.n_classes > 1:
            if self.class_weights is None:
                return nn.CrossEntropyLoss(reduction=reduction)
            else:
                class_weights = self.class_weights.float()
                return nn.CrossEntropyLoss(weight=class_weights, reduction=reduction)
        else:
            pos_weight = None
            if self.class_weights is not None:
                pos_weight = Tensor([self.class_weights[1] / (self.class_weights[0] + 1e-5)])
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)

    def get_activation(self) -> Callable:
        if self.n_classes > 1:
            return nn.Softmax(dim=-1)
        else:
            return nn.Sigmoid()

    @staticmethod
    def get_bag_label(labels: Tensor) -> Tensor:
        """ Get bag label as the majority class of the bag's samples. For slides pipeline, we already have a single
        label per bag so we just return that label. For tiles pipeline, we need to get the majority class as labels
        are duplicated for each tile in the bag."""
        if len(labels.shape) == 0:
            return labels
        bag_label = mode(labels).values
        return bag_label.view(1)

    def get_metrics(self) -> nn.ModuleDict:
        if self.n_classes > 1:
            return nn.ModuleDict({
                MetricsKey.ACC: MulticlassAccuracy(num_classes=self.n_classes, average='micro'),
                MetricsKey.AUROC: MulticlassAUROC(num_classes=self.n_classes),
                MetricsKey.AVERAGE_PRECISION: MulticlassAveragePrecision(num_classes=self.n_classes),
                # Quadratic Weighted Kappa (QWK) used in PANDA challenge
                # is calculated using Cohen's Kappa with quadratic weights
                # https://www.kaggle.com/code/reighns/understanding-the-quadratic-weighted-kappa/
                MetricsKey.COHENKAPPA: MulticlassCohenKappa(num_classes=self.n_classes, weights='quadratic'),
                MetricsKey.CONF_MATRIX: MulticlassConfusionMatrix(num_classes=self.n_classes),
                # Metrics below are computed for multi-class case only
                MetricsKey.ACC_MACRO: MulticlassAccuracy(num_classes=self.n_classes, average='macro'),
                MetricsKey.ACC_WEIGHTED: MulticlassAccuracy(num_classes=self.n_classes, average='weighted')})
        else:
            return nn.ModuleDict({
                MetricsKey.ACC: BinaryAccuracy(),
                MetricsKey.AUROC: BinaryAUROC(),
                # Average precision is a measure of area under the PR curve
                MetricsKey.AVERAGE_PRECISION: BinaryAveragePrecision(),
                MetricsKey.COHENKAPPA: BinaryCohenKappa(weights='quadratic'),
                MetricsKey.CONF_MATRIX: BinaryConfusionMatrix(),
                # Metrics below are computed for binary case only
                MetricsKey.F1: BinaryF1Score(),
                MetricsKey.PRECISION: BinaryPrecision(),
                MetricsKey.RECALL: BinaryRecall(),
                MetricsKey.SPECIFICITY: BinarySpecificity()})

    def get_extra_prefix(self) -> str:
        """Get extra prefix for the metrics name to avoir overriding best validation metrics."""
        return EXTRA_PREFIX if self._on_extra_val_epoch else ""

    def log_metrics(self, stage: str, prefix: str = '') -> None:
        valid_stages = set([stage for stage in ModelKey])
        if stage not in valid_stages:
            raise Exception(f"Invalid stage. Chose one of {valid_stages}")
        for metric_name, metric_object in self.get_metrics_dict(stage).items():
            if metric_name == MetricsKey.CONF_MATRIX:
                metric_value = metric_object.compute()
                metric_value_n = metric_value / metric_value.sum(axis=1, keepdims=True)
                for i in range(metric_value_n.shape[0]):
                    log_on_epoch(self, f'{prefix}{stage}/{self.class_names[i]}', metric_value_n[i, i])
            else:
                log_on_epoch(self, f'{prefix}{stage}/{metric_name}', metric_object)

    def get_instance_features(self, instances: Tensor) -> Tensor:
        if not self.encoder_params.tune_encoder:
            self.encoder.eval()
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
        attentions, bag_features = self.aggregation_fn(instance_features)  # K x N | K x L
        bag_features = bag_features.view(1, -1)
        return attentions, bag_features

    def get_bag_logit(self, bag_features: Tensor) -> Tensor:
        if not self.classifier_params.tune_classifier:
            self.classifier_fn.eval()
        bag_logit = self.classifier_fn(bag_features)
        return bag_logit

    def forward(self, instances: Tensor) -> Tuple[Tensor, Tensor]:  # type: ignore
        instance_features = self.get_instance_features(instances)
        attentions, bag_features = self.get_attentions_and_bag_features(instance_features)
        bag_logit = self.get_bag_logit(bag_features)
        return bag_logit, attentions

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                          lr=self.optimizer_params.l_rate,
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

    def update_results_with_metadata(self, batch: Dict, results: Dict) -> None:
        """Update results with metadata. This can be either tiles or slides metadata including tiles coordinates."""
        results.update({ResultsKey.SLIDE_ID: batch[SlideKey.SLIDE_ID],
                        ResultsKey.TILE_ID: batch[TileKey.TILE_ID]})
        # Add tiles coordinates if available
        coordinates_keys = [TileKey.TILE_TOP, TileKey.TILE_LEFT, TileKey.TILE_RIGHT, TileKey.TILE_LEFT]
        if all([key in batch for key in coordinates_keys]):
            for key in coordinates_keys:
                results[key] = batch[key]
        elif TilesDataset.TILE_X_COLUMN in batch and TilesDataset.TILE_X_COLUMN in batch:
            results[ResultsKey.TILE_LEFT] = batch[TilesDataset.TILE_X_COLUMN]
            results[ResultsKey.TILE_TOP] = batch[TilesDataset.TILE_Y_COLUMN]
        else:
            rank_zero_warn(message="Coordinates not found in batch. If this is not expected check your"
                           "input tiles dataset.")

    def update_slides_selection(self, stage: str, batch: Dict, results: Dict) -> None:
        if (
            (stage == ModelKey.TEST or (stage == ModelKey.VAL and self._on_extra_val_epoch))
            and self.outputs_handler
            and self.outputs_handler.tiles_selector
        ):
            self.outputs_handler.tiles_selector.update_slides_selection(batch, results)

    def _compute_loss(self, loss_fn: Callable, bag_logits: Tensor, bag_labels: Tensor) -> Tensor:
        if self.n_classes > 1:
            return loss_fn(bag_logits, bag_labels.long())
        else:
            return loss_fn(bag_logits.squeeze(1), bag_labels.float())

    def _shared_step(self, batch: Dict, batch_idx: int, stage: str) -> BatchResultsType:

        bag_logits, bag_labels, bag_attn_list = self.compute_bag_labels_logits_and_attn_maps(batch)

        loss = self._compute_loss(self.loss_fn, bag_logits, bag_labels)

        predicted_probs = self.activation_fn(bag_logits)
        if self.n_classes > 1:
            predicted_labels = argmax(predicted_probs, dim=1)
            probs_perclass = predicted_probs
        else:
            predicted_labels = round(predicted_probs).int()
            probs_perclass = Tensor([[1.0 - predicted_probs[i][0].item(), predicted_probs[i][0].item()]
                                     for i in range(len(predicted_probs))])

        loss = loss.view(-1, 1)
        predicted_labels = predicted_labels.view(-1, 1)
        batch_size = predicted_labels.shape[0]

        if self.n_classes == 1:
            predicted_probs = predicted_probs.squeeze(dim=1)

        results = dict()

        if self.analyse_loss and stage in [ModelKey.TRAIN, ModelKey.VAL]:
            loss_per_sample = self._compute_loss(self.loss_fn_no_reduction, bag_logits, bag_labels)
            results[ResultsKey.LOSS_PER_SAMPLE] = loss_per_sample.detach().cpu().numpy()

        bag_labels = bag_labels.view(-1, 1)

        for metric_object in self.get_metrics_dict(stage).values():
            metric_object.update(predicted_probs, bag_labels.view(batch_size,).int())
        results.update({ResultsKey.LOSS: loss,
                        ResultsKey.PROB: predicted_probs,
                        ResultsKey.CLASS_PROBS: probs_perclass,
                        ResultsKey.PRED_LABEL: predicted_labels,
                        ResultsKey.TRUE_LABEL: bag_labels.int(),
                        ResultsKey.BAG_ATTN: bag_attn_list
                        })
        self.update_results_with_metadata(batch=batch, results=results)
        self.update_slides_selection(stage=stage, batch=batch, results=results)
        return results

    def training_step(self, batch: Dict, batch_idx: int) -> BatchResultsType:  # type: ignore
        train_result = self._shared_step(batch, batch_idx, ModelKey.TRAIN)
        self.log('train/loss', train_result[ResultsKey.LOSS], on_epoch=True, on_step=True, logger=True, sync_dist=True)
        if self.verbose:
            print(f"After loading images batch {batch_idx} -", _format_cuda_memory_stats())
        results = {ResultsKey.LOSS: train_result[ResultsKey.LOSS]}
        if self.analyse_loss:
            results.update({ResultsKey.LOSS_PER_SAMPLE: train_result[ResultsKey.LOSS_PER_SAMPLE],
                            ResultsKey.CLASS_PROBS: train_result[ResultsKey.CLASS_PROBS],
                            ResultsKey.TILE_ID: train_result[ResultsKey.TILE_ID]})
        return results

    def validation_step(self, batch: Dict, batch_idx: int) -> BatchResultsType:  # type: ignore
        val_result = self._shared_step(batch, batch_idx, ModelKey.VAL)
        name = f'{self.get_extra_prefix()}val/loss'
        self.log(name, val_result[ResultsKey.LOSS], on_epoch=True, on_step=True, logger=True, sync_dist=True)
        return val_result

    def test_step(self, batch: Dict, batch_idx: int) -> BatchResultsType:  # type: ignore
        test_result = self._shared_step(batch, batch_idx, ModelKey.TEST)
        self.log('test/loss', test_result[ResultsKey.LOSS], on_epoch=True, on_step=True, logger=True, sync_dist=True)
        return test_result

    def training_epoch_end(self, outputs: EpochResultsType) -> None:  # type: ignore
        self.log_metrics(ModelKey.TRAIN)

    def validation_epoch_end(self, epoch_results: EpochResultsType) -> None:  # type: ignore
        self.log_metrics(stage=ModelKey.VAL, prefix=self.get_extra_prefix())
        if self.outputs_handler:
            self.outputs_handler.save_validation_outputs(
                epoch_results=epoch_results,
                metrics_dict=self.get_metrics_dict(ModelKey.VAL),  # type: ignore
                epoch=self.current_epoch,
                is_global_rank_zero=self.global_rank == 0,
                on_extra_val=self._on_extra_val_epoch
            )

    def test_epoch_end(self, epoch_results: EpochResultsType) -> None:  # type: ignore
        self.log_metrics(ModelKey.TEST)
        if self.outputs_handler:
            self.outputs_handler.save_test_outputs(
                epoch_results=epoch_results,
                is_global_rank_zero=self.global_rank == 0
            )

    def on_run_extra_validation_epoch(self) -> None:
        """Hook to be called at the beginning of an extra validation epoch to set validation plots options to the same
        as the test plots options."""
        self._on_extra_val_epoch = True
        if self.outputs_handler:
            self.outputs_handler.val_plots_handler.plot_options = self.outputs_handler.test_plots_handler.plot_options
