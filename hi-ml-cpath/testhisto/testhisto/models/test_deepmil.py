#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from copy import deepcopy
import os
from pytorch_lightning import Trainer
import torch
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Any, Callable, Dict, Iterable, List, Optional, Type

from torch import Tensor, argmax, nn, rand, randint, randn, round, stack, allclose
from torch.utils.data._utils.collate import default_collate
from health_cpath.configs.classification.DeepSMILESlidesPandaBenchmark import SlidesPandaSSLMILBenchmark
from health_cpath.datamodules.panda_module import PandaTilesDataModule

from health_ml.networks.layers.attention_layers import AttentionLayer, TransformerPoolingBenchmark
from health_cpath.configs.classification.BaseMIL import BaseMIL, BaseMILTiles

from health_cpath.configs.classification.DeepSMILECrck import DeepSMILECrck, TcgaCrckSSLMIL
from health_cpath.configs.classification.DeepSMILEPanda import (
    BaseDeepSMILEPanda, DeepSMILETilesPanda, SlidesPandaSSLMIL, TilesPandaSSLMIL
)
from health_cpath.datamodules.base_module import HistoDataModule, TilesDataModule
from health_cpath.datasets.base_dataset import DEFAULT_LABEL_COLUMN, TilesDataset
from health_cpath.datasets.default_paths import PANDA_5X_TILES_DATASET_ID, TCGA_CRCK_DATASET_DIR
from health_cpath.models.deepmil import DeepMILModule
from health_cpath.models.encoders import IdentityEncoder, ImageNetEncoder, Resnet18, TileEncoder
from health_cpath.utils.deepmil_utils import ClassifierParams, EncoderParams, PoolingParams
from health_cpath.utils.naming import DeepMILSubmodules, MetricsKey, ResultsKey
from testhisto.mocks.base_data_generator import MockHistoDataType
from testhisto.mocks.tiles_generator import MockPandaTilesGenerator
from testhisto.mocks.container import MockDeepSMILETilesPanda, MockDeepSMILESlidesPanda
from health_ml.utils.common_utils import is_gpu_available
from health_ml.utils.checkpoint_utils import CheckpointParser
from health_cpath.configs.run_ids import innereye_ssl_checkpoint_crck_4ws, innereye_ssl_checkpoint_binary
from testhisto.models.test_encoders import TEST_SSL_RUN_ID

no_gpu = not is_gpu_available()


def get_supervised_imagenet_encoder_params(tune_encoder: bool = True, is_caching: bool = False) -> EncoderParams:
    return EncoderParams(encoder_type=Resnet18.__name__, tune_encoder=tune_encoder, is_caching=is_caching)


def get_attention_pooling_layer_params(pool_out_dim: int = 1, tune_pooling: bool = True) -> PoolingParams:
    return PoolingParams(pool_type=AttentionLayer.__name__, pool_out_dim=pool_out_dim, pool_hidden_dim=5,
                         tune_pooling=tune_pooling)


def get_transformer_pooling_layer_params(num_layers: int, num_heads: int,
                                         hidden_dim: int, transformer_dropout: float) -> PoolingParams:
    return PoolingParams(pool_type=TransformerPoolingBenchmark.__name__,
                         num_transformer_pool_layers=num_layers,
                         num_transformer_pool_heads=num_heads,
                         pool_hidden_dim=hidden_dim,
                         transformer_dropout=transformer_dropout)


def _test_lightningmodule(
    n_classes: int,
    batch_size: int,
    max_bag_size: int,
    pool_out_dim: int,
    dropout_rate: Optional[float],
) -> None:

    assert n_classes > 0

    module = DeepMILModule(
        label_column=DEFAULT_LABEL_COLUMN,
        n_classes=n_classes,
        classifier_params=ClassifierParams(dropout_rate=dropout_rate),
        encoder_params=get_supervised_imagenet_encoder_params(),
        pooling_params=get_attention_pooling_layer_params(pool_out_dim)
    )

    bag_images = rand([batch_size, max_bag_size, *module.encoder.input_dim])
    bag_labels_list = []
    bag_logits_list = []
    bag_attn_list = []
    for bag in bag_images:
        if n_classes > 1:
            labels = randint(n_classes, size=(max_bag_size,))
        else:
            labels = randint(n_classes + 1, size=(max_bag_size,))
        bag_labels_list.append(module.get_bag_label(labels))
        logit, attn = module(bag)
        assert logit.shape == (1, n_classes)
        assert attn.shape == (pool_out_dim, max_bag_size)
        bag_logits_list.append(logit.view(-1))
        bag_attn_list.append(attn)

    bag_logits = stack(bag_logits_list)
    bag_labels = stack(bag_labels_list).view(-1)

    assert bag_logits.shape[0] == (batch_size)
    assert bag_labels.shape[0] == (batch_size)

    if module.n_classes > 1:
        loss = module.loss_fn(bag_logits, bag_labels)
    else:
        loss = module.loss_fn(bag_logits.squeeze(1), bag_labels.float())

    assert loss > 0
    assert loss.shape == ()

    probs = module.activation_fn(bag_logits)
    assert ((probs >= 0) & (probs <= 1)).all()

    if n_classes > 1:
        predlabels = argmax(probs, dim=1)
    else:
        predlabels = round(probs)

    predlabels = predlabels.view(-1, 1)
    assert predlabels.shape[0] == batch_size
    assert probs.shape == (batch_size, n_classes)

    bag_labels = bag_labels.view(-1, 1)
    if n_classes == 1:
        probs = probs.squeeze(dim=1)
    for metric_name, metric_object in module.train_metrics.items():
        if metric_name == MetricsKey.CONF_MATRIX:
            continue
        score = metric_object(probs, bag_labels.view(batch_size,))
        if metric_name == MetricsKey.COHENKAPPA:
            # A NaN value could result due to a division-by-zero error
            assert torch.all(score[~score.isnan()] >= -1)
            assert torch.all(score[~score.isnan()] <= 1)
        elif metric_name == MetricsKey.AVERAGE_PRECISION:
            assert torch.all(score[~score.isnan()] >= 0)
            assert torch.all(score[~score.isnan()] <= 1)
        else:
            assert torch.all(score >= 0)
            assert torch.all(score <= 1)


@pytest.mark.parametrize("n_classes", [1, 3])
@pytest.mark.parametrize("batch_size", [1, 5])
@pytest.mark.parametrize("max_bag_size", [1, 5])
@pytest.mark.parametrize("pool_out_dim", [1, 6])
@pytest.mark.parametrize("dropout_rate", [None, 0.5])
def test_lightningmodule_attention(
    n_classes: int,
    batch_size: int,
    max_bag_size: int,
    pool_out_dim: int,
    dropout_rate: Optional[float],
) -> None:
    _test_lightningmodule(n_classes=n_classes,
                          batch_size=batch_size,
                          max_bag_size=max_bag_size,
                          pool_out_dim=pool_out_dim,
                          dropout_rate=dropout_rate)


def validate_metric_inputs(scores: torch.Tensor, labels: torch.Tensor) -> None:
    def is_integral(x: torch.Tensor) -> bool:
        return (x == x.long()).all()  # type: ignore

    assert labels.shape == (scores.shape[0], )
    assert torch.is_floating_point(scores), "Received scores with integer dtype"
    assert not is_integral(scores), "Received scores with integral values"
    assert is_integral(labels), "Received labels with floating-point values"


def add_callback(fn: Callable, callback: Callable) -> Callable:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        callback(*args, **kwargs)
        return fn(*args, **kwargs)
    return wrapper


@pytest.mark.parametrize("n_classes", [1, 3])
def test_metrics(n_classes: int) -> None:
    input_dim = (128,)

    def _mock_get_encoder(self, outputs_folder: Optional[Path]) -> TileEncoder:  # type: ignore
        return IdentityEncoder(input_dim=input_dim)

    with patch("health_cpath.models.deepmil.EncoderParams.get_encoder", new=_mock_get_encoder):
        module = DeepMILModule(label_column=DEFAULT_LABEL_COLUMN,
                               n_classes=n_classes,
                               pooling_params=get_attention_pooling_layer_params(pool_out_dim=1))

        # Patching to enable running the module without a Trainer object
        module.trainer = MagicMock(world_size=1)  # type: ignore
        module.log = MagicMock()  # type: ignore
        module.outputs_handler = MagicMock()

        batch_size = 20
        bag_size = 5
        if n_classes > 1:
            class_weights = torch.rand(n_classes)
        else:
            class_weights = torch.tensor([0.8, 0.2])
        bags: List[Dict] = []
        for slide_idx in range(batch_size):
            bag_label = torch.multinomial(class_weights, 1)
            sample: Dict[str, Iterable] = {
                TilesDataset.SLIDE_ID_COLUMN: [str(slide_idx)] * bag_size,
                TilesDataset.TILE_ID_COLUMN: [f"{slide_idx}-{tile_idx}" for tile_idx in range(bag_size)],
                TilesDataset.IMAGE_COLUMN: rand(bag_size, *input_dim),
                DEFAULT_LABEL_COLUMN: bag_label.expand(bag_size),
            }
            sample[TilesDataset.PATH_COLUMN] = [tile_id + '.png'
                                                for tile_id in sample[TilesDataset.TILE_ID_COLUMN]]
            bags.append(sample)
        batch = default_collate(bags)

        # ================
        # Test that the module metrics match manually computed metrics with the correct inputs
        module_metrics_dict = module.test_metrics
        independent_metrics_dict = module.get_metrics()

        # Patch the metrics to check that the inputs are valid. In particular, test that the scores
        # do not have integral values, which would suggest that hard labels were passed instead.
        for metric_obj in module_metrics_dict.values():
            metric_obj.update = add_callback(metric_obj.update, validate_metric_inputs)

        results = module.test_step(batch, 0)
        predicted_probs = results[ResultsKey.PROB]
        true_labels = results[ResultsKey.TRUE_LABEL]

        for key, metric_obj in module_metrics_dict.items():
            value = metric_obj.compute()
            expected_value = independent_metrics_dict[key](predicted_probs, true_labels.view(batch_size,))
            assert torch.allclose(value, expected_value), f"Discrepancy in '{key}' metric"

        assert all(key in results.keys() for key in [ResultsKey.SLIDE_ID, ResultsKey.TILE_ID])


def move_batch_to_expected_device(batch: Dict[str, List], use_gpu: bool) -> Dict:
    device = "cuda" if use_gpu else "cpu"
    return {
        key: [
            value.to(device) if isinstance(value, Tensor) else value for value in values
        ]
        for key, values in batch.items()
    }


def assert_train_step(module: DeepMILModule, data_module: HistoDataModule, use_gpu: bool) -> None:
    train_data_loader = data_module.train_dataloader()
    for batch_idx, batch in enumerate(train_data_loader):
        batch = move_batch_to_expected_device(batch, use_gpu)
        loss = module.training_step(batch, batch_idx)[ResultsKey.LOSS]
        loss.retain_grad()
        loss.backward()
        assert loss.grad is not None
        assert loss.shape == (1, 1)
        assert isinstance(loss, Tensor)
        break


def assert_validation_step(module: DeepMILModule, data_module: HistoDataModule, use_gpu: bool) -> None:
    val_data_loader = data_module.val_dataloader()
    for batch_idx, batch in enumerate(val_data_loader):
        batch = move_batch_to_expected_device(batch, use_gpu)
        outputs_dict = module.validation_step(batch, batch_idx)
        loss = outputs_dict[ResultsKey.LOSS]  # noqa
        assert loss.shape == (1, 1)  # noqa
        assert isinstance(loss, Tensor)
        break


def assert_test_step(module: DeepMILModule, data_module: HistoDataModule, use_gpu: bool) -> None:
    test_data_loader = data_module.test_dataloader()
    for batch_idx, batch in enumerate(test_data_loader):
        batch = move_batch_to_expected_device(batch, use_gpu)
        outputs_dict = module.test_step(batch, batch_idx)
        loss = outputs_dict[ResultsKey.LOSS]  # noqa
        assert loss.shape == (1, 1)  # noqa
        assert isinstance(loss, Tensor)
        break


CONTAINER_DATASET_DIR = {
    DeepSMILETilesPanda: PANDA_5X_TILES_DATASET_ID,
    DeepSMILECrck: TCGA_CRCK_DATASET_DIR,
}


@pytest.mark.parametrize("container_type", [DeepSMILETilesPanda,
                                            DeepSMILECrck])
@pytest.mark.parametrize("use_gpu", [True, False])
def test_container(container_type: Type[BaseMILTiles], use_gpu: bool) -> None:
    dataset_dir = CONTAINER_DATASET_DIR[container_type]
    if not os.path.isdir(dataset_dir):
        pytest.skip(
            f"Dataset for container {container_type.__name__} "
            f"is unavailable: {dataset_dir}"
        )
    if container_type in [DeepSMILECrck, DeepSMILETilesPanda]:
        container = container_type(encoder_type=ImageNetEncoder.__name__)
    else:
        container = container_type()

    container.setup()
    container.batch_size = 10
    container.batch_size_inf = 10

    data_module: TilesDataModule = container.get_data_module()  # type: ignore

    module = container.create_model()
    module.outputs_handler = MagicMock()
    module.trainer = MagicMock(world_size=1)  # type: ignore
    module.log = MagicMock()  # type: ignore
    if use_gpu:
        module.cuda()

    assert_train_step(module, data_module, use_gpu)
    assert_validation_step(module, data_module, use_gpu)
    assert_test_step(module, data_module, use_gpu)


def _test_mock_panda_container(use_gpu: bool, mock_container: BaseDeepSMILEPanda, tmp_path: Path) -> None:
    container = mock_container(tmp_path=tmp_path)
    container.setup()
    data_module = container.get_data_module()
    module = container.create_model()

    module.trainer = MagicMock(world_size=1)  # type: ignore
    module.outputs_handler = MagicMock()
    module.log = MagicMock()  # type: ignore
    if use_gpu:
        module.cuda()

    assert_train_step(module, data_module, use_gpu)
    assert_validation_step(module, data_module, use_gpu)
    assert_test_step(module, data_module, use_gpu)


def test_mock_tiles_panda_container_cpu(mock_panda_tiles_root_dir: Path) -> None:
    _test_mock_panda_container(use_gpu=False, mock_container=MockDeepSMILETilesPanda,  # type: ignore
                               tmp_path=mock_panda_tiles_root_dir)


@pytest.mark.skipif(no_gpu, reason="Test requires GPU")
@pytest.mark.gpu
@pytest.mark.parametrize("mock_container, tmp_path", [(MockDeepSMILETilesPanda, "mock_panda_tiles_root_dir"),
                                                      (MockDeepSMILESlidesPanda, "mock_panda_slides_root_dir")])
def test_mock_panda_container_gpu(mock_container: BaseDeepSMILEPanda,
                                  tmp_path: str,
                                  request: pytest.FixtureRequest) -> None:
    _test_mock_panda_container(use_gpu=True, mock_container=mock_container, tmp_path=request.getfixturevalue(tmp_path))


def test_class_weights_binary() -> None:
    class_weights = Tensor([0.5, 3.5])
    n_classes = 1

    module = DeepMILModule(
        label_column=DEFAULT_LABEL_COLUMN,
        n_classes=n_classes,
        class_weights=class_weights,
        encoder_params=get_supervised_imagenet_encoder_params(),
        pooling_params=get_attention_pooling_layer_params(pool_out_dim=1)
    )

    logits = Tensor(randn(1, n_classes))
    bag_label = randint(n_classes + 1, size=(1,))

    pos_weight = Tensor([class_weights[1] / (class_weights[0] + 1e-5)])
    loss_weighted = module.loss_fn(logits.squeeze(1), bag_label.float())
    criterion_unweighted = nn.BCEWithLogitsLoss()
    loss_unweighted = criterion_unweighted(logits.squeeze(1), bag_label.float())
    if bag_label.item() == 1:
        assert allclose(loss_weighted, pos_weight * loss_unweighted)
    else:
        assert allclose(loss_weighted, loss_unweighted)


def test_class_weights_multiclass() -> None:
    class_weights = Tensor([0.33, 0.33, 0.33])
    n_classes = 3

    module = DeepMILModule(
        label_column=DEFAULT_LABEL_COLUMN,
        n_classes=n_classes,
        class_weights=class_weights,
        encoder_params=get_supervised_imagenet_encoder_params(),
        pooling_params=get_attention_pooling_layer_params(pool_out_dim=1)
    )

    logits = Tensor(randn(1, n_classes))
    bag_label = randint(n_classes, size=(1,))

    loss_weighted = module.loss_fn(logits, bag_label)
    criterion_unweighted = nn.CrossEntropyLoss()
    loss_unweighted = criterion_unweighted(logits, bag_label)
    # The weighted and unweighted loss functions give the same loss values for batch_size = 1.
    # https://stackoverflow.com/questions/67639540/pytorch-cross-entropy-loss-weights-not-working
    # TODO: the test should reflect actual weighted loss operation for the class weights after
    # batch_size > 1 is implemented.
    assert allclose(loss_weighted, loss_unweighted)


def test_wrong_tuning_options() -> None:
    with pytest.raises(ValueError,
                       match=r"At least one of the encoder, pooling or classifier should be fine tuned"):
        _ = MockDeepSMILETilesPanda(
            tmp_path=Path("foo"),
            tune_encoder=False,
            tune_pooling=False,
            tune_classifier=False
        )


def _get_datamodule(tmp_path: Path) -> PandaTilesDataModule:
    tiles_generator = MockPandaTilesGenerator(
        dest_data_path=tmp_path,
        mock_type=MockHistoDataType.FAKE,
        n_tiles=4,
        n_slides=10,
        n_channels=3,
        tile_size=28,
        img_size=224,
        tiling_version=1,
    )
    tiles_generator.generate_mock_histo_data()
    datamodule = PandaTilesDataModule(root_path=tmp_path, batch_size=2, max_bag_size=4)
    return datamodule


@pytest.mark.parametrize("tune_classifier", [False, True])
@pytest.mark.parametrize("tune_pooling", [False, True])
@pytest.mark.parametrize("tune_encoder", [False, True])
def test_finetuning_options(
    tune_encoder: bool, tune_pooling: bool, tune_classifier: bool, tmp_path: Path
) -> None:
    module = DeepMILModule(
        n_classes=1,
        label_column=DEFAULT_LABEL_COLUMN,
        encoder_params=get_supervised_imagenet_encoder_params(tune_encoder=tune_encoder),
        pooling_params=get_attention_pooling_layer_params(pool_out_dim=1, tune_pooling=tune_pooling),
        classifier_params=ClassifierParams(tune_classifier=tune_classifier),
    )

    assert module.encoder_params.tune_encoder == tune_encoder
    assert module.pooling_params.tune_pooling == tune_pooling
    assert module.classifier_params.tune_classifier == tune_classifier

    for params in module.encoder.parameters():
        assert params.requires_grad == tune_encoder
    for params in module.aggregation_fn.parameters():
        assert params.requires_grad == tune_pooling
    for params in module.classifier_fn.parameters():
        assert params.requires_grad == tune_classifier

    instances = torch.randn(4, 3, 224, 224)

    def _assert_existing_gradients_fn(tensor: Tensor, tuning_flag: bool) -> None:
        assert tensor.requires_grad == tuning_flag
        if tuning_flag:
            assert tensor.grad_fn is not None
        else:
            assert tensor.grad_fn is None

    with torch.enable_grad():
        instance_features = module.get_instance_features(instances)
        _assert_existing_gradients_fn(instance_features, tuning_flag=tune_encoder)
        assert module.encoder.training == tune_encoder

        attentions, bag_features = module.get_attentions_and_bag_features(instances)
        _assert_existing_gradients_fn(attentions, tuning_flag=tune_pooling)
        _assert_existing_gradients_fn(bag_features, tuning_flag=tune_pooling)
        assert module.aggregation_fn.training == tune_pooling

        bag_logit = module.get_bag_logit(bag_features)
        # bag_logit gradients are required for pooling layer gradients computation, hence
        # "tuning_flag=tune_classifier or tune_pooling"
        _assert_existing_gradients_fn(bag_logit, tuning_flag=tune_classifier or tune_pooling)
        assert module.classifier_fn.training == tune_classifier


@pytest.mark.parametrize("tune_classifier", [False, True])
@pytest.mark.parametrize("tune_pooling", [False, True])
@pytest.mark.parametrize("tune_encoder", [False, True])
def test_training_with_different_finetuning_options(
    tune_encoder: bool, tune_pooling: bool, tune_classifier: bool, tmp_path: Path
) -> None:
    if any([tune_encoder, tune_pooling, tune_classifier]):
        module = DeepMILModule(
            n_classes=6,
            label_column=MockPandaTilesGenerator.ISUP_GRADE,
            encoder_params=get_supervised_imagenet_encoder_params(tune_encoder=tune_encoder),
            pooling_params=get_attention_pooling_layer_params(pool_out_dim=1, tune_pooling=tune_pooling),
            classifier_params=ClassifierParams(tune_classifier=tune_classifier),
        )

        def _assert_existing_gradients(module: nn.Module, tuning_flag: bool) -> None:
            for param in module.parameters():
                if tuning_flag:
                    assert param.grad is not None
                else:
                    assert param.grad is None

        with patch.object(module, "validation_step"):
            trainer = Trainer(max_epochs=1)
            trainer.fit(module, datamodule=_get_datamodule(tmp_path))

            _assert_existing_gradients(module.classifier_fn, tuning_flag=tune_classifier)
            _assert_existing_gradients(module.aggregation_fn, tuning_flag=tune_pooling)
            _assert_existing_gradients(module.encoder, tuning_flag=tune_encoder)


def test_missing_src_checkpoint_with_pretraining_flags() -> None:
    with pytest.raises(ValueError, match=r"You need to specify a source checkpoint, to use a pretrained"):
        _ = MockDeepSMILETilesPanda(tmp_path=Path("foo"), pretrained_classifier=True, pretrained_encoder=True)


@pytest.mark.parametrize("pretrained_classifier", [False, True])
@pytest.mark.parametrize("pretrained_pooling", [False, True])
@pytest.mark.parametrize("pretrained_encoder", [False, True])
def test_init_weights_options(pretrained_encoder: bool, pretrained_pooling: bool, pretrained_classifier: bool) -> None:
    n_classes = 1
    module = DeepMILModule(
        n_classes=n_classes,
        label_column=DEFAULT_LABEL_COLUMN,
        encoder_params=get_supervised_imagenet_encoder_params(),
        pooling_params=get_attention_pooling_layer_params(pool_out_dim=1),
    )
    module.encoder_params.pretrained_encoder = pretrained_encoder
    module.pooling_params.pretrained_pooling = pretrained_pooling
    module.classifier_params.pretrained_classifier = pretrained_classifier

    with patch.object(module, "load_from_checkpoint") as mock_load_from_checkpoint:
        with patch.object(module, "copy_weights") as mock_copy_weights:
            mock_load_from_checkpoint.return_value = MagicMock(n_classes=n_classes)
            module.transfer_weights(Path("foo"))
            assert mock_copy_weights.call_count == sum(
                [int(pretrained_encoder), int(pretrained_pooling), int(pretrained_classifier)]
            )


def _get_tiles_deepmil_module(
    pretrained_encoder: bool = True,
    pretrained_pooling: bool = True,
    pretrained_classifier: bool = True,
    n_classes: int = 3,
    num_layers: int = 2,
    num_heads: int = 1,
    hidden_dim: int = 8,
    transformer_dropout: float = 0.1,
    is_caching: bool = False,
) -> DeepMILModule:
    module = DeepMILModule(
        n_classes=n_classes,
        label_column=MockPandaTilesGenerator.ISUP_GRADE,
        encoder_params=get_supervised_imagenet_encoder_params(is_caching=is_caching),
        pooling_params=get_transformer_pooling_layer_params(num_layers, num_heads, hidden_dim, transformer_dropout),
        classifier_params=ClassifierParams(pretrained_classifier=pretrained_classifier),
    )
    module.encoder_params.pretrained_encoder = pretrained_encoder
    module.pooling_params.pretrained_pooling = pretrained_pooling
    module.classifier_params.pretrained_classifier = pretrained_classifier
    return module


def get_pretrained_module(encoder_val: int = 5, pooling_val: int = 6, classifier_val: int = 7) -> nn.Module:
    module = _get_tiles_deepmil_module()

    def _fix_sub_module_weights(submodule: nn.Module, constant_val: int) -> None:
        for param in submodule.state_dict().values():
            param.data.fill_(constant_val)

    _fix_sub_module_weights(module.encoder, encoder_val)
    _fix_sub_module_weights(module.aggregation_fn, pooling_val)
    _fix_sub_module_weights(module.classifier_fn, classifier_val)

    return module


@pytest.mark.parametrize("pretrained_classifier", [False, True])
@pytest.mark.parametrize("pretrained_pooling", [False, True])
@pytest.mark.parametrize("pretrained_encoder", [False, True])
def test_transfer_weights_same_config(
    pretrained_encoder: bool, pretrained_pooling: bool, pretrained_classifier: bool,
) -> None:
    encoder_val = 5
    pooling_val = 6
    classifier_val = 7
    module = _get_tiles_deepmil_module(pretrained_encoder, pretrained_pooling, pretrained_classifier)
    pretrained_module = get_pretrained_module(encoder_val, pooling_val, classifier_val)

    encoder_random_weights = deepcopy(module.encoder.state_dict())
    pooling_random_weights = deepcopy(module.aggregation_fn.state_dict())
    classification_random_weights = deepcopy(module.classifier_fn.state_dict())

    with patch.object(module, "load_from_checkpoint") as mock_load_from_checkpoint:
        mock_load_from_checkpoint.return_value = pretrained_module
        module.transfer_weights(Path("foo"))

    encoder_transfer_weights = module.encoder.state_dict()
    pooling_transfer_weights = module.aggregation_fn.state_dict()
    classification_transfer_weights = module.classifier_fn.state_dict()

    def _assert_weights_equal(
        random_weights: Dict, transfer_weights: Dict, pretrained_flag: bool, expected_val: int
    ) -> None:
        for r_param_name, t_param_name in zip(random_weights, transfer_weights):
            assert r_param_name == t_param_name, "Param names do not match"
            r_param = random_weights[r_param_name]
            t_param = transfer_weights[t_param_name]
            if pretrained_flag:
                assert torch.equal(t_param.data, torch.full_like(t_param.data, expected_val))
            else:
                assert torch.equal(t_param.data, r_param.data)

    _assert_weights_equal(encoder_random_weights, encoder_transfer_weights, pretrained_encoder, encoder_val)
    _assert_weights_equal(pooling_random_weights, pooling_transfer_weights, pretrained_pooling, pooling_val)
    _assert_weights_equal(
        classification_random_weights, classification_transfer_weights, pretrained_classifier, classifier_val
    )


def test_transfer_weights_different_encoder() -> None:
    module = _get_tiles_deepmil_module(pretrained_encoder=True)
    pretrained_module = _get_tiles_deepmil_module()
    pretrained_module.encoder = IdentityEncoder(tile_size=224)

    with patch.object(module, "load_from_checkpoint") as mock_load_from_checkpoint:
        mock_load_from_checkpoint.return_value = pretrained_module
        with pytest.raises(
            ValueError, match=rf"Submodule {DeepMILSubmodules.ENCODER} has different number of parameters "
        ):
            module.transfer_weights(Path("foo"))


def test_transfer_weights_different_pooling() -> None:
    module = _get_tiles_deepmil_module(num_heads=2, hidden_dim=24, pretrained_pooling=True)
    pretrained_module = _get_tiles_deepmil_module(num_heads=1, hidden_dim=8)

    with patch.object(module, "load_from_checkpoint") as mock_load_from_checkpoint:
        mock_load_from_checkpoint.return_value = pretrained_module
        with pytest.raises(
            ValueError, match=rf"Submodule {DeepMILSubmodules.POOLING} has different number of parameters "
        ):
            module.transfer_weights(Path("foo"))


def test_transfer_weights_different_classifier() -> None:
    module = _get_tiles_deepmil_module(n_classes=4, pretrained_classifier=True)
    pretrained_module = _get_tiles_deepmil_module(n_classes=3)

    with patch.object(module, "load_from_checkpoint") as mock_load_from_checkpoint:
        mock_load_from_checkpoint.return_value = pretrained_module
        with pytest.raises(
            ValueError,
            match=r"Number of classes in pretrained model 3 does not match number of classes in current model 4."
        ):
            module.transfer_weights(Path("foo"))


def test_wrong_encoding_chunk_size() -> None:
    with pytest.raises(
        ValueError, match=r"The encoding chunk size should be at least as large as the maximum bag size"
    ):
        _ = BaseMIL(encoding_chunk_size=1, max_bag_size=4, tune_encoder=True, max_num_gpus=2, pl_sync_batchnorm=True)


@pytest.mark.parametrize("container_type", [DeepSMILETilesPanda,
                                            DeepSMILECrck])
@pytest.mark.parametrize("primary_val_metric", [m for m in MetricsKey])
@pytest.mark.parametrize("maximise_primary_metric", [True, False])
def test_checkpoint_name(container_type: Type[BaseMILTiles], primary_val_metric: MetricsKey,
                         maximise_primary_metric: bool) -> None:

    if container_type in [DeepSMILECrck, DeepSMILETilesPanda]:
        container = container_type(
            encoder_type=ImageNetEncoder.__name__,
            primary_val_metric=primary_val_metric,
            maximise_primary_metric=maximise_primary_metric)
    else:
        container = container_type(
            primary_val_metric=primary_val_metric,
            maximise_primary_metric=maximise_primary_metric)

    metric_optim = "max" if maximise_primary_metric else "min"
    assert container.best_checkpoint_filename == f"checkpoint_{metric_optim}_val_{primary_val_metric.value}"


def test_on_run_extra_val_epoch(mock_panda_tiles_root_dir: Path) -> None:
    container = MockDeepSMILETilesPanda(tmp_path=mock_panda_tiles_root_dir)
    container.setup()
    container.data_module = MagicMock()
    container.create_lightning_module_and_store()
    assert not container.model._on_extra_val_epoch
    assert (
        container.model.outputs_handler.test_plots_handler.plot_options  # type: ignore
        != container.model.outputs_handler.val_plots_handler.plot_options  # type: ignore
    )
    container.on_run_extra_validation_epoch()
    assert container.model._on_extra_val_epoch
    assert (
        container.model.outputs_handler.test_plots_handler.plot_options  # type: ignore
        == container.model.outputs_handler.val_plots_handler.plot_options  # type: ignore
    )


@pytest.mark.parametrize(
    "container_type", [TcgaCrckSSLMIL, TilesPandaSSLMIL, SlidesPandaSSLMIL, SlidesPandaSSLMILBenchmark]
)
def test_ssl_containers_default_checkpoint(container_type: BaseMIL) -> None:
    if container_type == TcgaCrckSSLMIL:
        default_checkpoint = innereye_ssl_checkpoint_crck_4ws
    else:
        default_checkpoint = innereye_ssl_checkpoint_binary
    assert container_type().ssl_checkpoint.checkpoint == default_checkpoint

    container = container_type(ssl_checkpoint=CheckpointParser(TEST_SSL_RUN_ID))
    assert container.ssl_checkpoint.checkpoint != default_checkpoint


@pytest.mark.parametrize("is_caching", [True, False])
def test_reset_encoder_to_identity_encoder(is_caching: bool) -> None:
    model = _get_tiles_deepmil_module(is_caching=is_caching)
    assert isinstance(model.encoder, IdentityEncoder) if is_caching else isinstance(model.encoder, Resnet18)
