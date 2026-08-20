import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from hafnia.dataset.benchmark.inference_model import ImageType, InferenceModel
from hafnia.dataset.hafnia_dataset_types import ModelInfo, TaskInfo
from hafnia.dataset.primitives import Classification, Primitive
from hafnia.log import user_logger
from pydantic import BaseModel
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import v2

MODEL_CONFIG_NAME = "model_config.json"
DEFAULT_MODEL_NAME = "resnet18"
MODEL_OPTIONS = ["resnet18"]
DEFAULT_IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

PATH_PRETRAINED_MODELS = Path(__file__).parent.parent.parent / "pretrained_models"


def model_from_name(name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    """Build a torchvision classification model with its final layer sized to ``num_classes``.

    When ``pretrained`` is set the backbone is initialized from ImageNet weights; the classification
    head (``fc``) is always replaced *after* construction so it matches the dataset's class count.
    """
    if name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    raise ValueError(f"Model {name} not recognized. Available options: {MODEL_OPTIONS}")


def load_weights_into_model(model: nn.Module, weights_path: Union[str, Path]) -> nn.Module:
    """Load a state dict into ``model``, keeping only parameters whose shape matches.

    This makes a single helper work for both cases the trainer needs:
    - ImageNet pretrained weights, where the 1000-class ``fc.*`` tensors do not match the dataset's
      class count and are therefore skipped (the freshly-initialized head is kept).
    - A genuine same-dataset checkpoint, where every tensor (including ``fc.*``) loads.
    """
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model_state = model.state_dict()
    compatible = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}

    skipped = sorted(set(state) - set(compatible))
    if skipped:
        user_logger.info(f"Skipping weights with incompatible shape (kept freshly initialized): {skipped}")

    model.load_state_dict(compatible, strict=False)
    return model


class ModelConfig(BaseModel):
    name: str
    task: TaskInfo
    model_weight_path: Optional[str]
    image_size: int = DEFAULT_IMAGE_SIZE

    def save_model(self, path_archive: Union[str, Path]):
        """Save the model as a single compressed (zip) archive at ``path_archive``.

        The archive bundles the serialized model config (with a relative weight path) together
        with the weights file. Any existing archive at the destination is overwritten.
        """
        path_archive = Path(path_archive)
        path_archive.parent.mkdir(parents=True, exist_ok=True)

        # The config stores the weights as a relative filename so it resolves inside the archive.
        weight_name = None
        if self.model_weight_path is not None:
            weight_name = Path(self.model_weight_path).name
        config_json = self.model_copy(update={"model_weight_path": weight_name}).model_dump_json(indent=4)

        with zipfile.ZipFile(path_archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(MODEL_CONFIG_NAME, config_json)
            if self.model_weight_path is not None:
                archive.write(self.model_weight_path, arcname=weight_name)

    @staticmethod
    def from_path(path_archive: Union[str, Path], use_weights: bool) -> "ModelConfig":
        path_archive = Path(path_archive)
        # The weights are extracted to a temporary directory that persists for the lifetime of
        # the process, so they remain on disk when the trainer loads them afterwards.
        extract_dir = Path(tempfile.mkdtemp(prefix="trainer_model_"))
        model_config: ModelConfig = _load_config_and_weights(path_archive, extract_dir)

        if use_weights and model_config.model_weight_path is None:
            user_logger.warning(
                f"The specified model '{path_archive}' does not have pretrained weights available, but "
                "'pretrained=True' was set. The model will be trained from scratch."
            )

        if not use_weights and model_config.model_weight_path is not None:
            user_logger.warning(
                f"The specified model '{path_archive}' has pretrained weights available, but "
                "'pretrained=False' was set. The model will be trained from scratch without using "
                "the pretrained weights."
            )
        return model_config

    def as_model(self, num_classes: Optional[int] = None, pretrained: bool = False) -> nn.Module:
        """Build a model from this config, optionally loading its weights.

        ``num_classes`` defaults to the class count in this config's task. Pass it explicitly to size
        the classification head to a different dataset (e.g. fine-tuning ImageNet-pretrained weights
        onto a dataset with a different number of classes); the incompatible head weights are then
        skipped and freshly initialized by ``load_weights_into_model``.
        """
        if num_classes is None:
            num_classes = len(self.task.get_class_names())
        model = model_from_name(self.name, num_classes=num_classes, pretrained=False)
        if pretrained and self.model_weight_path is not None:
            load_weights_into_model(model, self.model_weight_path)
        return model


class InferenceConfig(BaseModel):
    compile: bool = False
    batch_size: int = 1


class InferenceModelResNet18(InferenceModel):
    def __init__(
        self,
        model: nn.Module,
        task: TaskInfo,
        inference_config: InferenceConfig,
        image_size: int = DEFAULT_IMAGE_SIZE,
    ):
        self.model = model
        self.task = task
        self.inference_config = inference_config
        self.image_size = image_size
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # Inference transform mirrors 'create_transforms' but drops the random flip for determinism.
        self._transform = v2.Compose(
            [
                v2.Resize((image_size, image_size)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name=self.model.__class__.__name__, tasks=[self.task])

    def optimize_for_inference(self):
        self.model = self.model.to(self.device).eval()
        # torch.compile is only worthwhile (and reliable) on CUDA; skip it on CPU.
        if self.inference_config.compile and self.device.type == "cuda":
            self.model = torch.compile(self.model)

    @torch.no_grad()
    def predict(self, images: Union[ImageType, List[ImageType]], sample_dict: Optional[dict] = None) -> List[Primitive]:
        image = _ensure_3ch_rgb(images)
        inputs = self._transform(image).unsqueeze(0).to(self.device)
        logits = self.model(inputs)
        probabilities = logits.softmax(dim=1)
        confidence, class_idx = probabilities.max(dim=1)
        class_idx = int(class_idx.item())
        classification = Classification(
            class_idx=class_idx,
            class_name=self.task.classes[class_idx].name,
            confidence=float(confidence.item()),
            ground_truth=False,
            task_name=self.task.name,
        )
        return [classification]

    @staticmethod
    def load_model(
        path_archive: Union[str, Path],
        inference_config: InferenceConfig,
    ) -> "InferenceModelResNet18":
        path_archive = Path(path_archive)
        # Weights are extracted into a temporary directory and loaded into the model while the
        # directory is still alive; the extracted file is no longer needed once the model is built.
        with tempfile.TemporaryDirectory(prefix="trainer_model_") as extract_dir:
            model_config = _load_config_and_weights(path_archive, Path(extract_dir))
            num_classes = len(model_config.task.get_class_names())
            model = model_from_name(model_config.name, num_classes=num_classes, pretrained=False)
            if model_config.model_weight_path is not None:
                load_weights_into_model(model, model_config.model_weight_path)

        return InferenceModelResNet18(
            model=model,
            task=model_config.task,
            inference_config=inference_config,
            image_size=model_config.image_size,
        )


def _load_config_and_weights(path_archive: Path, extract_dir: Path) -> ModelConfig:
    """Read the model config from a zipped model archive and extract its weights into ``extract_dir``.

    The returned config's ``model_weight_path`` is rewritten to the absolute path of the extracted
    weights file, or left as ``None`` when the archive contains no weights.
    """
    with zipfile.ZipFile(path_archive, "r") as archive:
        model_config = ModelConfig.model_validate_json(archive.read(MODEL_CONFIG_NAME))
        if model_config.model_weight_path is not None:
            weight_name = Path(model_config.model_weight_path).name
            archive.extract(weight_name, path=extract_dir)
            model_config.model_weight_path = (extract_dir / weight_name).as_posix()
    return model_config


def _ensure_3ch_rgb(image: ImageType) -> np.ndarray:
    """Normalize a model input image to a 3-channel RGB ``np.ndarray`` (H, W, 3).

    ``HafniaDataset.read_image`` returns RGB arrays, but grayscale datasets (e.g. MNIST) yield 2D
    ``(H, W)`` or single-channel ``(H, W, 1)`` arrays and some images carry an alpha channel. The
    resnet backbone expects exactly 3 channels, so expand/trim accordingly.
    """
    if isinstance(image, np.ndarray):
        array = image
    else:  # PIL.Image or path-like - defer to PIL for decoding
        from PIL import Image

        if isinstance(image, (str, Path)):
            array = np.array(Image.open(image).convert("RGB"))
        else:
            array = np.array(image.convert("RGB"))
        return array

    if array.ndim == 2:  # (H, W) grayscale
        array = np.stack([array] * 3, axis=-1)
    elif array.ndim == 3:
        channels = array.shape[2]
        if channels == 1:
            array = np.repeat(array, 3, axis=2)
        elif channels == 4:  # RGBA -> drop alpha
            array = array[:, :, :3]
    return array
