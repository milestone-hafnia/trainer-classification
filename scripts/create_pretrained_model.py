"""Populate the local pretrained-model cache used by the other scripts.

For every entry in ``MODEL_OPTIONS`` this downloads the corresponding torchvision ImageNet
weights, builds an ``InitModelConfig`` with the ImageNet-1k class names, and writes the weights
together with a serialized model config into a single compressed
``pretrained_models/<ModelName>.zip`` archive. Any pre-existing archive for a given model is
overwritten.

Run this once after cloning the repository (or whenever ``MODEL_OPTIONS`` changes) to refresh the
``pretrained_models/`` cache that ``train.py``, ``benchmark.py`` and ``export_onnx.py`` load models
from.
"""

import tempfile
from pathlib import Path

import torch
from hafnia.dataset.hafnia_dataset_types import TaskInfo
from hafnia.dataset.primitives import Classification
from torchvision.models import ResNet18_Weights

from trainer_classification.wrapped_model import (
    DEFAULT_IMAGE_SIZE,
    MODEL_OPTIONS,
    PATH_PRETRAINED_MODELS,
    ModelConfig,
    model_from_name,
)

# Weights enum and ImageNet class names for each supported architecture.
_PRETRAINED_WEIGHTS = {
    "resnet18": ResNet18_Weights.DEFAULT,
}


if __name__ == "__main__":
    for model_name in MODEL_OPTIONS:
        weights = _PRETRAINED_WEIGHTS[model_name]
        class_names = list(weights.meta["categories"])
        num_classes = len(class_names)

        # Build the model with the ImageNet head so the saved weights include 'fc' for 'num_classes'.
        model = model_from_name(model_name, num_classes=num_classes, pretrained=True)

        with tempfile.TemporaryDirectory(prefix="pretrained_") as tmp_dir:
            weights_path = Path(tmp_dir) / f"{model_name}.pth"
            torch.save(model.state_dict(), weights_path)

            pretrained_model_cfg = ModelConfig(
                name=model_name,
                task=TaskInfo.from_class_names(primitive=Classification, class_names=class_names),
                model_weight_path=str(weights_path),
                image_size=DEFAULT_IMAGE_SIZE,
            )
            path_model = PATH_PRETRAINED_MODELS / f"{model_name}.zip"
            pretrained_model_cfg.save_model(path_model)
            print(f"Saved pretrained model archive to '{path_model}'")
