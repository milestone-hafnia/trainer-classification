import hashlib
import importlib
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from hafnia.experiment.command_builder import (
    DEFAULT_ORDER,
    CommandBuilderSchema,
    auto_save_command_builder_schema,
    path_of_function,
    simulate_form_data,
)

from trainer_classification.utils import CLI_TOOL


def file_hash(zip_file, name):
    """Get hash of the uncompressed file content inside a zip archive."""
    with zip_file.open(name) as f:
        return hashlib.md5(f.read()).hexdigest()


def compare_zip_files(zip_path1, zip_path2):
    files_changed = []
    with zipfile.ZipFile(zip_path1, "r") as z1, zipfile.ZipFile(zip_path2, "r") as z2:
        z1_files = sorted(z1.namelist())
        z2_files = sorted(z2.namelist())

        if z1_files != z2_files:
            print("The new trainer package contain new files")
            return False

        for name in z1_files:
            if file_hash(z1, name) != file_hash(z2, name):
                print(f"File content differs: {name}")
                files_changed.append(name)

    if len(files_changed) > 0:
        print(f"The following files have changed: {files_changed}")
        return False

    return True


class _StubLogger:
    """Minimal stand-in for ``HafniaLogger`` exposing only the checkpoints path."""

    def __init__(self, checkpoints_path):
        self._checkpoints_path = Path(checkpoints_path)

    def path_model_checkpoints(self):
        return self._checkpoints_path


def _make_checkpoint_zip(
    archive_path, class_names=("zero", "one"), name="resnet18", image_size=224, real_weights=False
):
    """Build a checkpoint archive (model config + weights) the way ``train.py`` does."""
    from hafnia.dataset.hafnia_dataset_types import TaskInfo
    from hafnia.dataset.primitives import Classification

    from trainer_classification.wrapped_model import ModelConfig, model_from_name

    archive_path = Path(archive_path)
    with tempfile.TemporaryDirectory() as source_dir:
        weights_path = Path(source_dir) / f"{archive_path.stem}.pth"
        if real_weights:
            model = model_from_name(name, num_classes=len(class_names))
            torch.save(model.state_dict(), weights_path)
        else:
            weights_path.write_bytes(b"dummy-weights")
        model_config = ModelConfig(
            name=name,
            task=TaskInfo.from_class_names(primitive=Classification, class_names=list(class_names)),
            model_weight_path=str(weights_path),
            image_size=image_size,
        )
        model_config.save_model(archive_path)
    return archive_path


def test_get_checkpoint_if_available(tmp_path):
    """A checkpoint is discovered only when a ``*.zip`` archive is present, deterministically."""
    from trainer_classification.utils import get_checkpoint_if_available

    checkpoints_dir = tmp_path / "checkpoints"
    logger = _StubLogger(checkpoints_dir)

    # Missing checkpoints directory -> no checkpoint
    assert get_checkpoint_if_available(logger) is None

    # Empty directory -> no checkpoint
    checkpoints_dir.mkdir()
    assert get_checkpoint_if_available(logger) is None

    # Non-archive files are ignored
    (checkpoints_dir / "state.json").write_text("{}")
    assert get_checkpoint_if_available(logger) is None

    # A single checkpoint archive is returned
    _make_checkpoint_zip(checkpoints_dir / "accuracy_0.90_epoch_0.zip")
    assert get_checkpoint_if_available(logger) == checkpoints_dir / "accuracy_0.90_epoch_0.zip"

    # With multiple archives the selection is deterministic (sorted by name)
    _make_checkpoint_zip(checkpoints_dir / "accuracy_0.95_epoch_1.zip")
    assert get_checkpoint_if_available(logger) == checkpoints_dir / "accuracy_0.90_epoch_0.zip"


def test_checkpoint_is_loaded(tmp_path):
    """An available checkpoint is discovered and can be loaded back into a model config."""
    from trainer_classification.utils import get_checkpoint_if_available
    from trainer_classification.wrapped_model import ModelConfig

    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()
    archive_path = _make_checkpoint_zip(
        checkpoints_dir / "accuracy_0.90_epoch_0.zip", class_names=["cat", "dog", "bird"], image_size=64
    )
    logger = _StubLogger(checkpoints_dir)

    checkpoint_model_path = get_checkpoint_if_available(logger)
    assert checkpoint_model_path == archive_path

    # The discovered checkpoint loads, with its weights extracted to an existing file on disk.
    model_config = ModelConfig.from_path(checkpoint_model_path, use_weights=True)
    assert model_config.name == "resnet18"
    assert [c.name for c in model_config.task.classes] == ["cat", "dog", "bird"]
    assert model_config.image_size == 64
    assert Path(model_config.model_weight_path).exists()


def test_wrapped_model_predict(tmp_path):
    """A WrappedModel loaded from an archive predicts a single Classification per image."""
    from hafnia.dataset.primitives import Classification

    from trainer_classification.wrapped_model import InferenceConfig, InferenceModelResNet18

    class_names = ["cat", "dog", "bird"]
    archive_path = _make_checkpoint_zip(
        tmp_path / "model.zip", class_names=class_names, image_size=64, real_weights=True
    )

    model = InferenceModelResNet18.load_model(archive_path, inference_config=InferenceConfig())
    model.optimize_for_inference()

    # An RGB image and a 2D grayscale image (e.g. MNIST) must both work.
    rgb_image = np.random.randint(0, 255, size=(32, 32, 3), dtype=np.uint8)
    gray_image = np.random.randint(0, 255, size=(28, 28), dtype=np.uint8)
    for image in (rgb_image, gray_image):
        predictions = model.predict(image)
        assert len(predictions) == 1
        prediction = predictions[0]
        assert isinstance(prediction, Classification)
        assert prediction.ground_truth is False
        assert 0 <= prediction.class_idx < len(class_names)
        assert prediction.class_name == class_names[prediction.class_idx]
        assert 0.0 <= prediction.confidence <= 1.0


def _script_main(script_name: str):
    """Import the ``main`` function from a script module by name."""
    module = importlib.import_module(f"scripts.{script_name}")
    return module.main


@pytest.mark.parametrize("script_name", ["train", "benchmark", "export_onnx"])
def test_command_builder_schema(script_name: str):
    """Test that the launch schema is up-to-date for each script."""
    main = _script_main(script_name)

    path_function = path_of_function(main)
    path_function_schema = path_function.with_suffix(".schema.json")

    order = 0 if script_name == "train" else DEFAULT_ORDER
    if not path_function_schema.exists():
        auto_save_command_builder_schema(main, cli_tool=CLI_TOOL, order=order)
        pytest.fail("Launch schema file not found. Schema file have been generated. Please run the test again.")

    actual_schema = CommandBuilderSchema.from_function(main, cli_tool=CLI_TOOL, order=order)
    current_schema = CommandBuilderSchema.from_json_file(path_function_schema)

    schema_is_up_to_date = current_schema == actual_schema
    assert schema_is_up_to_date, (
        f"Launch schema in '{path_function_schema}' is outdated. Please delete the schema file "
        f"({path_function_schema}) and rerun this test to regenerate it."
    )


def test_train_command_runs():
    """Test that the train script can be invoked end-to-end via the generated CLI args."""
    from scripts.train import main

    actual_schema = CommandBuilderSchema.from_function(main, cli_tool=CLI_TOOL, order=0)
    form_data = simulate_form_data(main, user_args={"stop_early": "True"})
    cmd_args = actual_schema.command_args_from_form_data(form_data)
    subprocess.run(cmd_args, shell=True, check=True)


def test_train_script():
    """Full training smoke test (CPU-friendly, bounded) producing zip checkpoints."""
    from scripts.train import main

    logger = main(project_name="test_project", epochs=1, max_steps_per_epoch=2, batch_size=16, num_workers=0)
    assert logger is not None
    model_archives = list(Path(logger.path_model()).glob("*.zip"))
    assert len(model_archives) > 0, "No model archives were produced."
    checkpoint_archives = list(Path(logger.path_model_checkpoints()).glob("*.zip"))
    assert len(checkpoint_archives) > 0, "No checkpoint archives were produced."


def test_export_onnx_script():
    """ONNX export writes a model to both the model and checkpoints directories."""
    from scripts.export_onnx import main

    logger = main()
    onnx_models = list(Path(logger.path_model()).glob("*.onnx"))
    assert len(onnx_models) > 0, "No ONNX models were exported."
    checkpoint_models = list(Path(logger.path_model_checkpoints()).glob("*.onnx"))
    assert len(checkpoint_models) > 0, "No ONNX models were exported to the checkpoints directory."
