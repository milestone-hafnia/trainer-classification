import shutil
from pathlib import Path
from typing import Annotated

import torch
from cyclopts import App, Parameter
from hafnia.experiment import HafniaLogger
from hafnia.experiment.command_builder import auto_save_command_builder_schema
from hafnia.log import user_logger

from trainer_classification import utils
from trainer_classification.wrapped_model import InferenceConfig, InferenceModelResNet18

app = App(name="export_onnx", help="Export classification model to ONNX")

""" ONNX export examples
# Export the default pretrained model to ONNX
python scripts/export_onnx.py

# Export a trained checkpoint with a dynamic batch dimension and opset 19
python scripts/export_onnx.py --model-path ./local_stuff/checkpoint.zip --dynamic-batch --opset-version 19
"""


@app.default
def main(
    model_path: Annotated[
        str,
        Parameter(
            help=(
                "Path to the model archive (.zip) to export. Note: this is ignored when a checkpoint "
                "is available (e.g. a checkpoint selected for the experiment on the Hafnia platform) - "
                "the checkpoint is exported instead of this model."
            )
        ),
    ] = "./pretrained_models/resnet18.zip",
    opset_version: Annotated[int, Parameter(help="ONNX opset version to target")] = 17,
    batch_size: Annotated[int, Parameter(help="Static batch size baked into the ONNX graph")] = 1,
    dynamic_batch: Annotated[
        bool,
        Parameter(help="Export with a dynamic batch dimension so the model accepts variable batch sizes at runtime"),
    ] = False,
    verbose: Annotated[bool, Parameter(help="Print export progress information")] = True,
):
    """Export a classification model archive to ONNX format.

    Loads the model from the compressed archive pointed to by ``model_path`` (or a user-selected
    checkpoint when one is available) and exports it to ONNX via ``torch.onnx.export``. The
    resulting ``.onnx`` file is written to the experiment checkpoints folder and copied to the
    model folder so it is collected as a model artifact on the Hafnia platform.

    ``opset_version`` selects the ONNX opset, ``batch_size`` bakes a static batch dimension into
    the graph (use ``dynamic_batch`` for a variable batch dimension instead). The input resolution
    is taken from the model archive's ``image_size`` so it matches training.
    """
    logger = HafniaLogger(project_name="Export ImageClassification ONNX")

    # Prefer a user-selected checkpoint over the configured model when one is available.
    checkpoint_model_path = utils.get_checkpoint_if_available(logger)
    if checkpoint_model_path is not None:
        user_logger.info(f"Using checkpoint '{checkpoint_model_path.name}' instead of '{model_path}'")
        model_path = checkpoint_model_path.as_posix()

    # Load the model without 'optimize_for_inference' (no torch.compile), as ONNX export traces the
    # raw model. The inference settings (InferenceConfig) are required by WrappedModel but unused here.
    wrapped_model = InferenceModelResNet18.load_model(model_path, inference_config=InferenceConfig())

    # Export on CPU so it works the same locally and on the platform; export is a trace, not training.
    device = torch.device("cpu")
    model = wrapped_model.model.to(device).eval()
    image_size = wrapped_model.image_size

    output_dir = Path(logger.path_model_checkpoints())
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / f"{wrapped_model.model.__class__.__name__}.onnx"

    dummy_input = torch.randn(batch_size, 3, image_size, image_size, device=device)
    dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}} if dynamic_batch else None

    configuration = {
        "model_filename": Path(model_path).name,
        "output_dir": output_dir.as_posix(),
        "opset_version": opset_version,
        "batch_size": batch_size,
        "dynamic_batch": dynamic_batch,
        "image_size": image_size,
    }
    logger.log_configuration(configuration)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path.as_posix(),
            input_names=["input"],
            output_names=["output"],
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            # Embed weights in a single self-contained '.onnx' file (no companion '.onnx.data'),
            # so the exported model stays portable when copied to the model folder.
            external_data=False,
            verbose=verbose,
        )
    user_logger.info(f"Exported ONNX model to '{onnx_path}'")

    # Store the model as both a checkpoint and a model artifact.
    path_exported_models = logger.path_model()
    path_exported_models.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_path, path_exported_models / onnx_path.name)
    user_logger.info(f"Copied exported model to '{path_exported_models / onnx_path.name}'")

    return logger


if __name__ == "__main__":
    # Creates launch schema file for the CLI function 'main'
    path_launch_schema = auto_save_command_builder_schema(main, cli_tool=utils.CLI_TOOL)
    user_logger.info(f"Launch schema saved to: {path_launch_schema}")

    app()
