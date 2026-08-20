from typing import Annotated

import torch
from cyclopts import App, Parameter
from hafnia import utils as hafnia_utils
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.primitives import Classification
from hafnia.experiment import HafniaLogger
from hafnia.experiment.command_builder import auto_save_command_builder_schema
from hafnia.log import user_logger

from trainer_classification import utils
from trainer_classification.train_utils import (
    create_dataloaders,
    train_loop,
)
from trainer_classification.wrapped_model import (
    ModelConfig,
)

CLI_TOOL = utils.CLI_TOOL

app = App(name="train", help="PyTorch Training")


@app.default
def main(
    project_name: Annotated[
        str, Parameter(help="Project name for the experiment")
    ] = "Trainer ImageClassification Pytorch",
    model_path: Annotated[
        str,
        Parameter(
            help=(
                "Path to a compressed (zip) pretrained model used as the training starting point. "
                "Note: this is ignored when a checkpoint is available (e.g. a checkpoint selected for "
                "the experiment on the Hafnia platform) - training resumes from the checkpoint instead "
                "(and '--pretrained' is forced to True)."
            )
        ),
    ] = "./pretrained_models/resnet18.zip",
    pretrained: Annotated[
        bool, Parameter(help="Initialize the model from the pretrained weights in 'model_path'")
    ] = True,
    epochs: Annotated[int, Parameter(help="Number of epochs to train")] = 3,
    learning_rate: Annotated[float, Parameter(help="Learning rate for optimizer")] = 0.001,
    resize: Annotated[int, Parameter(help="Resize image to specified size. Default (0), will drop resizing.")] = 0,
    batch_size: Annotated[int, Parameter(help="Batch size for training")] = 128,
    num_workers: Annotated[int, Parameter(help="Number of workers for DataLoader")] = 8,
    log_interval: Annotated[int, Parameter(help="Interval for logging")] = 5,
    max_steps_per_epoch: Annotated[int, Parameter(help="Max steps per epoch")] = 20,
    stop_early: Annotated[
        bool,
        Parameter(help="Break script before training starts. Can be used to avoid long training times during testing."),
    ] = False,
):
    """Train an image-classification model on a Hafnia dataset.

    Loads the dataset (the hidden dataset when running on the Hafnia platform, otherwise a public
    sample dataset), initializes a model from the compressed model archive pointed to by
    ``model_path`` (optionally using its pretrained weights), and runs the training loop.

    During training, each epoch checkpoint is stored as a standalone compressed Hafnia model
    archive (weights + serialized model config bundled into a single ``.zip``) under both the
    experiment model and checkpoints folders, so they can be used downstream for benchmarking,
    ONNX export, or as a resume point. When a checkpoint is selected for the experiment on the
    platform, training resumes from it instead of ``model_path``.
    """
    resize_shape = None if resize == 0 else resize
    # Check cuda availability
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        print("CUDA is available. Training on GPU.")
    else:
        print("CUDA is not available. Training on CPU.")

    logger = HafniaLogger(project_name=project_name)

    if hafnia_utils.is_hafnia_cloud_job():  # In hafnia cloud, the path to the full/hidden dataset is returned
        path_dataset = hafnia_utils.get_dataset_path_in_hafnia_cloud()
        dataset = HafniaDataset.from_path(path_dataset)
    else:
        # For local execution, a public/sample dataset is returned by name
        dataset = HafniaDataset.from_name("mnist", version="1.0.0")

    classification_task = dataset.info.get_task_by_primitive(Classification)
    has_variable_image_sizes = dataset.info.dataset_name in ["caltech-101", "caltech-256"]

    if has_variable_image_sizes and resize_shape is None:
        resize_shape = 128
        print(
            f"The '{dataset.info.dataset_name}' dataset has variable image sizes. "
            f"A resize transformation ('{resize_shape}x{resize_shape}') is added in the dataset loader\n"
            f"to ensure a consistent input size for the model.\n"
            "You can override resize shape with the '--resize X' argument."
        )

    num_classes = len(classification_task.get_class_names())

    # Resolve the training starting point: a user-selected checkpoint takes precedence over 'model_path'.
    checkpoint_model_path = utils.get_checkpoint_if_available(logger)
    if checkpoint_model_path is not None:
        user_logger.info(f"Using checkpoint '{checkpoint_model_path.name}' as the starting model")
        model_path = checkpoint_model_path.as_posix()
        pretrained = True  # Resuming from a checkpoint always uses its weights, regardless of the '--pretrained' flag.

    model_config = ModelConfig.from_path(model_path, use_weights=pretrained)
    # The classification head is sized to the dataset's class count (not the archive's), so the same
    # archive works both as an ImageNet-pretrained starting point and as a same-dataset resume point.
    model = model_config.as_model(num_classes=num_classes, pretrained=pretrained)

    model_name = model_config.name
    image_size = model_config.image_size if resize_shape is None else resize_shape

    logger.log_configuration(
        {
            "project_name": project_name,
            "model_path": model_path,
            "model_name": model_config.name,
            "pretrained": pretrained,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "resize_shape": resize_shape,
            "image_size": image_size,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "log_interval": log_interval,
            "max_steps_per_epoch": max_steps_per_epoch,
            "dataset_name": dataset.info.dataset_name,
            "dataset_version": dataset.info.version,
            "num_samples": len(dataset),
        }
    )

    train_dataloader, test_dataloader = create_dataloaders(
        dataset=dataset,
        batch_size=batch_size,
        resize=resize_shape,
        num_workers=num_workers,
    )

    if stop_early:
        user_logger.info("Early stopping before training was activated with '--stop_early' flag.")
        return None

    train_loop(
        logger=logger,
        classification_task=classification_task,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        model=model,
        learning_rate=learning_rate,
        epochs=epochs,
        log_interval=log_interval,
        max_steps_per_epoch=max_steps_per_epoch,
        num_classes=num_classes,
        model_name=model_name,
        image_size=image_size,
    )

    return logger


if __name__ == "__main__":
    # Creates launch schema file for the CLI function 'main'
    path_launch_schema = auto_save_command_builder_schema(main, cli_tool=CLI_TOOL, order=0)
    user_logger.info(f"Launch schema saved to: {path_launch_schema}")

    app()
