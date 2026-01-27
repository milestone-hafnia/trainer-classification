from typing import Annotated

import torch
from cyclopts import App, Parameter
from hafnia import utils
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.primitives import Classification
from hafnia.experiment import HafniaLogger
from hafnia.experiment.command_builder import auto_save_command_builder_schema
from hafnia.utils import user_logger

from trainer_classification.train_utils import (
    create_dataloaders,
    create_model,
    train_loop,
)

CLI_TOOL = "cyclopts"

app = App(name="train", help="PyTorch Training")


@app.default
def main(
    project_name: Annotated[
        str, Parameter(help="Project name for the experiment")
    ] = "Trainer ImageClassification Pytorch",
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
    resize_shape = None if resize == 0 else resize
    # Check cuda availability
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        print("CUDA is available. Training on GPU.")
    else:
        print("CUDA is not available. Training on CPU.")

    logger = HafniaLogger(project_name=project_name)
    ckpt_dir = logger.path_model_checkpoints()  # Store checkpoints models here to make them available in the UI.
    model_dir = logger.path_model()  # Store model here to make it available in the UI.

    if utils.is_hafnia_cloud_job():  # In hafnia cloud, the path to the full/hidden dataset is returned
        path_dataset = utils.get_dataset_path_in_hafnia_cloud()
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

    logger.log_configuration(
        {
            "project_name": project_name,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "resize_shape": resize_shape,
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

    num_classes = len(classification_task.class_names)
    model = create_model(num_classes=num_classes)

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
        ckpt_dir=ckpt_dir,
        max_steps_per_epoch=max_steps_per_epoch,
        num_classes=num_classes,
    )

    # Important: Save model in 'model_dir' to make it available in the hafnia platform
    torch.save(model.state_dict(), model_dir / "model.pth")


if __name__ == "__main__":
    # Creates launch schema file for the CLI function 'main'
    path_launch_schema = auto_save_command_builder_schema(main, cli_tool=CLI_TOOL)
    user_logger.info(f"Launch schema saved to: {path_launch_schema}")

    app()
