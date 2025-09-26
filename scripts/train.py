import argparse

import torch
from hafnia import utils
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.experiment import HafniaLogger

from trainer_classification.train_utils import create_dataloaders, create_model, train_loop


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument("--dataset", type=str, default="mnist", help="Dataset being used locally")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--resize", type=int, default=None, help="Resize image to specified size")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for DataLoader")
    parser.add_argument("--log_interval", type=int, default=5, help="Interval for logging")
    parser.add_argument("--max_steps_per_epoch", type=int, default=20, help="Max steps per epoch")

    return parser.parse_args()


def main(args: argparse.Namespace):
    # Check cuda availability
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        print("CUDA is available. Training on GPU.")
    else:
        print("CUDA is not available. Training on CPU.")
    logger = HafniaLogger()
    ckpt_dir = logger.path_model_checkpoints()  # Store checkpoints models here to make them available in the UI.
    model_dir = logger.path_model()  # Store model here to make it available in the UI.

    logger.log_configuration(vars(args))  # Log the configuration to the UI

    if utils.is_hafnia_cloud_job():  # In hafnia cloud, the path to the full/hidden dataset is returned
        path_dataset = utils.get_dataset_path_in_hafnia_cloud()
        dataset = HafniaDataset.from_path(path_dataset)
    elif args.dataset:  # For local execution, a public/sample dataset is returned by name
        dataset = HafniaDataset.from_name(args.dataset)
    else:
        raise ValueError("You must provide a dataset name with the '--dataset DATASET_NAME' argument")

    classification_task = dataset.info.tasks[0]
    dataset_name = dataset.info.dataset_name
    has_variable_image_sizes = dataset_name in ["caltech-101", "caltech-256"]
    resize_shape = args.resize
    if has_variable_image_sizes and resize_shape is None:
        resize_shape = 128
        print(
            f"The '{dataset_name}' dataset has variable image sizes. "
            f"A resize transformation ('{resize_shape}x{resize_shape}') is added in the dataset loader\n"
            f"to ensure a consistent input size for the model.\n"
            "You can override resize shape with the '--resize X' argument."
        )

    train_dataloader, test_dataloader = create_dataloaders(
        dataset=dataset,
        batch_size=args.batch_size,
        resize=resize_shape,
        num_workers=args.num_workers,
    )

    num_classes = len(classification_task.class_names)
    model = create_model(num_classes=num_classes)
    train_loop(
        logger=logger,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        model=model,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        log_interval=args.log_interval,
        ckpt_dir=ckpt_dir,
        max_steps_per_epoch=args.max_steps_per_epoch,
        num_classes=num_classes,
    )

    # Important: Save model in 'model_dir' to make it available in the hafnia platform
    torch.save(model.state_dict(), model_dir / "model.pth")

    return logger


if __name__ == "__main__":
    args = parse_args()
    main(args)
