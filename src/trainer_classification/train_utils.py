import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from hafnia.dataset import torch_helpers
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.hafnia_dataset_types import TaskInfo
from hafnia.experiment import HafniaLogger
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, Metric
from torchvision.transforms import v2

from trainer_classification.wrapped_model import DEFAULT_MODEL_NAME, ModelConfig, model_from_name


def create_transforms(resize: Optional[int] = None) -> v2.Compose:
    """
    Creates a composition of image transformations for data augmentation and normalization

    Returns:
        transforms.Compose: A composed transform for image preprocessing.
    """
    transforms = [
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    if resize is not None:
        transforms.insert(0, v2.Resize((resize, resize)))
    return v2.Compose(transforms)


def create_dataloaders(
    dataset: HafniaDataset, batch_size: int, num_workers: int = 4, resize: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates training and testing DataLoaders.

    Args:
        data_root (str): Root directory of the dataset.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and testing DataLoaders.
    """
    transforms = create_transforms(resize=resize)

    train_split = dataset.create_split_dataset("train")
    test_split = dataset.create_split_dataset("test")

    # Strictly using our helper functions is not necessary for an Image classification dataset.
    # It is enough to simply do:
    #   train_split.set_transform(transforms)
    #   test_split.set_transform(transforms)
    #   collate_fn = None
    # However, we use helpers here for demonstrational purposes as it will work across datasets with
    # different tasks such as object detection, segmentation, etc.
    train_split = torch_helpers.TorchvisionDataset(train_split, transforms=transforms, keep_metadata=True)
    test_split = torch_helpers.TorchvisionDataset(test_split, transforms=transforms, keep_metadata=True)
    collate_fn = torch_helpers.TorchVisionCollateFn()

    train_loader = DataLoader(
        train_split, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_split, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn
    )
    return train_loader, test_loader


def create_model(num_classes: int, pretrained: bool = False) -> nn.Module:
    """
    Creates and returns the default classification model adjusted for the specified number of classes.

    Args:
        num_classes (int): Number of output classes.
        pretrained (bool): Initialize the backbone from ImageNet weights.

    Returns:
        nn.Module: The classification model with its head sized to ``num_classes``.
    """
    return model_from_name(DEFAULT_MODEL_NAME, num_classes=num_classes, pretrained=pretrained)


def run_train_epoch(
    epoch: int,
    classification_task: TaskInfo,
    dataloader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    criterion: nn.Module,
    metrics: Metric,
    device: torch.device,
    ml_logger: HafniaLogger,
    log_interval: int,
    max_steps_per_epoch: int,
) -> Dict[str, Union[float, int]]:
    """
    Runs one epoch of training.

    Args:
        epoch (int): Current epoch number.
        dataloader (DataLoader): Training DataLoader.
        model (nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): Optimizer.
        criterion (nn.Module): Loss function.
        metrics (Metric): Metrics calculator.
        device (torch.device): Computation device.
        ml_logger (HafniaLogger): Logger for metrics.
        log_interval (int): Interval for logging.
        max_steps_per_epoch (int): Maximum steps per epoch.

    Returns:
        Dict[str, float]: Dictionary containing average loss and accuracy.
    """
    model.train()
    metrics.reset()
    epoch_loss = 0.0
    total_samples = 0

    step = epoch * len(dataloader)

    class_idx_name = f"{classification_task.primitive.column_name()}.{classification_task.name}.class_idx"
    for i, batch in enumerate(dataloader):
        inputs, targets_and_metadata = batch
        targets = targets_and_metadata[class_idx_name]
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)
        metrics.update(outputs.softmax(dim=1), targets)

        if i % log_interval == 0:
            avg_loss = epoch_loss / total_samples
            accuracy = metrics.compute().item()
            ml_logger.log_scalar("train/loss", avg_loss, step=step)

        step += 1
        if i >= max_steps_per_epoch:
            print(f"Max steps per train epoch reached: {max_steps_per_epoch}")
            break

    avg_loss = epoch_loss / total_samples
    accuracy = metrics.compute().item()
    ml_logger.log_metric(name="train/accuracy", value=accuracy, step=step)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "step": step,
    }


@torch.no_grad()
def run_eval(
    step: int,
    classification_task: TaskInfo,
    dataloader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    metrics: Metric,
    device: torch.device,
    ml_logger: HafniaLogger,
):
    """
    Runs evaluation on the test dataset.

    Args:
        step (int): Current step number.
        classification_task: TaskInfo,
        dataloader (DataLoader): Evaluation DataLoader.
        model (nn.Module): The model to evaluate.
        criterion (nn.Module): Loss function.
        metrics (Metric): Metrics calculator.
        device (torch.device): Computation device.
        ml_logger (HafniaLogger): Logger for metrics.

    Returns:
        Dict[str, float]: Dictionary containing average loss and accuracy.
    """
    model.eval()
    metrics.reset()
    epoch_loss = 0.0
    total_samples = 0
    class_idx_name = f"{classification_task.primitive.column_name()}.{classification_task.name}.class_idx"
    for batch in dataloader:
        inputs, targets_and_metadata = batch
        targets = targets_and_metadata[class_idx_name]
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        epoch_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)
        metrics.update(outputs.softmax(dim=1), targets)

    avg_loss = epoch_loss / total_samples
    accuracy = metrics.compute().item()

    ml_logger.log_scalar("test/loss", avg_loss, step)
    ml_logger.log_metric("test/accuracy", accuracy, step)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
    }


def train_loop(
    logger: HafniaLogger,
    classification_task: TaskInfo,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    model: nn.Module,
    learning_rate: float,
    epochs: int,
    log_interval: int,
    max_steps_per_epoch: int,
    num_classes: int,
    model_name: str,
    image_size: int,
):
    """
    Main training loop.

    Args:
        logger (HafniaLogger): Logger for metrics.
        classification_task (TaskInfo): Information about the classification task.
        train_dataloader (DataLoader): Training DataLoader.
        test_dataloader (DataLoader): Testing DataLoader.
        model (nn.Module): The model to train.
        learning_rate (float): Learning rate for the optimizer.
        epochs (int): Number of epochs to train.
        log_interval (int): Interval for logging metrics.
        max_steps_per_epoch (int): Maximum steps per epoch.
        num_classes (int): Number of classes in the classification task.
        model_name (str): Architecture name stored in the saved model archives (e.g. 'resnet18').
        image_size (int): Inference image size stored in the saved model archives.
    """
    # Each epoch checkpoint is stored - as a single compressed model archive (weights + config) - in
    # both the model and checkpoints folders, so the platform collects them as model artifacts and
    # resumable checkpoints.
    model_dir = logger.path_model()
    ckpt_dir = logger.path_model_checkpoints()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    model.train()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    metrics = Accuracy(task="multiclass", num_classes=num_classes).to(device)

    for epoch in range(epochs):
        results = run_train_epoch(
            epoch=epoch,
            classification_task=classification_task,
            dataloader=train_dataloader,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            metrics=metrics,
            device=device,
            ml_logger=logger,
            log_interval=log_interval,
            max_steps_per_epoch=max_steps_per_epoch,
        )
        eval_metrics = run_eval(
            step=results["step"],
            classification_task=classification_task,
            dataloader=test_dataloader,
            model=model,
            criterion=criterion,
            metrics=metrics,
            device=device,
            ml_logger=logger,
        )
        ckpt_stem = f"accuracy_{eval_metrics['accuracy']:.2f}_epoch_{epoch}"
        save_model_archive(
            model=model,
            model_name=model_name,
            classification_task=classification_task,
            image_size=image_size,
            stem=ckpt_stem,
            model_dir=model_dir,
            ckpt_dir=ckpt_dir,
        )


def save_model_archive(
    model: nn.Module,
    model_name: str,
    classification_task: TaskInfo,
    image_size: int,
    stem: str,
    model_dir: Path,
    ckpt_dir: Path,
) -> None:
    """Save ``model`` as a single compressed model archive (weights + config) to both folders.

    The archive bundles the model weights with a serialized ``ModelConfig`` so it can be loaded back
    for benchmarking, ONNX export, or as a resume point. It is written to ``model_dir`` (collected as
    a model artifact) and ``ckpt_dir`` (discovered as a resumable checkpoint).
    """
    with tempfile.TemporaryDirectory(prefix="trainer_ckpt_") as tmp_dir:
        weights_path = Path(tmp_dir) / f"{stem}.pth"
        torch.save(model.state_dict(), weights_path)
        archive_config = ModelConfig(
            name=model_name,
            task=classification_task,
            model_weight_path=str(weights_path),
            image_size=image_size,
        )
        archive_config.save_model(model_dir / f"{stem}.zip")
        archive_config.save_model(ckpt_dir / f"{stem}.zip")
