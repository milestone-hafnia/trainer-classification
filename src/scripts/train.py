from typing import Optional
import numpy as np  # noqa E402 Set MKL_SERVICE_FORCE_INTEL to force it

import argparse
import os
from pathlib import Path
import torch
from train_utils import create_dataloaders, create_model, train_loop
from mdi_python_tools.data import load_dataset
from mdi_python_tools.experiment import MDILogger

from train_utils import flatten_dataset_fields

MDI_EXP_DIR = os.getenv("MDI_EXP_DIR", "/opt/ml/")
IS_LOCAL = os.getenv("MDI_IS_LOCAL", "false").lower() == "true"
DATA_DIR = os.getenv("MDI_DATASET_DIR", f"{MDI_EXP_DIR}input/data/training")
ARTIFACT_DIR = os.getenv("MDI_ARTIFACT_DIR", f"{MDI_EXP_DIR}output/data")
MODEL_DIR = os.getenv("MDI_MODEL_DIR", f"{MDI_EXP_DIR}model")
CKPT_DIR = os.getenv("MDI_CHECKPOINT_DIR", f"{MDI_EXP_DIR}checkpoints")


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument("--dataset", type=str, default=3, help="Dataset being used locally")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train")
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate for optimizer"
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--log_interval", type=int, default=5, help="Interval for logging")
    parser.add_argument("--max_steps_per_epoch", type=int, default=20, help="Max steps per epoch")
    return parser.parse_args()

def main(args: argparse.Namespace):
    artifacts_dir = Path(ARTIFACT_DIR)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = Path(CKPT_DIR)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model_dir = Path(MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)

    logger = MDILogger(artifacts_dir)
    logger.log_configuration(vars(args))
    if IS_LOCAL:
        dataset = load_dataset(args.dataset)
    else:
        dataset = load_dataset(DATA_DIR)
    
    dataset = flatten_dataset_fields(dataset)
    train_dataloader, test_dataloader = create_dataloaders(dataset, args.batch_size)
    
    class_mapping = dataset["train"].features["classification.class_idx"]
    # class_mapping = dataset["train"].features["label"]
    num_classes = len(class_mapping.names)
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
        num_classes=num_classes
    )
    torch.save(model.state_dict(), model_dir / "model.pth")


if __name__ == "__main__":
    args = parse_args()
    main(args)