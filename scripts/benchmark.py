from pathlib import Path
from typing import Annotated, Optional

import polars as pl
from cyclopts import App, Parameter
from hafnia.dataset.benchmark.benchmark import metric_calculations, run_inference_on_dataset
from hafnia.dataset.dataset_names import SampleField, SplitName
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.experiment import HafniaLogger
from hafnia.experiment.command_builder import auto_save_command_builder_schema
from hafnia.log import user_logger
from hafnia.utils import get_dataset_path_in_hafnia_cloud, is_hafnia_cloud_job

from trainer_classification import utils
from trainer_classification.wrapped_model import InferenceConfig, InferenceModelResNet18

app = App(name="benchmark", help="Benchmark")

""" Benchmarking examples
# Benchmark a trained model archive on the test split
python scripts/benchmark.py --model-path ./.data/experiments/<timestamp>/model/<checkpoint>.zip
"""


@app.default
def main(
    model_path: Annotated[
        str,
        Parameter(
            help=(
                "Path to the trained model archive (.zip). Note: this is ignored when a checkpoint is "
                "available (e.g. a checkpoint selected for the experiment on the Hafnia platform) - the "
                "checkpoint is benchmarked instead of this model."
            )
        ),
    ] = "./pretrained_models/resnet18.zip",
    inference: Annotated[Optional[InferenceConfig], Parameter(help="Inference configuration for the model")] = None,
    split_name: Annotated[str, Parameter(help="Dataset split to run on")] = SplitName.TEST,
    save_annotations: Annotated[
        bool,
        Parameter(help="Write the predictions (annotations only, no image data) to the experiment artifacts folder."),
    ] = True,
    samples: Annotated[
        Optional[int],
        Parameter(help="Limit the number of samples to run on. Useful for faster testing."),
    ] = None,
):
    """Run a model on a Hafnia dataset split and compute classification metrics when ground truth is available.

    Loads the dataset (the hidden dataset when running on the Hafnia platform, otherwise a public
    sample dataset), runs the model on the requested split, and - when the split has ground-truth
    annotations - computes classification metrics and logs them through ``HafniaLogger``. When the
    split has no ground truth the metric step is skipped, so the same script can also be used as a
    pure inference pass. When ``save_annotations`` is set (default), the dataset with predictions
    appended as a new prediction task on each sample is written - annotations only, no image data -
    to the experiment artifacts folder for downstream analysis or visualization.
    """
    inference = inference or InferenceConfig()
    logger = HafniaLogger(project_name="Benchmarking ImageClassification")
    if is_hafnia_cloud_job():  # For hafnia cloud execution
        path_dataset = get_dataset_path_in_hafnia_cloud()  # The path to the full/hidden dataset is returned
        dataset = HafniaDataset.from_path(path_dataset)
    else:
        # The small/public sample dataset is returned by name
        dataset = HafniaDataset.from_name("mnist", version="1.0.0")

    # Prefer a user-selected checkpoint over the configured model when one is available.
    checkpoint_model_path = utils.get_checkpoint_if_available(logger)
    if checkpoint_model_path is not None:
        user_logger.info(f"Using checkpoint '{checkpoint_model_path.name}' instead of '{model_path}'")
        model_path = checkpoint_model_path.as_posix()

    model = InferenceModelResNet18.load_model(model_path, inference_config=inference)
    model.optimize_for_inference()

    dataset_split = dataset.create_split_dataset(split_name=split_name)
    dataset_task_info = dataset.info.get_task_by_primitive(model.task.primitive)

    if samples is not None:
        dataset_split = dataset_split.select_samples(n_samples=samples, seed=42)

    configuration = {
        "model": model.__class__.__name__,
        "model_name": model.task.name,
        "compile": inference.compile,
        "batch_size": inference.batch_size,
        "image_size": model.image_size,
        "dataset": dataset.info.dataset_name,
        "dataset_version": dataset.info.version,
        "model_filename": Path(model_path).name,
        "num_samples": len(dataset_split),
        "split_name": split_name,
    }
    logger.log_configuration(configuration)

    # Run inference on the dataset. Predictions are appended as new tasks on each sample.
    prediction_post_fix = "/predictions"
    dataset_with_predictions = run_inference_on_dataset(
        dataset=dataset_split,
        model=model,
        task_name_prediction_postfix=prediction_post_fix,
    )

    # Save predictions to the experiment artifacts folder (annotations only, drops image-related columns)
    if save_annotations:
        drop_columns = [SampleField.FILE_PATH, SampleField.VIDEO_INFO, SampleField.CAMERA_INFO, SampleField.META]
        dataset_with_predictions.samples = dataset_with_predictions.samples.drop(drop_columns, strict=False)
        dataset_with_predictions.write_annotations(logger._path_artifacts())

    # Skip metric calculation for splits without ground-truth annotations
    gt_column = dataset_task_info.primitive.column_name()
    no_gt_data = dataset_split.samples.select(pl.col(gt_column).list.len()).sum().item() == 0
    if no_gt_data:
        user_logger.warning("No ground-truth annotations found in the selected split. Skipping metric calculation.")
        return logger

    metrics = metric_calculations(
        prediction_dataset=dataset_with_predictions,
        prediction_task_name_postfix=prediction_post_fix,
    )
    for metric_name, metric_value in metrics.items():
        logger.log_metric(metric_name, metric_value, step=0)

    return logger


if __name__ == "__main__":
    # Creates launch schema file for the CLI function 'main'
    path_launch_schema = auto_save_command_builder_schema(main, cli_tool=utils.CLI_TOOL)
    user_logger.info(f"Launch schema saved to: {path_launch_schema}")

    app()
