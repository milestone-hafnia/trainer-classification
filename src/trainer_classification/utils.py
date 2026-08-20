from pathlib import Path
from typing import Optional

from hafnia.experiment import HafniaLogger
from hafnia.log import user_logger

CLI_TOOL = "cyclopts"


def get_checkpoint_if_available(logger: HafniaLogger) -> Optional[Path]:
    """Return the path to a user-selected checkpoint archive, or ``None`` if none is available.

    On the Hafnia platform a checkpoint selected for an experiment is placed in the checkpoints
    directory (see ``HafniaLogger.path_model_checkpoints``). A checkpoint is a single compressed
    model archive (see ``InitModelConfig.save_model``), so only ``*.zip`` files are considered.
    """
    checkpoints_folder_path = logger.path_model_checkpoints()

    msg_no_checkpoint = "No checkpoint was found. Using pretrained model."
    if not checkpoints_folder_path.exists():
        user_logger.info(msg_no_checkpoint)
        return None

    checkpoint_files = sorted(checkpoints_folder_path.glob("*.zip"))
    if len(checkpoint_files) == 0:
        user_logger.info(msg_no_checkpoint)
        return None

    if len(checkpoint_files) > 1:
        checkpoint_names = [f.name for f in checkpoint_files]
        user_logger.warning(
            f"Only one checkpoint is expected, but multiple were found: {checkpoint_names}. "
            f"Using '{checkpoint_files[0].name}'."
        )

    return checkpoint_files[0]
