import hashlib
import shutil
import zipfile
from pathlib import Path

from hafnia import utils
from hafnia.platform.builder import validate_trainer_package_format


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


def test_trainer_zip_outdated(tmp_path: Path):
    """Test the trainer package generation and validation."""
    path_trainer_zip_actual = tmp_path / "trainer.zip"
    path_trainer_zip_expected = Path(__file__).parents[1] / "trainer.zip"
    path_source = Path("./.")
    utils.archive_dir(path_source, output_path=path_trainer_zip_actual)
    validate_trainer_package_format(path_trainer_zip_actual)

    if not path_trainer_zip_expected.exists():
        shutil.copy2(path_trainer_zip_actual, path_trainer_zip_expected)
        assert 0 == 1, "Trainer package zip file not found. Package have been regenerated. Please run the test again."

    assert_msg = (
        "Trainer package contents differ. Please check the differences. "
        f"Delete the '{path_trainer_zip_expected}' file to regenerate it or "
        "run 'hafnia trainer create-zip .' in terminal to update the recipe."
    )
    assert compare_zip_files(path_trainer_zip_actual, path_trainer_zip_expected), assert_msg
