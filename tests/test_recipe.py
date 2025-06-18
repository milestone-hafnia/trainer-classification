import hashlib
import shutil
import zipfile
from pathlib import Path

from hafnia import utils
from hafnia.platform.builder import validate_recipe_format


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
            print("The new recipe contain a different set of files.")
            return False

        for name in z1_files:
            if file_hash(z1, name) != file_hash(z2, name):
                print(f"File content differs: {name}")
                files_changed.append(name)

    if len(files_changed) > 0:
        print(f"The following files have changed: {files_changed}")
        return False

    return True


def test_recipe_outdated(tmp_path: Path):
    """Test the recipe generation and validation."""
    path_recipe_actual = tmp_path / "recipe.zip"
    path_recipe_expected = Path(__file__).parents[1] / "recipe.zip"
    path_source = Path("./.")
    utils.archive_dir(path_source, output_path=path_recipe_actual)
    validate_recipe_format(path_recipe_actual)

    if not path_recipe_expected.exists():
        shutil.copy2(path_recipe_actual, path_recipe_expected)
        assert 0 == 1, "Recipe file not found. Recipe file have been regenerated. Please run the test again."

    assert_msg = (
        "Recipe file contents differ. Please check the differences. "
        f"Delete the '{path_recipe_expected}' file to regenerate it or "
        "run 'hafnia experiment create_recipe' in terminal to update the recipe."
    )
    assert compare_zip_files(path_recipe_actual, path_recipe_expected), assert_msg
