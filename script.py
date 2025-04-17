import os
from pathlib import Path

# Define the root directory name for the project
project_name = "credit_risk_project"

# Define the directory structure
# Using Path objects for better cross-platform compatibility
base_path = Path(project_name)

# Directories to create
directories = [
    base_path / "data",
    base_path / "models",
    base_path / "notebooks",
    base_path / "src",
]

# Files to create (including empty __init__.py and .gitkeep files)
files = [
    base_path / "README.md",
    base_path / "requirements.txt",
    base_path / "config.py",
    base_path / "main.py",
    base_path / "data" / ".gitkeep", # To ensure the data directory is tracked by git even if empty
    base_path / "models" / ".gitkeep", # To ensure the models directory is tracked by git even if empty
    base_path / "notebooks" / "exploratory_analysis.ipynb",
    base_path / "src" / "__init__.py", # Makes 'src' a Python package
    base_path / "src" / "data_preprocessing.py",
    base_path / "src" / "feature_engineering.py",
    base_path / "src" / "feature_selection.py",
    base_path / "src" / "model_training.py",
    base_path / "src" / "model_evaluation.py",
    base_path / "src" / "utils.py",
]

def create_project_structure(root_dir, dirs_to_create, files_to_create):
    """
    Creates the project directory structure.

    Args:
        root_dir (Path): The root path of the project.
        dirs_to_create (list[Path]): A list of directory paths to create.
        files_to_create (list[Path]): A list of file paths to create.
    """
    print(f"Creating project structure at: {root_dir.resolve()}")

    # Create the root directory if it doesn't exist
    root_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {root_dir}")

    # Create sub-directories
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")

    # Create files
    for file_path in files_to_create:
        # Create parent directories if they don't exist (e.g., for src/__init__.py)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        # Create the file
        file_path.touch(exist_ok=True)
        print(f"Created file: {file_path}")

    print("\nProject structure created successfully!")

# --- Main Execution ---
if __name__ == "__main__":
    create_project_structure(base_path, directories, files)
