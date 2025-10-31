# data_process/split_loader.py
from pathlib import Path

def load_split(split_file_path: Path) -> list[str]:
    """
    Load the list of patch basenames from the original split file.

    Args:
        split_file_path (Path): Path to the split file (e.g., train.txt).

    Returns:
        list[str]: List of patch basenames.
    """
    if not split_file_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_file_path}")
    
    with open(split_file_path, 'r') as f:
        basenames = [line.strip() for line in f if line.strip()]
    
    return basenames