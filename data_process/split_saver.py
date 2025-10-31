# data_process/split_saver.py
from pathlib import Path

def save_balanced_split(final_patch_list: list[str], output_file_path: Path):
    """
    Save the balanced list of patch basenames to a new split file.

    Args:
        final_patch_list (list[str]): List of basenames to save.
        output_file_path (Path): Path to the output split file.
    """
    with open(output_file_path, 'w') as f:
        for item in final_patch_list:
            f.write(f"{item}\n")