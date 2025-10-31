# data_process/patch_classifier.py
import numpy as np
from pathlib import Path
import concurrent.futures
from tqdm import tqdm
import os

def check_patch_is_positive(label_path_str: str) -> tuple[str, bool]:
    """
    Determine if a label patch contains any positive (heritage) elements.

    Args:
        label_path_str (str): String path to the label .npy file.

    Returns:
        tuple[str, bool]: (basename, is_positive) where is_positive is True if any label > 0.
    """
    label_path = Path(label_path_str)
    basename = label_path.stem
    try:
        label_array = np.load(label_path)
        return basename, np.any(label_array > 0)
    except Exception as e:
        print(f"Error processing file {label_path}: {e}")
        return basename, False

def classify_patches(label_paths: list[str], num_workers: int) -> tuple[list[str], list[str]]:
    """
    Classify patches into positive and negative using multiprocessing.

    Args:
        label_paths (list[str]): List of label file paths as strings.
        num_workers (int): Number of parallel workers.

    Returns:
        tuple[list[str], list[str]]: (positive_patches, negative_patches) lists of basenames.
    """
    positive_patches = []
    negative_patches = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_path = {executor.submit(check_patch_is_positive, path): path for path in label_paths}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_path), total=len(label_paths), desc="Classifying patches"):
            basename, is_positive = future.result()
            if is_positive:
                positive_patches.append(basename)
            else:
                negative_patches.append(basename)
    
    return positive_patches, negative_patches