# data_process/config.py
from pathlib import Path
import os

def get_config():
    """
    Retrieve the configuration settings for the dataset balancing process.

    Returns:
        dict: A dictionary containing configuration parameters including directories and hyperparameters.
    """
    config = {
        'BASE_DIR': Path("./preprocessed_dataset"),
        'ORIGINAL_SPLIT_DIR': Path("./preprocessed_dataset/splits"),
        'LABEL_PATCH_DIR': Path("./preprocessed_dataset/labels"),
        'NEW_SPLIT_DIR': Path("./preprocessed_dataset/splits_balancedV3"),
        'NEGATIVE_TO_POSITIVE_RATIO': 0.01,
        'NUM_WORKERS': os.cpu_count() or 16
    }
    return config