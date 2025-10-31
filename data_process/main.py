# data_process/main.py
from .config import get_config
from .split_loader import load_split
from .patch_classifier import classify_patches
from .balancer import balance_samples
from .split_saver import save_balanced_split
from pathlib import Path

def main():
    """
    Main function to perform dataset balancing for all splits.
    """
    config = get_config()
    original_split_dir = config['ORIGINAL_SPLIT_DIR']
    label_patch_dir = config['LABEL_PATCH_DIR']
    new_split_dir = config['NEW_SPLIT_DIR']
    ratio = config['NEGATIVE_TO_POSITIVE_RATIO']
    num_workers = config['NUM_WORKERS']
    
    print("--- Starting Dataset Balancing Process ---")
    if not original_split_dir.exists() or not label_patch_dir.exists():
        print(f"Error: Required directories not found. Ensure '{original_split_dir}' and '{label_patch_dir}' exist.")
        return

    new_split_dir.mkdir(parents=True, exist_ok=True)
    print(f"Balanced split files will be saved to: {new_split_dir.absolute()}")
    print(f"Target Negative-to-Positive Ratio: {ratio}:1")
    print(f"Using {num_workers} parallel workers.")

    for split_name in ["train", "val", "test"]:
        print(f"\n--- Processing '{split_name}' set ---")
        
        original_split_file = original_split_dir / f"{split_name}.txt"
        if not original_split_file.exists():
            print(f"Warning: Split file not found, skipping: {original_split_file}")
            continue
        
        basenames = load_split(original_split_file)
        
        label_paths = [str(label_patch_dir / f"{name}.npy") for name in basenames]
        
        positive_patches, negative_patches = classify_patches(label_paths, num_workers)
        
        num_pos = len(positive_patches)
        num_neg = len(negative_patches)
        print(f"Analysis complete. Found:")
        print(f"  - Positive Patches (contains heritage): {num_pos}")
        print(f"  - Negative Patches (all background):    {num_neg}")

        final_patch_list = balance_samples(positive_patches, negative_patches, ratio)
        
        new_split_file = new_split_dir / f"{split_name}.txt"
        save_balanced_split(final_patch_list, new_split_file)
        
        print(f"Balanced split file created: {new_split_file}")
        print(f"  - Total patches: {len(final_patch_list)}")
        print(f"  - Positive count: {num_pos}")
        print(f"  - Negative count: {len(final_patch_list) - num_pos}")

    print("\nDataset balancing complete!")

if __name__ == '__main__':
    main()