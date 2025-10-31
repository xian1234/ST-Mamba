# data_process/balancer.py
import random

def balance_samples(positive_patches: list[str], negative_patches: list[str], ratio: float) -> list[str]:
    """
    Balance the dataset by downsampling negative patches according to the specified ratio.

    Args:
        positive_patches (list[str]): List of positive patch basenames.
        negative_patches (list[str]): List of negative patch basenames.
        ratio (float): Negative-to-positive ratio.

    Returns:
        list[str]: Combined list of basenames after balancing and shuffling.
    """
    num_pos = len(positive_patches)
    num_neg = len(negative_patches)
    num_neg_desired = int(num_pos * ratio)
    
    if num_neg > num_neg_desired:
        sampled_neg_patches = random.sample(negative_patches, num_neg_desired)
    else:
        sampled_neg_patches = negative_patches
    
    final_patch_list = positive_patches + sampled_neg_patches
    random.shuffle(final_patch_list)
    
    return final_patch_list