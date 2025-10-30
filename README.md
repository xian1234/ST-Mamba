# ST-Mamba

**Short Description:** Official PyTorch implementation for the paper (under submission): *Decoupling Static Context and Dynamic Change: A Spatio-Temporal Decoder with Time-Averaged Priors for Multi-Decadal Monitoring*. This project provides a novel spatio-temporal segmentation architecture that decouples long-term dynamic priors (from all timesteps) from high-resolution static context (from the last timestep) for fine-grained time-series monitoring.

## 📜 Overview

This project aims to perform fine-grained monitoring and segmentation of specific sites using multi-decadal long-term time-series (LTS) remote sensing imagery. The core challenge in processing such LTS data is how to effectively distinguish between the stable spatial background (**Static Context**) and the subtle changes that occur over time (**Dynamic Change**).

We propose a new spatio-temporal segmentation architecture centered on the idea of **decoupling**. Our model (`SpatioTemporalFusion`) uses a special decoder that reconstructs the segmentation map from two independent information streams:

1.  **Time-Averaged Priors**: A deep feature vector fused from the bottleneck features of the *entire* time series (e.g., 1990-2025). This vector represents a global summary of "dynamic change" and is injected at the decoder's bottleneck.
2.  **Static Context**: High-resolution spatial feature maps extracted *only from the latest timestep* (e.g., 2025). These features are provided to the decoder via skip-connections, ensuring the spatial precision and detail of the final segmentation.

## 🏛️ Methodology

Our core contribution lies in the design of the Decoder. As shown in the figure below, the decoder receives information from two decoupled sources:

1.  A shared-weight 2D U-Net encoder processes each timestep ($T_1, ..., T_N$) independently.
2.  The bottleneck features from all timesteps are collected and fed into a temporal fusion module (like GRU or Mamba) to generate a single, fused vector $\mathbf{v}_{\text{fused}}$ representing "dynamic change."
3.  This $\mathbf{v}_{\text{fused}}$ vector is reshaped to serve as the *initial feature map* for the decoder.
4.  The decoder then upsamples using skip connections from *only the last timestep* ($T_N$), thus combining the high-frequency "static context" with the fused "dynamic prior."

![Methodology Diagram](method.png)

## 📦 Repository Structure
. ├── method.png # The methodology diagram ├── models.py # Contains all neural network architectures ├── train_parally.py # DDP (multi-GPU) training script ├── balance_sample.py # (Preprocessing) Script for dataset balancing ├── data_process.py # (Not provided) Should contain the Dataset class ├── requirements.txt # (Recommended) Dependency list └── README.md # This README file

### Core Script Descriptions

* **`models.py`**: Defines the key models compared in this paper:
    * **`SpatioTemporalFusion`** (aliased as `SpatioTemporalModel` in `train_parally.py`): **Our core model**. Implements the decoupled decoder described above. The temporal module can be selected via the `temporal_module` argument (`'gru'` or `'mamba'`).
    * **`BaselineUNet`**: A baseline model. A standard 2D U-Net that simply concatenates all timesteps in the channel dimension (i.e., `(B, T*C, H, W)`).
    * **`UNet3D`**: A baseline model. A 3D U-Net using (2+1)D convolutions that processes spacetime simultaneously.

* **`train_parally.py`**: The main training file. It uses PyTorch DDP (`DistributedDataParallel`) for efficient multi-GPU training. It handles data loading, model initialization, and the training/validation loops.

* **`balance_sample.py`**: A utility script to address the extreme class imbalance in segmentation tasks. It analyzes all data patches and down-samples the negative samples (all background) by a fixed ratio (`NEGATIVE_TO_POSITIVE_RATIO`) to create a more balanced training set.

## 🚀 How to Use

### 1. Environment Setup

Please ensure you have all necessary dependencies installed. The `train_parally.py` script uses DDP for distributed training. Additionally, if you wish to use Mamba, you must install `mamba_ssm`.

```bash
# Recommended to use a new conda or venv environment
pip install torch torchvision numpy tqdm
pip install mamba_ssm  # If you want to use --temporal-module mamba
```

# TODO: The full ReadMe will be released soon!!!

Before training, run balance_sample.py to create balanced train/val/test split files.
