Decoupling Static Context and Dynamic Change: A Spatio-Temporal Decoder with Time-Averaged Priors for Multi-Decadal Heritage Monitoring

📜 Overview

This project aims to perform fine-grained monitoring and segmentation of urban heritage sites using multi-decadal long-term time-series (LTS) remote sensing imagery. The core challenge in processing such LTS data is how to effectively distinguish between the stable spatial background (Static Context) and the subtle changes that occur over time (Dynamic Change).

We propose a new spatio-temporal segmentation architecture centered on the idea of decoupling. Our model (SpatioTemporalFusion) uses a special decoder that reconstructs the segmentation map from two independent information streams:

Time-Averaged Priors: A deep feature vector fused from the bottleneck features of the entire time series (e.g., 1990-2025). This vector represents a global summary of "dynamic change" and is injected at the decoder's bottleneck.

Static Context: High-resolution spatial feature maps extracted only from the latest timestep (e.g., 2025). These features are provided to the decoder via skip-connections, ensuring the spatial precision and detail of the final segmentation.

🏛️ Methodology

Our core contribution lies in the design of the Decoder. As shown in the figure below, the decoder receives information from two decoupled sources:

A shared-weight 2D U-Net encoder processes each timestep ($T_1, ..., T_N$) independently.

The bottleneck features from all timesteps are collected and fed into a temporal fusion module (like GRU or Mamba) to generate a single, fused vector $\mathbf{v}_{\text{fused}}$ representing "dynamic change."

This $\mathbf{v}_{\text{fused}}$ vector is reshaped to serve as the initial feature map for the decoder.

The decoder then upsamples using skip connections from only the last timestep ($T_N$), thus combining the high-frequency "static context" with the fused "dynamic prior."

📦 Repository Structure

.
├── method.png              # The methodology diagram
├── models.py               # Contains all neural network architectures
├── train_parally.py        # DDP (multi-GPU) training script
├── balance_sample.py       # (Preprocessing) Script for dataset balancing
├── data_process.py         # (Not provided) Should contain the HeritageDataset class
├── requirements.txt        # (Recommended) Dependency list
└── README.md               # This README file


Core Script Descriptions

models.py: Defines the key models compared in this paper:

SpatioTemporalFusion (aliased as SpatioTemporalModel in train_parally.py): Our core model. Implements the decoupled decoder described above. The temporal module can be selected via the temporal_module argument ('gru' or 'mamba').

BaselineUNet: A baseline model. A standard 2D U-Net that simply concatenates all timesteps in the channel dimension (i.e., (B, T*C, H, W)).

UNet3D: A baseline model. A 3D U-Net using (2+1)D convolutions that processes spacetime simultaneously.

train_parally.py: The main training file. It uses PyTorch DDP (DistributedDataParallel) for efficient multi-GPU training. It handles data loading, model initialization, and the training/validation loops.

balance_sample.py: A utility script to address the extreme class imbalance in remote sensing imagery. It analyzes all data patches and down-samples the negative samples (all background) by a fixed ratio (NEGATIVE_TO_POSITIVE_RATIO) to create a more balanced training set.

🚀 How to Use

1. Environment Setup

Please ensure you have all necessary dependencies installed. The train_parally.py script uses DDP for distributed training. Additionally, if you wish to use Mamba, you must install mamba_ssm.

# Recommended to use a new conda or venv environment
pip install torch torchvision numpy tqdm
pip install mamba_ssm  # If you want to use --temporal-module mamba


2. Data Preparation

This repository assumes that data has been prepared in the following format and is located in the --data-dir (default: ./dataset_info):

landsat_stackV2.npy: Numpy array containing all years of image data.

labelV2.npy: Numpy array containing the label data.

train_coordsV2.npy / val_coordsV2.npy: Arrays containing coordinates for training/validation patches.

norm_stats.pkl: A pickle file containing the mean and std for normalization.

Data Balancing (Optional but Recommended):

Before training, run balance_sample.py to create balanced train/val/test split files.

# Assuming your original split files (train.txt, val.txt) are in ./preprocessed_dataset/splits/
# Assuming your label patches (patch_xxxx.npy) are in ./preprocessed_dataset/labels/

python balance_sample.py

# This script will output new, balanced split files to ./preprocessed_dataset/splits_balancedV3/
# You may need to adjust your data_process.py (not provided) to read these new files.


3. Model Training

Use torchrun (or torch.distributed.launch) to start DDP training. You must specify the model to train using the --model argument.

Assuming you have 4 GPUs:

# Set the number of GPUs per node
GPUS_PER_NODE=4

# Example 1: Train our core model (SpatioTemporalModel) with GRU
torchrun --nproc_per_node=$GPUS_PER_NODE train_parally.py \
    --model SpatioTemporalModel \
    --temporal-module gru \
    --batch-size 16 \
    --epochs 100 \
    --lr 1e-4 \
    --save-dir ./checkpoints/st_gru

# Example 2: Train our core model (SpatioTemporalModel) with Mamba
torchrun --nproc_per_node=$GPUS_PER_NODE train_parally.py \
    --model SpatioTemporalModel \
    --temporal-module mamba \
    --batch-size 16 \
    --epochs 100 \
    --lr 1e-4 \
    --save-dir ./checkpoints/st_mamba

# Example 3: Train the BaselineUNet (2D U-Net + Channel Concat)
torchrun --nproc_per_node=$GPUS_PER_NODE train_parally.py \
    --model BaselineUNet \
    --batch-size 16 \
    --epochs 100 \
    --lr 1e-4 \
    --save-dir ./checkpoints/baseline_unet

# Example 4: Train the Baseline UNet3D ((2+1)D U-Net)
torchrun --nproc_per_node=$GPUS_PER_NODE train_parally.py \
    --model UNet3D \
    --batch-size 8 \
    --epochs 100 \
    --lr 1e-4 \
    --save-dir ./checkpoints/unet_3d


Key Command-Line Arguments

--model: (Required) The model to train. Choices: ['BaselineUNet', 'UNet3D', 'SpatioTemporalModel'].

--temporal-module: (For SpatioTemporalModel only) The temporal module to use. Choices: ['gru', 'mamba'].

--data-dir: Directory containing the .npy and .pkl dataset files.

--save-dir: Directory to save model checkpoints.

--batch-size: Batch size per GPU.

--epochs: Total number of training epochs.

--lr: Base learning rate (the script automatically scales this by world_size).

--years: Number of recent N years of data to use for training.

Citation

If this work is helpful for your research, please consider citing our paper (currently under review) once it is published:

@article{YourName_2025_Decoupling,
  title={Decoupling Static Context and Dynamic Change: A Spatio-Temporal Decoder with Time-Averaged Priors for Multi-Decadal Heritage Monitoring},
  author={Your Name and Co-authors},
  journal={TBD},
  year={2025}
}


License

This project is licensed under the MIT License.
