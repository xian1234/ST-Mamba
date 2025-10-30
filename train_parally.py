import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle

# --- DDP Imports ---
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# --- Import custom modules ---
# Ensure data_process.py and the new models.py are in the same directory
from data_process import HeritageDataset
from models import BaselineUNet, UNet3D, SpatioTemporalFusion as SpatioTemporalModel #SpatioTemporalConvGRU as SpatioTemporalModel 

# --- DDP Helper Functions ---
def setup_ddp():
    """Initializes the distributed process group."""
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()

def is_main_process():
    """Checks if the current process is the main one (rank 0)."""
    return dist.get_rank() == 0

# --- Loss Function ---
class CombinedLoss(nn.Module):
    """Combines BCEWithLogitsLoss and DiceLoss."""
    def __init__(self, weight_bce=0.5, weight_dice=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice

    def forward(self, outputs, targets):
        bce = self.bce_loss(outputs, targets)
        
        probs = torch.sigmoid(outputs)
        intersection = torch.sum(probs * targets)
        union = torch.sum(probs) + torch.sum(targets)
        dice = 1 - (2. * intersection + 1e-6) / (union + 1e-6)
        
        return self.weight_bce * bce + self.weight_dice * dice

# --- Metric Calculation Helpers ---
def calculate_stats_for_batch(preds, masks, threshold=0.5):
    """Calculates the fundamental statistics (TP, FP, FN) for a single batch."""
    preds = (preds > threshold).float()
    
    # --- Stats for Class 1 (Heritage) ---
    tp_1 = (preds * masks).sum()
    fp_1 = (preds * (1 - masks)).sum()
    fn_1 = ((1 - preds) * masks).sum()
    
    # --- Stats for Class 0 (Background) ---
    preds_0, masks_0 = 1 - preds, 1 - masks
    tp_0 = (preds_0 * masks_0).sum()
    fp_0 = (preds_0 * (1 - masks_0)).sum()
    fn_0 = ((1 - preds_0) * masks_0).sum()
    
    return {
        'tp_1': tp_1.item(), 'fp_1': fp_1.item(), 'fn_1': fn_1.item(),
        'tp_0': tp_0.item(), 'fp_0': fp_0.item(), 'fn_0': fn_0.item(),
    }

# --- Training Function ---
def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, model_type):
    model.train()
    total_loss = 0.0
    dataloader.sampler.set_epoch(epoch)
    
    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch}", disable=not is_main_process())

    for images, masks in progress_bar:
        # Input images from dataset are (N, T, C, H, W)
        images, masks = images.to(device), masks.to(device)

        # --- Handle data shapes based on model type ---
        if model_type == 'BaselineUNet':
            # Reshape from (N, T, C, H, W) to (N, T*C, H, W)
            batch_size, time_steps, channels, height, width = images.shape
            images = images.view(batch_size, time_steps * channels, height, width)
        elif model_type == 'UNet3D':
            # Permute from (N, T, C, H, W) to (N, C, T, H, W)
            images = images.permute(0, 2, 1, 3, 4)
        # For SpatioTemporalModel, the expected input is (N, T, C, H, W), so no change needed.

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        if is_main_process():
            progress_bar.set_postfix(loss=f'{loss.item():.4f}')

    # Average loss across all processes for consistent logging
    loss_tensor = torch.tensor([total_loss / len(dataloader)], device=device)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
    return loss_tensor.item()

# --- Validation Function ---
@torch.no_grad()
def evaluate(model, dataloader, criterion, device, model_type):
    model.eval()
    local_loss = 0.0
    local_stats = {'tp_1': 0.0, 'fp_1': 0.0, 'fn_1': 0.0, 'tp_0': 0.0, 'fp_0': 0.0, 'fn_0': 0.0}
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Validating", disable=not is_main_process())

    for images, masks in progress_bar:
        images, masks = images.to(device), masks.to(device)
        
        # --- Handle data shapes consistently with training ---
        if model_type == 'BaselineUNet':
            batch_size, time_steps, channels, height, width = images.shape
            images = images.view(batch_size, time_steps * channels, height, width)
        elif model_type == 'UNet3D':
            images = images.permute(0, 2, 1, 3, 4)

        outputs = model(images)
        loss = criterion(outputs, masks)
        local_loss += loss.item()

        preds = torch.sigmoid(outputs)
        batch_stats = calculate_stats_for_batch(preds, masks)
        for k in local_stats:
            local_stats[k] += batch_stats[k]
        
        num_batches += 1

    # --- Consolidate stats from all GPUs ---
    metrics_tensor = torch.tensor([
        local_loss, num_batches,
        local_stats['tp_1'], local_stats['fp_1'], local_stats['fn_1'],
        local_stats['tp_0'], local_stats['fp_0'], local_stats['fn_0']
    ], device=device)
    
    dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)

    if is_main_process():
        # Unpack the global sums from the tensor
        total_loss, total_batches, total_tp_1, total_fp_1, total_fn_1, \
        total_tp_0, total_fp_0, total_fn_0 = metrics_tensor.tolist()

        # --- Perform final metric calculation ONCE on aggregated stats ---
        epsilon = 1e-6
        iou_1 = (total_tp_1 + epsilon) / (total_tp_1 + total_fp_1 + total_fn_1 + epsilon)
        dice_1 = (2 * total_tp_1 + epsilon) / (2 * total_tp_1 + total_fp_1 + total_fn_1 + epsilon)
        iou_0 = (total_tp_0 + epsilon) / (total_tp_0 + total_fp_0 + total_fn_0 + epsilon)
        dice_0 = (2 * total_tp_0 + epsilon) / (2 * total_tp_0 + total_fp_0 + total_fn_0 + epsilon)
        avg_loss = total_loss / total_batches if total_batches > 0 else 0

        return avg_loss, iou_0, dice_0, iou_1, dice_1
    
    return None, None, None, None, None

# --- Main Worker Function ---
def main(args):
    setup_ddp()
    device = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if is_main_process():
        print(f"--- Starting DDP Training with {world_size} GPUs ---")
        print("Loading preprocessed data...")

    # Load data
    stacked_landsat = np.load(os.path.join(args.data_dir, 'landsat_stackV2.npy'))
    label_data = np.load(os.path.join(args.data_dir, 'labelV2.npy'))
    train_coords = np.load(os.path.join(args.data_dir, 'train_coordsV2.npy'))
    val_coords = np.load(os.path.join(args.data_dir, 'val_coordsV2.npy'))
    with open(os.path.join(args.data_dir, 'norm_stats.pkl'), 'rb') as f:
        norm_stats = pickle.load(f)
    
    all_years = [y for y in range(1990, 2025) if y != 2012]
    years_to_use_list = all_years[-args.years:]
    time_steps = len(years_to_use_list)
    if is_main_process():
        print(f"Training with {time_steps} years of data: {years_to_use_list}")

    # Datasets and Dataloaders
    train_dataset = HeritageDataset(stacked_landsat, label_data, train_coords, args.patch_size, norm_stats, augment=True, years_to_use=years_to_use_list)
    val_dataset = HeritageDataset(stacked_landsat, label_data, val_coords, args.patch_size, norm_stats, augment=False, years_to_use=years_to_use_list)
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=train_sampler, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=val_sampler, pin_memory=True)

    # --- CORRECTED MODEL INITIALIZATION ---
    if is_main_process():
        print(f"Initializing model: {args.model}")
        
    if args.model == 'BaselineUNet':
        input_channels = 4 * time_steps 
        model = BaselineUNet(n_channels=input_channels, n_classes=1)
    elif args.model == 'UNet3D':
        model = UNet3D(n_channels=4, n_classes=1)



    elif args.model == 'SpatioTemporalModel':
        model = SpatioTemporalModel(
            n_channels=4, 
            n_classes=1, 
            # temporal_module=args.temporal_module, # 方案二会用到这个参数
            patch_size=args.patch_size
        )
        # model = SpatioTemporalModel(
        #     n_channels=4, n_classes=1, 
        #     temporal_module=args.temporal_module, 
        #     patch_size=args.patch_size,
        #     time_steps=time_steps
        # )
    else:
        raise ValueError(f"Model '{args.model}' not recognized.")
    
    model.to(device)
    model = DDP(model, device_ids=[device])

    # Optimizer, Loss, and Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr * world_size, weight_decay=1e-5)
    criterion = CombinedLoss()
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.1, verbose=is_main_process())

    best_dice = 0.0
    if is_main_process():
        os.makedirs(args.save_dir, exist_ok=True)

    # --- Main Training Loop ---
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, args.model)
        val_loss, val_iou_0, val_dice_0, val_iou_1, val_dice_1 = evaluate(model, val_loader, criterion, device, args.model)
        
        if is_main_process():
            scheduler.step(val_dice_1)
            
            print(f"\n--- Epoch {epoch}/{args.epochs} Summary ---")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  --- Background (Class 0) ---")
            print(f"    IoU: {val_iou_0:.4f}, Dice: {val_dice_0:.4f}")
            print(f"  --- Heritage (Class 1) ---")
            print(f"    IoU: {val_iou_1:.4f}, Dice: {val_dice_1:.4f}")

            if val_dice_1 > best_dice:
                best_dice = val_dice_1
                model_name_suffix = f"_{args.temporal_module}" if args.model == 'SpatioTemporalModel' else ''
                save_path = os.path.join(args.save_dir, f"{args.model}{model_name_suffix}__best_model.pth")
                torch.save(model.module.state_dict(), save_path)
                print(f"Model saved to {save_path} (Class 1 Dice: {best_dice:.4f})")
        
        dist.barrier()

    if is_main_process():
        print("\n--- Training finished. ---")
    cleanup_ddp()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train segmentation models on heritage site data using DDP.")
    parser.add_argument('--data-dir', type=str, default='./dataset_info', help="Directory with dataset info.")
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help="Directory to save checkpoints.")
    parser.add_argument('--patch-size', type=int, default=128, help="Patch size for training.")
    parser.add_argument('--model', type=str, required=True, choices=['BaselineUNet', 'UNet3D', 'SpatioTemporalModel'], help="Model architecture.")
    parser.add_argument('--temporal-module', type=str, default='gru', choices=['gru', 'mamba'], help="Temporal module for SpatioTemporalModel.")
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs.")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size PER GPU.")
    parser.add_argument('--lr', type=float, default=1e-4, help="Base learning rate.")
    parser.add_argument('--num-workers', type=int, default=8, help="Number of workers for DataLoader.")
    parser.add_argument('--years', type=int, default=34, help="Number of years of data to use (from most recent).")
    
    args = parser.parse_args()
    main(args)