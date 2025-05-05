# train.py
# %%
import os
import torch
import math
import random
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import argparse
from metrics import get_cindex, ci
from dataset_new import *
from model import MGraphDTA
from utils import (
    AverageMeter, 
    BestMeter, 
    save_checkpoint, 
    load_checkpoint, 
    save_model_dict, 
    load_model_dict
)
from log.train_logger import TrainLogger
from torchmetrics.regression import ConcordanceCorrCoef
import torch_geometric.transforms as T
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from collections import defaultdict
from dataset_new import *


def setup_seed(seed):
    random.seed(seed)                          
    np.random.seed(seed)                       
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True  


def val(model, criterion, dataloader, device):
    model.eval()
    running_loss = AverageMeter()

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Validation", leave=False):
            data = [data_elem.to(device) for data_elem in data]
            y = data[2]
            pred = model(data)
            loss = criterion(pred.view(-1), y.view(-1))
            running_loss.update(loss.item(), y.size(0))

    epoch_loss = running_loss.get_average()
    running_loss.reset()

    return epoch_loss


def adjust_weight_decay(optimizer, epoch, step_size, gamma):
    """
    Adjusts the weight decay parameter in the optimizer's param groups.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer.
        epoch (int): Current epoch.
        step_size (int): Step size for decay.
        gamma (float): Decay factor.
    """
    if epoch % step_size == 0 and epoch > 0:
        for param_group in optimizer.param_groups:
            old_wd = param_group['weight_decay']
            param_group['weight_decay'] *= gamma
            print(f"Adjusted weight_decay from {old_wd} to {param_group['weight_decay']} at epoch {epoch}")


def calculate_learning_rate(batch_size, base_batch_size=512, base_lr=5e-4):
    """
    Calculate appropriate learning rate based on batch size.
    
    Args:
        batch_size (int): Current batch size
        base_batch_size (int): Reference batch size (default: 512)
        base_lr (float): Learning rate for base batch size (default: 5e-4)
    
    Returns:
        float: Scaled learning rate
    """
    # Linear scaling rule
    lr = base_lr * (batch_size / base_batch_size)
    
    # Clamp learning rate to reasonable bounds
    return max(min(lr, 1e-2), 1e-5)


def main():
    parser = argparse.ArgumentParser(description='Train MGraphDTA Model with Checkpointing and Schedulers')
    setup_seed(100)

    # Add arguments
    parser.add_argument('--dataset', required=True, help='Dataset name (e.g., davis or kiba)')
    parser.add_argument('--save_model', action='store_true', help='Whether to save the model')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (if None, will be calculated based on batch size)')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training')
    parser.add_argument('--save_interval', type=int, default=50, help='How many epochs to wait before saving a checkpoint')
    parser.add_argument('--early_stop_epoch', type=int, default=400, help='Number of epochs with no improvement after which training will be stopped')
    parser.add_argument('--wd_step_size', type=int, default=100, help='Step size for weight decay scheduler')
    parser.add_argument('--wd_gamma', type=float, default=0.1, help='Gamma for weight decay scheduler')
    args = parser.parse_args()

    # Calculate learning rate based on batch size if not specified
    if args.lr is None:
        args.lr = calculate_learning_rate(args.batch_size)
        print(f"Using automatically calculated learning rate: {args.lr:.2e} for batch size {args.batch_size}")

    params = dict(
        data_root="data",
        save_dir="save",
        dataset=args.dataset,
        save_model=args.save_model,
        lr=args.lr,
        batch_size=args.batch_size
    )

    logger = TrainLogger(params)
    logger.info(f"Starting training script: {__file__}")

    DATASET = params.get("dataset")
    save_model_flag = params.get("save_model")
    data_root = params.get("data_root")
    fpath = os.path.join(data_root, DATASET)
    
    # Initialize datasets
    train_set = GNNDataset(DATASET, split='train')
    val_set = GNNDataset(DATASET, split='valid')
    
    labels = train_set.get_labels()
    
    # Initialize custom sampler
    # sampler = BalancedRegressionBatchSampler2(labels, params.get('batch_size'), minority_ratio=0.6, shuffle=True)
    sampler = AdaptiveBalancedSampler(labels, params.get('batch_size'), n_clusters=5, shuffle=True, adaptive_ratio=True)
    # Initialize DataLoaders
    # Ensure that 'collate' is defined or imported appropriately
    train_loader = DataLoader(
        train_set, 
        batch_sampler=sampler, 
        num_workers=8, 
        collate_fn=collate  
    )
    
    val_loader = DataLoader(
        val_set, 
        batch_size=params.get('batch_size'), 
        shuffle=False, 
        num_workers=8, 
        collate_fn=collate  
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = MGraphDTA(
        prot_feat_dim=1332, 
        drug_feat_dim=34, 
        prot_edge_dim=12,
        drug_edge_dim=8,
        filt=32, 
        out_f=1
    ).to(device)

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=params.get('lr'), weight_decay=0.01)

    # Define learning rate scheduler
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Define weight decay scheduler
    # This is a custom scheduler; adjust weight decay every 'wd_step_size' epochs by multiplying with 'wd_gamma'
    # Implemented via the adjust_weight_decay function below

    # Define criterion
    criterion = nn.MSELoss()

    # Early stopping parameters
    early_stop_epoch = args.early_stop_epoch

    # Initialize meters
    running_loss = AverageMeter()
    running_cindex = AverageMeter()
    running_best_mse = BestMeter("min")

    # Initialize training state
    start_epoch = 0
    best_val_loss = float('inf')
    break_flag = False

    # Load checkpoint if resume is specified
    if args.resume:
        checkpoint = load_checkpoint(model, optimizer, lr_scheduler, args.resume)
        if checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            logger.info(f"Resumed training from epoch {start_epoch} with best_val_loss {best_val_loss:.4f}")
        else:
            logger.info("Starting training from scratch.")
    else:
        logger.info("Starting training from scratch.")

    model.train()

    for epoch in range(start_epoch, args.epochs):
        if break_flag:
            break

        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        model.train()
        for data in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", leave=False):
            data = [data_elem.to(device) for data_elem in data]
            optimizer.zero_grad()
            pred = model(data)
            
            y = data[2]
            # print(y)
            
            loss = criterion(pred.view(-1), y.view(-1))         
            cindex = get_cindex(
                y.detach().cpu().numpy().reshape(-1), 
                pred.detach().cpu().numpy().reshape(-1)
            )

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # Adjust weight decay if using scheduler
            adjust_weight_decay(optimizer, epoch, args.wd_step_size, args.wd_gamma)

            running_loss.update(loss.item(), y.size(0)) 
            running_cindex.update(cindex, y.size(0))

        epoch_loss = running_loss.get_average()
        epoch_cindex = running_cindex.get_average()
        running_loss.reset()
        running_cindex.reset()

        # Validation
        val_loss = val(model, criterion, val_loader, device)

        msg = f"Epoch-{epoch + 1}, Loss-{epoch_loss:.4f}, CIndex-{epoch_cindex:.4f}, Val_Loss-{val_loss:.4f}"
        logger.info(msg)

        # Check if this is the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            running_best_mse.update(val_loss)
            if save_model_flag:
                # Save the best model
                save_model_dict(model, logger.get_model_dir(), msg)
                logger.info(f"Saved Best Model at Epoch {epoch + 1} with Val Loss {val_loss:.4f}")

                # Additionally, save a comprehensive checkpoint for best model
                checkpoint_path = os.path.join(logger.get_model_dir(), f"best_checkpoint_epoch_{epoch + 1}.pth")
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                    'best_val_loss': best_val_loss
                }, filename=checkpoint_path)
                logger.info(f"Saved Best Checkpoint at {checkpoint_path}")
        else:
            running_best_mse.update(val_loss)
            if running_best_mse.counter() > early_stop_epoch:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        # Always save the latest model
        if save_model_flag:
            latest_checkpoint_path = os.path.join(logger.get_model_dir(), "latest_checkpoint.pth")
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'best_val_loss': best_val_loss
            }, filename=latest_checkpoint_path)
            logger.info(f"Saved Latest Checkpoint at {latest_checkpoint_path}")

    logger.info("Training Completed.")


if __name__ == "__main__":
    main()
