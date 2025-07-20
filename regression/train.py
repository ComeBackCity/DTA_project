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
from model_inception import MGraphDTAInception
from my_model import SimpleGATCrossModel
# from my_model_improved import SimpleGATCrossModel
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
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def setup_seed(seed):
    random.seed(seed)                          
    np.random.seed(seed)                       
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = True


def val(model, criterion, dataloader, device):
    model.eval()
    running_loss = AverageMeter()
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Validation", leave=False):
            data = [data_elem.to(device) for data_elem in data]
            y = data[2]
            pred = model(data)
            loss = criterion(pred.view(-1), y.view(-1))
            running_loss.update(loss.item(), y.size(0))
            all_targets.append(y.detach().cpu().numpy().reshape(-1))
            all_preds.append(pred.detach().cpu().numpy().reshape(-1))

    epoch_loss = running_loss.get_average()
    running_loss.reset()

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    epoch_cindex = get_cindex(all_targets, all_preds)

    return epoch_loss, epoch_cindex


# def calculate_learning_rate(batch_size, base_batch_size=16, base_lr=3e-4):
#     """
#     Scales the learning rate linearly with batch size, clamped between 1e-5 and 1e-2.
#     Args:
#         batch_size (int): Current batch size.
#         base_batch_size (int): Reference batch size. Defaults to 16.
#         base_lr (float): Base learning rate for base_batch_size. Defaults to 3e-4.
#     Returns:
#         float: Scaled learning rate.
#     """
#     lr = base_lr * (batch_size / base_batch_size)
#     return max(min(lr, 1e-2), 1e-5)

def calculate_learning_rate(batch_size, base_batch_size=16, base_lr=1e-4):
    """
    Scales the learning rate linearly with batch size, clamped between 1e-5 and 1e-2.
    Args:
        batch_size (int): Current batch size.
        base_batch_size (int): Reference batch size. Defaults to 16.
        base_lr (float): Base learning rate for base_batch_size. Defaults to 1e-4.
    Returns:
        float: Scaled learning rate.
    """
    lr = base_lr * (batch_size / base_batch_size)
    return max(min(lr, 5e-3), 1e-5)


def get_weight_decay(batch_size):
    if batch_size < 1024:
        return 1e-2
    else:
        return 2e-2


class CosineAnnealingWithWarmupWeightDecay:
    def __init__(self, optimizer, min_wd, max_wd, num_warmup_steps, num_training_steps, num_cycles=1):
        self.optimizer = optimizer
        self.min_wd = min_wd
        self.max_wd = max_wd
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.num_cycles = num_cycles
        self.last_step = 0

    def step(self):
        self.last_step += 1
        if self.last_step < self.num_warmup_steps:
            wd = self.min_wd + (self.max_wd - self.min_wd) * (self.last_step / self.num_warmup_steps)
        else:
            progress = (self.last_step - self.num_warmup_steps) / max(1, self.num_training_steps - self.num_warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * ((self.num_cycles * progress) % 1)))
            wd = self.min_wd + (self.max_wd - self.min_wd) * cosine_decay
        for param_group in self.optimizer.param_groups:
            param_group['weight_decay'] = wd
        return wd

    def state_dict(self):
        return {'last_step': self.last_step}

    def load_state_dict(self, state):
        self.last_step = state.get('last_step', 0)
        
        
from torch.optim.lr_scheduler import LRScheduler
from typing import Any, Dict, Optional


# class WarmupExponentialLR(LRScheduler):
#     """
#     Scheduler with linear warmup followed by exponential decay.

#     Args:
#         optimizer (Optimizer): Wrapped optimizer.
#         warmup_steps (int): Number of warmup steps.
#         total_steps (int): Total number of training steps.
#         peak_lr (float): Learning rate at the end of warmup.
#         final_lr (float): Learning rate at the end of training.
#         last_epoch (int): The index of the last epoch. Default: -1.
#     """

#     def __init__(
#         self,
#         optimizer,
#         warmup_steps: int,
#         total_steps: int,
#         peak_lr: float = 1e-3,
#         final_lr: float = 1e-5,
#         last_epoch: int = -1,
#     ):
#         if warmup_steps >= total_steps:
#             raise ValueError("warmup_steps must be less than total_steps")

#         self.warmup_steps = warmup_steps
#         self.total_steps = total_steps
#         self.peak_lr = peak_lr
#         self.final_lr = final_lr

#         self.decay_steps = total_steps - warmup_steps
#         self.decay_rate = final_lr / peak_lr

#         super().__init__(optimizer, last_epoch)

#     def get_lr(self):
#         step = self.last_epoch + 1

#         if step <= self.warmup_steps:
#             scale = step / max(1, self.warmup_steps)
#             return [self.peak_lr * scale for _ in self.optimizer.param_groups]
#         else:
#             decay_progress = (step - self.warmup_steps) / max(1, self.decay_steps)
#             decayed = self.peak_lr * (self.decay_rate ** decay_progress)
#             return [decayed for _ in self.optimizer.param_groups]

#     def step(self, epoch: Optional[int] = None) -> None:
#         """Advance one step. If `epoch` is provided, set the internal step to that value."""
#         if epoch is None:
#             self.last_epoch += 1
#         else:
#             self.last_epoch = epoch
#         for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
#             param_group['lr'] = lr

#     def get_last_lr(self):
#         """Return last computed LR by `step()`."""
#         return [group['lr'] for group in self.optimizer.param_groups]

#     def state_dict(self) -> Dict[str, Any]:
#         """Return state of the scheduler."""
#         return {
#             'warmup_steps': self.warmup_steps,
#             'total_steps': self.total_steps,
#             'peak_lr': self.peak_lr,
#             'final_lr': self.final_lr,
#             'last_epoch': self.last_epoch,
#         }

#     def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
#         """Load state of the scheduler."""
#         self.warmup_steps = state_dict['warmup_steps']
#         self.total_steps = state_dict['total_steps']
#         self.peak_lr = state_dict['peak_lr']
#         self.final_lr = state_dict['final_lr']
#         self.decay_steps = self.total_steps - self.warmup_steps
#         self.decay_rate = self.final_lr / self.peak_lr
#         self.last_epoch = state_dict['last_epoch']

class WarmupExponentialLR(LRScheduler):
    """
    Scheduler with linear warmup followed by exponential decay:
    lr(t) = final_lr + (peak_lr - final_lr) * exp(-decay_rate * (t - warmup_steps)/decay_steps)

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_steps (int): Steps for linear warmup.
        total_steps (int): Total number of training steps.
        peak_lr (float): Learning rate at end of warmup.
        final_lr (float): Final learning rate (asymptotic).
        decay_rate (float): Controls how quickly lr decays (higher = faster decay). Default: 5.0
        last_epoch (int): Index of last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        peak_lr: float = 1e-3,
        final_lr: float = 1e-5,
        decay_rate: float = 5.0,
        last_epoch: int = -1,
    ):
        if warmup_steps >= total_steps:
            raise ValueError("warmup_steps must be less than total_steps")
        if decay_rate <= 0:
            raise ValueError("decay_rate must be positive")

        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.peak_lr = peak_lr
        self.final_lr = final_lr
        self.decay_rate = decay_rate

        self.decay_steps = total_steps - warmup_steps

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1

        if step <= self.warmup_steps:
            scale = step / max(1, self.warmup_steps)
            return [self.peak_lr * scale for _ in self.optimizer.param_groups]
        else:
            decay_progress = (step - self.warmup_steps) / max(1, self.decay_steps)
            lr = self.final_lr + (self.peak_lr - self.final_lr) * math.exp(-self.decay_rate * decay_progress)
            return [lr for _ in self.optimizer.param_groups]

    def step(self, epoch: Optional[int] = None) -> None:
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self) -> Dict[str, Any]:
        return {
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'peak_lr': self.peak_lr,
            'final_lr': self.final_lr,
            'decay_rate': self.decay_rate,
            'last_epoch': self.last_epoch,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.warmup_steps = state_dict['warmup_steps']
        self.total_steps = state_dict['total_steps']
        self.peak_lr = state_dict['peak_lr']
        self.final_lr = state_dict['final_lr']
        self.decay_rate = state_dict['decay_rate']
        self.last_epoch = state_dict['last_epoch']
        self.decay_steps = self.total_steps - self.warmup_steps

def main():
    parser = argparse.ArgumentParser(description='Train MGraphDTA Model with Checkpointing and Schedulers')
    setup_seed(420)

    parser.add_argument('--dataset', required=True, help='Dataset name (e.g., davis or kiba)')
    parser.add_argument('--save_model', action='store_true', help='Whether to save the model')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (if None, will be calculated based on batch size)')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training')
    parser.add_argument('--save_interval', type=int, default=50, help='How many epochs to wait before saving a checkpoint')
    parser.add_argument('--early_stop_epoch', type=int, default=400, help='Number of epochs with no improvement after which training will be stopped')
    parser.add_argument('--weight_decay', type=float, default=None, help='Weight decay (if None, will be set based on batch size)')
    args = parser.parse_args()

    if args.lr is None:
        args.lr = calculate_learning_rate(args.batch_size, base_batch_size=32, base_lr=1e-6)
        print(f"Using automatically calculated learning rate: {args.lr:.2e} for batch size {args.batch_size}")
    if args.weight_decay is None:
        args.weight_decay = get_weight_decay(args.batch_size)
        print(f"Using weight decay: {args.weight_decay:.1e} for batch size {args.batch_size}")

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

    transforms = TupleCompose([
        MaskDrugNodeFeatures(prob=0.2, mask_prob=0.03),
        PerturbDrugNodeFeatures(prob=0.2, noise_std=0.005),
        PerturbDrugEdgeAttr(prob=0.2, noise_std=0.005),
        MaskProteinNodeFeatures(prob=0.2, mask_prob=0.03),
        PerturbProteinNodeFeatures(prob=0.2, noise_std=0.005),
        PerturbProteinEdgeAttr(prob=0.2, noise_std=0.005),
        DropProteinNodes(prob=0.25, drop_prob=0.03),
        DropProteinEdges(prob=0.25, drop_prob=0.03),
        AddRandomProteinEdges(prob=0.2, add_prob=0.005)
    ])
    
    train_set = GNNDataset(DATASET, split='train', transform = transforms)
    val_set = GNNDataset(DATASET, split='valid')
    
    labels = train_set.get_labels()

    

    # Compose drug transforms
    

    
    # sampler = BalancedRegressionBatchSampler2(labels, params.get('batch_size'), minority_ratio=.6, shuffle=True)
    # sampler = BalancedRegressionBatchSampler2(labels, params.get('batch_size'), minority_ratio=.55, shuffle=True)
    # sampler = AdaptiveBalancedSampler(labels, params.get('batch_size'), n_clusters=5, shuffle=True, adaptive_ratio=True)
    train_loader = DataLoader(
        train_set, 
        # batch_sampler=sampler, 
        batch_size=params.get('batch_size'),
        shuffle=True,
        num_workers=12, 
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=True,  # Keep workers alive for faster subsequent epochs
        # multiprocessing_context='fork' if os.name == 'posix' else None,  # Use 'fork' for better performance on Linux,
        # transform = transforms
    )
    
    # train_loader = DataLoader(
    #     train_set, 
    #     batch_size=params.get('batch_size'), 
    #     shuffle=True, 
    #     num_workers=8, 
    #     collate_fn=collate  
    # )

    val_loader = DataLoader(
        val_set, 
        batch_size=params.get('batch_size'), 
        shuffle=False, 
        num_workers=12, 
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=True,  # Keep workers alive for faster subsequent epochs
        # multiprocessing_context='fork' if os.name == 'posix' else None,  # Use 'fork' for better performance on Linux
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SimpleGATCrossModel(
        prot_feat_dim=1204, 
        drug_feat_dim=34, 
        prot_edge_dim=13,
        drug_edge_dim=8,
        hidden_dim=256,
        prot_layers=6,
        drug_layers=3, 
        out_dim=1,
    ).to(device)

    # model = torch.compile(model)
    
    # model = MGraphDTA(
    #     prot_feat_dim=1204, 
    #     drug_feat_dim=34, 
    #     prot_edge_dim=13,
    #     drug_edge_dim=8,
    #     hid_dim=256,
    #     prot_layers=4,
    #     drug_layers=2, 
    #     out_f=1,
    #     pe_dim=16
    # ).to(device)

    # model = MGraphDTAInception(
    #     prot_feat_dim= 52,
    #     drug_feat_dim=34,  # 34
    #     prot_edge_dim=13,  # 13
    #     drug_edge_dim=8,  # 8
    #     hid_dim=128,  # 256
    #     prot_layers=3,  # 4
    #     drug_layers=2,  # 2
    #     out_f=1,
    #     pe_dim=5,
    #     prot_lm_feat_dim=1152,
    #     drug_lm_feat_dim=2048
    # ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=params.get('lr'), weight_decay=args.weight_decay)
    
 
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = len(train_loader) * 5
    
    # Original cosine scheduler
    # lr_scheduler = get_cosine_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=num_warmup_steps,
    #     num_training_steps=num_training_steps,
    #     # num_cycles=args.epochs // 50
    #     num_cycles=args.epochs // 25
    # )
    
    # Alternative exponential scheduler with warmup
    # lr_scheduler = WarmupExponentialLR(
    #     optimizer,
    #     warmup_steps=num_warmup_steps,
    #     total_steps=num_training_steps,
    #     peak_lr=params.get('lr'),
    #     final_lr=1e-6,
    #     decay_rate=3.5,
    # )


    

    # min_wd = 1e-5
    # max_wd = 2e-2
    # wd_scheduler = CosineAnnealingWithWarmupWeightDecay(
    #     optimizer,
    #     min_wd=min_wd,
    #     max_wd=max_wd,
    #     num_warmup_steps=num_warmup_steps,
    #     num_training_steps=num_training_steps,
    #     num_cycles=args.epochs // 50
    # )

    criterion = nn.MSELoss()

    early_stop_epoch = args.early_stop_epoch

    running_loss = AverageMeter()
    running_cindex = AverageMeter()
    running_best_mse = BestMeter("min")

    start_epoch = 0
    best_val_loss = float('inf')
    best_epoch = 0  # Add variable to track best epoch
    break_flag = False

    if args.resume:
        # checkpoint = load_checkpoint(model, optimizer, lr_scheduler, args.resume)
        checkpoint = load_checkpoint(model, optimizer, None, args.resume)
        if checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            best_epoch = checkpoint.get('best_epoch', 0)  # Load best epoch from checkpoint
            
            # Restore loss function
            if checkpoint.get('loss_fn') is not None:
                loss_fn_name = checkpoint['loss_fn']
                loss_fn_params = checkpoint.get('loss_fn_params', {})
                if loss_fn_name == 'MSELoss':
                    criterion = nn.MSELoss(**loss_fn_params)
                elif loss_fn_name == 'L1Loss':
                    criterion = nn.L1Loss(**loss_fn_params)
                # Add more loss functions as needed
                
                if checkpoint.get('criterion_state') is not None:
                    criterion.load_state_dict(checkpoint['criterion_state'])
            
            # Restore sampler states
            if checkpoint.get('sampler_state') is not None:
                sampler_state = checkpoint['sampler_state']
                # if sampler_state.get('train') is not None and hasattr(sampler, 'load_state_dict'):
                #     sampler.load_state_dict(sampler_state['train'])
                #     logger.info("Restored training sampler state")
                
                # if sampler_state.get('val') is not None and hasattr(val_loader.batch_sampler, 'load_state_dict'):
                #     val_loader.batch_sampler.load_state_dict(sampler_state['val'])
                #     logger.info("Restored validation sampler state")
            
            wd_params = checkpoint.get('weight_decay_params')
            if wd_params is not None:
                args.weight_decay = wd_params.get('weight_decay', args.weight_decay)
            
            # Restore scheduler states
            # if checkpoint.get('scheduler_state_dict') is not None:
            #     print(checkpoint['scheduler_state_dict'])
            #     exit()
            #     lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            #     # Calculate the correct step for the scheduler
            #     steps_completed = start_epoch * len(train_loader)
            #     for _ in range(steps_completed):
            #         lr_scheduler.step()
            #     logger.info(f"Restored learning rate scheduler state at step {steps_completed}")
            
            # if checkpoint.get('wd_scheduler_state') is not None:
            #     wd_scheduler.load_state_dict(checkpoint['wd_scheduler_state'])
            #     # Set the last_step to the correct value based on completed steps
            #     steps_completed = start_epoch * len(train_loader)
            #     wd_scheduler.last_step = steps_completed
            #     logger.info(f"Restored weight decay scheduler state at step {steps_completed}")
            
            logger.info(f"Resumed training from epoch {start_epoch} with best_val_loss {best_val_loss:.4f}")
            logger.info(f"Restored loss function: {checkpoint.get('loss_fn', 'Unknown')}")
        else:
            logger.info("Starting training from scratch.")
    else:
        logger.info("Starting training from scratch.")

    writer = SummaryWriter(log_dir=os.path.join(logger.get_log_dir(), "tensorboard"))

    for epoch in range(start_epoch, args.epochs):
        if break_flag:
            break

        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        
        model.train()
        running_loss.reset()
        running_cindex.reset()
        
        # i =0
                
        for data in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", leave=False):    
            # if i < 344:
            #     continue   
            # i+=1  
            data = [data_elem.to(device) for data_elem in data]
            optimizer.zero_grad()
            pred = model(data)
            
            y = data[2]
            loss = criterion(pred.view(-1), y.view(-1))         
            cindex = get_cindex(
                y.detach().cpu().numpy().reshape(-1), 
                pred.detach().cpu().numpy().reshape(-1)
            )
            
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            # lr_scheduler.step()
            # wd_scheduler.step()

            running_loss.update(loss.item(), y.size(0)) 
            running_cindex.update(cindex, y.size(0))

        epoch_loss = running_loss.get_average()
        epoch_cindex = running_cindex.get_average()

        val_loss, val_cindex = val(model, criterion, val_loader, device)

        msg = f"Epoch-{epoch + 1}, Loss-{epoch_loss:.4f}, CIndex-{epoch_cindex:.4f}, Val_Loss-{val_loss:.4f}"
        logger.info(msg)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1  # Update best epoch when we get better loss
            running_best_mse.update(val_loss)
            if save_model_flag:
                save_model_dict(model, logger.get_model_dir(), msg)
                logger.info(f"Saved Best Model at Epoch {epoch + 1} with Val Loss {val_loss:.4f}")

                checkpoint_path = os.path.join(logger.get_model_dir(), f"best_checkpoint_epoch_{epoch + 1}.pth")
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # 'scheduler_state_dict': lr_scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_epoch': best_epoch,  # Save best epoch in checkpoint
                    'criterion_state': criterion.state_dict() if hasattr(criterion, 'state_dict') else None,
                    'weight_decay_params': {
                        'weight_decay': args.weight_decay,
                        'current_epoch': epoch
                    },
                    # 'wd_scheduler_state': wd_scheduler.state_dict(),
                    # 'sampler_state': {
                    #     'train': sampler.state_dict() if hasattr(sampler, 'state_dict') else None,
                    #     'val': val_loader.batch_sampler.state_dict() if hasattr(val_loader.batch_sampler, 'state_dict') else None
                    # },
                    'loss_fn': criterion.__class__.__name__,
                    'loss_fn_params': {
                        'reduction': criterion.reduction if hasattr(criterion, 'reduction') else None
                    }
                }, filename=checkpoint_path)
                logger.info(f"Saved Best Checkpoint at {checkpoint_path}")

        else:
            running_best_mse.update(val_loss)
            if running_best_mse.counter() > early_stop_epoch:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        if save_model_flag:
            latest_checkpoint_path = os.path.join(logger.get_model_dir(), "latest_checkpoint.pth")
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'scheduler_state_dict': lr_scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,  # Save best epoch in latest checkpoint
                'criterion_state': criterion.state_dict() if hasattr(criterion, 'state_dict') else None,
                'weight_decay_params': {
                    'weight_decay': args.weight_decay,
                    'current_epoch': epoch
                },
                # 'wd_scheduler_state': wd_scheduler.state_dict(),
                # 'sampler_state': {
                #     'train': sampler.state_dict() if hasattr(sampler, 'state_dict') else None,
                #     'val': val_loader.batch_sampler.state_dict() if hasattr(val_loader.batch_sampler, 'state_dict') else None
                # },
                'loss_fn': criterion.__class__.__name__,
                'loss_fn_params': {
                    'reduction': criterion.reduction if hasattr(criterion, 'reduction') else None
                }
            }, filename=latest_checkpoint_path)
            logger.info(f"Saved Latest Checkpoint at {latest_checkpoint_path}")

        writer.add_scalar("Train/Loss", epoch_loss, epoch)
        writer.add_scalar("Train/CIndex", epoch_cindex, epoch)
        writer.add_scalar("Val/Loss", val_loss, epoch)
        writer.add_scalar("Val/CIndex", val_cindex, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar("WeightDecay", optimizer.param_groups[0]['weight_decay'], epoch)
        writer.add_scalar("Best_Epoch", best_epoch, epoch)  # Add best epoch to tensorboard

    logger.info(f"Training Completed. Best model was at epoch {best_epoch} with validation loss {best_val_loss:.4f}")
    writer.close()


if __name__ == "__main__":
    main()
    

# %%
