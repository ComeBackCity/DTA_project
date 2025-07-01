# train.py
# %%
import os
import torch
import math
import random
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import argparse
import itertools

from torch_geometric.loader import DataLoader
from transformers import get_cosine_schedule_with_warmup
from torchmetrics.regression import ConcordanceCorrCoef
from tqdm import tqdm
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

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


def calculate_learning_rate(batch_size, base_batch_size=512, base_lr=5e-4):
    lr = base_lr * (batch_size / base_batch_size)
    return max(min(lr, 1e-2), 1e-5)


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


def main():
    parser = argparse.ArgumentParser(description='Train MGraphDTA Model with Checkpointing and Schedulers')
    setup_seed(100)

    parser.add_argument('--dataset', required=True, help='Dataset name (e.g., davis or kiba)')
    parser.add_argument('--save_model', action='store_true', help='Whether to save the model')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (if None, will be calculated based on batch size)')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs')
    parser.add_argument('--steps_per_epoch', type=int, default=100, help='Fixed number of steps per epoch')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training')
    parser.add_argument('--save_interval', type=int, default=50, help='How many epochs to wait before saving a checkpoint')
    parser.add_argument('--early_stop_epoch', type=int, default=400, help='Number of epochs with no improvement after which training will be stopped')
    parser.add_argument('--weight_decay', type=float, default=None, help='Weight decay (if None, will be set based on batch size)')
    args = parser.parse_args()

    if args.lr is None:
        args.lr = calculate_learning_rate(args.batch_size)
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
    
    train_set = GNNDataset(DATASET, split='train')
    val_set = GNNDataset(DATASET, split='valid')
    
    labels = train_set.get_labels()
    
    sampler = AdaptiveBalancedSampler(labels, params.get('batch_size'), n_clusters=5, shuffle=True, adaptive_ratio=True)
    train_loader = DataLoader(train_set, batch_sampler=sampler, num_workers=8, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=params.get('batch_size'), shuffle=False, num_workers=8, collate_fn=collate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = MGraphDTA(
        prot_feat_dim=1204, 
        drug_feat_dim=34, 
        prot_edge_dim=13,
        drug_edge_dim=8,
        hid_dim=256,
        prot_layers=4,
        drug_layers=2, 
        out_f=1,
        pe_dim=16
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=params.get('lr'), weight_decay=args.weight_decay)

    num_training_steps = args.epochs * args.steps_per_epoch
    num_warmup_steps = num_training_steps // 10
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=args.epochs // 50
    )

    wd_scheduler = CosineAnnealingWithWarmupWeightDecay(
        optimizer,
        min_wd=1e-5,
        max_wd=2e-2,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=args.epochs // 50
    )

    criterion = nn.MSELoss()
    early_stop_epoch = args.early_stop_epoch

    running_loss = AverageMeter()
    running_cindex = AverageMeter()
    running_best_mse = BestMeter("min")
    start_epoch = 0
    best_val_loss = float('inf')
    break_flag = False

    if args.resume:
        checkpoint = load_checkpoint(model, optimizer, lr_scheduler, args.resume)
        if checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            if checkpoint.get('criterion_state') is not None:
                criterion.load_state_dict(checkpoint['criterion_state'])
            if checkpoint.get('wd_scheduler_state') is not None:
                wd_scheduler.load_state_dict(checkpoint['wd_scheduler_state'])
            logger.info(f"Resumed training from epoch {start_epoch} with best_val_loss {best_val_loss:.4f}")
        else:
            logger.info("Starting training from scratch.")
    else:
        logger.info("Starting training from scratch.")

    writer = SummaryWriter(log_dir=os.path.join(logger.get_log_dir(), "tensorboard"))

    # Calculate number of iterations needed
    num_iter = math.ceil((args.epochs * args.steps_per_epoch) / len(train_loader))
    global_step = 0
    global_epoch = 0

    # Create epoch progress bar
    epoch_pbar = tqdm(total=args.epochs, desc="Epochs", position=0)
    epoch_pbar.update(start_epoch)

    for i in range(num_iter):
        if break_flag:
            break

        # Create step progress bar for current epoch
        step_pbar = tqdm(total=args.steps_per_epoch, desc=f"Steps (Epoch {global_epoch + 1})", position=1, leave=False)

        for data in train_loader:
            global_step += 1
            data = [d.to(device) for d in data]
            optimizer.zero_grad()
            pred = model(data)
            
            y = data[2]
            loss = criterion(pred.view(-1), y.view(-1))         
            cindex = get_cindex(
                y.detach().cpu().numpy().reshape(-1), 
                pred.detach().cpu().numpy().reshape(-1)
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            lr_scheduler.step()
            wd_scheduler.step()

            running_loss.update(loss.item(), y.size(0)) 
            running_cindex.update(cindex, y.size(0))

            # Update step progress bar
            step_pbar.update(1)
            step_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cindex': f'{cindex:.4f}'
            })

            if global_step % args.steps_per_epoch == 0:
                # Close and clear the step progress bar
                step_pbar.close()
                
                global_epoch += 1
                epoch_loss = running_loss.get_average()
                epoch_cindex = running_cindex.get_average()
                running_loss.reset()
                running_cindex.reset()

                val_loss, val_cindex = val(model, criterion, val_loader, device)
                msg = f"Epoch-{global_epoch}, Loss-{epoch_loss:.4f}, CIndex-{epoch_cindex:.4f}, Val_Loss-{val_loss:.4f}"
                logger.info(msg)

                # Update epoch progress bar
                epoch_pbar.update(1)
                epoch_pbar.set_postfix({
                    'train_loss': f'{epoch_loss:.4f}',
                    'val_loss': f'{val_loss:.4f}',
                    'val_cindex': f'{val_cindex:.4f}'
                })

                # Update TensorBoard metrics at epoch level
                writer.add_scalar("Train/Loss", epoch_loss, global_epoch)
                writer.add_scalar("Train/CIndex", epoch_cindex, global_epoch)
                writer.add_scalar("Val/Loss", val_loss, global_epoch)
                writer.add_scalar("Val/CIndex", val_cindex, global_epoch)
                writer.add_scalar("LR", optimizer.param_groups[0]['lr'], global_epoch)
                writer.add_scalar("WeightDecay", optimizer.param_groups[0]['weight_decay'], global_epoch)
                writer.flush()  # Ensure all metrics are written to disk

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    running_best_mse.update(val_loss)
                    if save_model_flag:
                        save_model_dict(model, logger.get_model_dir(), msg)
                        checkpoint_path = os.path.join(logger.get_model_dir(), f"best_checkpoint_epoch_{global_epoch}.pth")
                        save_checkpoint({
                            'epoch': global_epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': lr_scheduler.state_dict(),
                            'best_val_loss': best_val_loss,
                            'criterion_state': criterion.state_dict(),
                            'weight_decay_params': {
                                'weight_decay': args.weight_decay,
                                'current_epoch': global_epoch
                            },
                            'wd_scheduler_state': wd_scheduler.state_dict(),
                            'loss_fn': criterion.__class__.__name__,
                            'loss_fn_params': {
                                'reduction': criterion.reduction if hasattr(criterion, 'reduction') else None
                            }
                        }, filename=checkpoint_path)
                        logger.info(f"Saved Best Checkpoint at {checkpoint_path}")
                else:
                    running_best_mse.update(val_loss)
                    if running_best_mse.counter() > early_stop_epoch:
                        logger.info(f"Early stopping triggered at epoch {global_epoch}")
                        break_flag = True
                        break

                # Save latest checkpoint
                if save_model_flag:
                    latest_checkpoint_path = os.path.join(logger.get_model_dir(), "latest_checkpoint.pth")
                    save_checkpoint({
                        'epoch': global_epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': lr_scheduler.state_dict(),
                        'best_val_loss': best_val_loss,
                        'criterion_state': criterion.state_dict(),
                        'weight_decay_params': {
                            'weight_decay': args.weight_decay,
                            'current_epoch': global_epoch
                        },
                        'wd_scheduler_state': wd_scheduler.state_dict(),
                        'loss_fn': criterion.__class__.__name__,
                        'loss_fn_params': {
                            'reduction': criterion.reduction if hasattr(criterion, 'reduction') else None
                        }
                    }, filename=latest_checkpoint_path)
                    logger.info(f"Saved Latest Checkpoint at {latest_checkpoint_path}")

                if global_epoch >= args.epochs:
                    break_flag = True
                    break

                # Create new step progress bar for next epoch
                step_pbar = tqdm(total=args.steps_per_epoch, desc=f"Steps (Epoch {global_epoch + 1})", position=1, leave=False)

        # Close step progress bar if loop breaks
        if break_flag:
            step_pbar.close()
            break

    # Close progress bar
    epoch_pbar.close()
    logger.info("Training Completed.")
    writer.close()


if __name__ == "__main__":
    main()
