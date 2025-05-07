# utils.py
import os
import torch
from collections import defaultdict
import numpy as np
import math

class BestMeter(object):
    """Computes and stores the best value"""

    def __init__(self, best_type):
        self.best_type = best_type  
        self.count = 0      
        self.reset()

    def reset(self):
        if self.best_type == 'min':
            self.best = float('inf')
        else:
            self.best = -float('inf')

    def update(self, best):
        if self.best_type == 'min':
            if best < self.best:
                self.best = best
                self.count = 0
            else:
                self.count += 1
        else:
            if best > self.best:
                self.best = best
                self.count = 0
            else:
                self.count += 1

    def get_best(self):
        return self.best

    def counter(self):
        return self.count


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (self.count + 1e-12)

    def get_average(self):
        self.avg = self.sum / (self.count + 1e-12)
        return self.avg


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def save_checkpoint(state, filename='checkpoint.pth'):
    """
    Saves the training checkpoint.

    Args:
        state (dict): State dictionary containing model state, optimizer state, scheduler state, etc.
        filename (str): File path to save the checkpoint.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)
    print(f"Checkpoint has been saved to {filename}.")


def load_checkpoint(model, optimizer, scheduler, filename='checkpoint.pth'):
    """
    Loads the training checkpoint.

    Args:
        model (torch.nn.Module): The model to load state into.
        optimizer (torch.optim.Optimizer): The optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler to load state into.
        filename (str): File path from where to load the checkpoint.

    Returns:
        dict or None: Loaded checkpoint containing additional information like epoch and best validation loss.
    """
    if os.path.isfile(filename):
        print(f"=> Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load criterion state if available
        criterion_state = checkpoint.get('criterion_state')
        
        # Load weight decay parameters if available
        weight_decay_params = checkpoint.get('weight_decay_params')
        
        epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"=> Loaded checkpoint '{filename}' (epoch {epoch}) with best_val_loss {best_val_loss:.4f}")
        
        return {
            'epoch': epoch, 
            'best_val_loss': best_val_loss,
            'criterion_state': criterion_state,
            'weight_decay_params': weight_decay_params
        }
    else:
        print(f"=> No checkpoint found at '{filename}'")
        return None


def save_model_dict(model, model_dir, msg):
    """
    Saves the model's state dictionary.

    Args:
        model (torch.nn.Module): The model to save.
        model_dir (str): Directory where the model will be saved.
        msg (str): Message or identifier for the saved model.
    """
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, msg + '.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Model has been saved to {model_path}.")


def load_model_dict(model, ckpt):
    """
    Loads the model's state dictionary.

    Args:
        model (torch.nn.Module): The model to load state into.
        ckpt (str): Path to the checkpoint file.
    """
    model.load_state_dict(torch.load(ckpt))
    print(f"Model has been loaded from {ckpt}.")


