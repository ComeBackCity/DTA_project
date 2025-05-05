# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import math
import random
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader, HGTLoader
import torch.nn.functional as F
import argparse
from metrics import get_cindex, ci
from dataset_hetero import  GNNDataset, collate  # Changed import to hetero dataset
from my_model import MGraphDTA_Hetero      # Changed import to hetero model
from utils import *
from log.train_logger import TrainLogger
from torchmetrics.regression import ConcordanceCorrCoef
import torch_geometric.transforms as T
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def val(model, criterion, dataloader, device, epoch):
    model.eval()
    running_loss = AverageMeter()

    for data in tqdm(dataloader, desc=f"Validationm (Epoch {epoch})", leave=False):
        # data = {key: val.to(device) for key, val in data.items()}  # Adjusted for heterogeneous data
        data = data.to(device)
        y = data['y']  # Update to the correct key for target variable in heterogeneous data
        
        with torch.no_grad():
            pred = model(data)
            loss = criterion(pred.view(-1), y.view(-1))
            running_loss.update(loss.item(), y.size(0))

    epoch_loss = running_loss.get_average()
    running_loss.reset()

    return epoch_loss

def main():
    parser = argparse.ArgumentParser()
    setup_seed(100)

    # Add argument
    parser.add_argument('--dataset', required=True, help='davis or kiba')
    parser.add_argument('--save_model', action='store_true', help='whether save model or not')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
    parser.add_argument('--epochs', type=int, default=2000, help='epochs')
    args = parser.parse_args()

    params = dict(
        data_root="data",
        save_dir="save",
        dataset=args.dataset,
        save_model=args.save_model,
        lr=args.lr,
        batch_size=args.batch_size
    )

    logger = TrainLogger(params)
    logger.info(__file__)

    DATASET = params.get("dataset")
    save_model = params.get("save_model")
    data_root = params.get("data_root")
    fpath = os.path.join(data_root, DATASET)

    # Load heterogeneous dataset
    train_set = GNNDataset(DATASET, split='train')
    val_set = GNNDataset(DATASET, split='valid')
    

    train_loader = DataLoader(train_set, batch_size=params.get('batch_size'), shuffle=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=params.get('batch_size'), shuffle=False, num_workers=8)

    # for data in train_loader:
    #     print(data)
    #     print(data.x_dict)
    #     print(data.batch_dict)
    #     print(data.edge_attr_dict)
    #     print(data.edge_index_dict)
    #     exit()

    device = torch.device('cuda:0')
    # device = torch.device('cpu')

    model = MGraphDTA_Hetero(protein_in_dim=1314, drug_in_dim=27, super_feat_dim=27,
                           protein_edge_dim=8, drug_edge_dim=6, hidden_dim=256, 
                           out_dim=1, num_heads=1).to(device)
    
    model_weights = torch.load('./save/20241122_113152_davis/model/epoch-188, loss-0.7349, cindex-0.6167, val_loss-0.7004.pt')
    model.load_state_dict(model_weights)

    epochs = args.epochs
    break_flag = False

    optimizer = optim.Adam(model.parameters(), lr=params.get('lr'), weight_decay=0.01)
    criterion = nn.MSELoss()

    early_stop_epoch = 400
    running_loss = AverageMeter()
    running_cindex = AverageMeter()
    running_best_mse = BestMeter("min")

    model.train()

    for i in range(epochs):
        if break_flag:
            break

        for data in tqdm(train_loader, desc=f"Training (Epoch {i})", leave=False):
            # data = {key: val.to(device) for key, val in data.items()}  # Adjusted for heterogeneous data
            data = data.to(device)
            optimizer.zero_grad()
            # print("=============>")
            # print(data)
            # print("=============>")
            pred = model(data)
            y = data['y']  # Update to the correct key for target variable in heterogeneous data
            
            loss = criterion(pred.view(-1), y.view(-1))
            cindex = get_cindex(y.detach().cpu().numpy().reshape(-1),
                                pred.detach().cpu().numpy().reshape(-1))

            loss.backward()
            optimizer.step()

            running_loss.update(loss.item(), y.size(0))
            running_cindex.update(cindex, y.size(0))

        epoch_loss = running_loss.get_average()
        epoch_cindex = running_cindex.get_average()
        running_loss.reset()
        running_cindex.reset()

        val_loss = val(model, criterion, val_loader, device, i)

        msg = "epoch-%d, loss-%.4f, cindex-%.4f, val_loss-%.4f" % (i, epoch_loss, epoch_cindex, val_loss)
        logger.info(msg)

        if val_loss < running_best_mse.get_best():
            running_best_mse.update(val_loss)
            if save_model:
                save_model_dict(model, logger.get_model_dir(), msg)
        else:
            count = running_best_mse.counter()
            if count > early_stop_epoch:
                logger.info(f"early stop in epoch {i}")
                break_flag = True
                break

if __name__ == "__main__":
    main()
