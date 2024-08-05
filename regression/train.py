# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import math
import random
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import argparse
# from metrics import get_cindex
from dataset_new import *
# from model import MGraphDTA
from model import MGraphDTA
from utils import *
from log.train_logger import TrainLogger
from torchmetrics.regression import ConcordanceCorrCoef
import torch_geometric.transforms as T
from transformers import get_cosine_schedule_with_warmup

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

    for data in dataloader:
        data = [data_elem.to(device) for data_elem in data]
        
        y = data[2]
        
        with torch.no_grad():
            pred = model(data)
            # print(pred.view(-1), y.view(-1))
            loss = criterion(pred.view(-1), y.view(-1))
            label = y
            running_loss.update(loss.item(), label.size(0))
            

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
    
    prot_transform = T.Compose([T.AddLaplacianEigenvectorPE(5 ,attr_name='pe')])  
    mol_transform = T.Compose([T.AddLaplacianEigenvectorPE(5 ,attr_name='pe')])  

    # train_set = GNNDataset(DATASET, split='train', prot_transform=prot_transform, mol_transform=mol_transform)
    # val_set = GNNDataset(DATASET, split='valid', prot_transform=prot_transform, mol_transform=mol_transform)
    
    train_set = GNNDataset(DATASET, split='train')
    val_set = GNNDataset(DATASET, split='valid')

    train_loader = DataLoader(train_set, batch_size=params.get('batch_size'), shuffle=True, num_workers=8, collate_fn = collate)
    val_loader = DataLoader(val_set, batch_size=params.get('batch_size'), shuffle=False, num_workers=8, collate_fn = collate)

    device = torch.device('cuda:0')
    
    # metrics
    get_cindex = ConcordanceCorrCoef().to(device)

    model = MGraphDTA(protein_feat_dim=1313, drug_feat_dim=27, 
                      protein_edge_dim=6, drug_edge_dim=6, filter_num=256, out_dim=1).to(device)

    epochs = 100
    steps_per_epoch = 50
    num_iter = math.ceil((epochs * steps_per_epoch) / len(train_loader))
    break_flag = False

    optimizer = optim.Adam(model.parameters(), lr=params.get('lr'))
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps_per_epoch * 5)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=num_iter,
        num_cycles=1
    )
    criterion = nn.MSELoss()

    global_step = 0
    global_epoch = 0
    early_stop_epoch = 400

    running_loss = AverageMeter()
    running_cindex = AverageMeter()
    running_best_mse = BestMeter("min")

    model.train()

    for i in range(num_iter):
        if break_flag:
            break

        for data in train_loader:

            global_step += 1       
            data = [data_elem.to(device) for data_elem in data]
            pred = model(data)
            
            y = data[2]

            loss = criterion(pred.view(-1), y.view(-1))
            cindex = get_cindex(pred.view(-1), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss.update(loss.item(), y.size(0)) 
            running_cindex.update(cindex, y.size(0))

            if global_step % steps_per_epoch == 0:

                global_epoch += 1

                epoch_loss = running_loss.get_average()
                epoch_cindex = running_cindex.get_average()
                running_loss.reset()
                running_cindex.reset()

                test_loss = val(model, criterion, val_loader, device)

                msg = "epoch-%d, loss-%.4f, cindex-%.4f, test_loss-%.4f" % (global_epoch, epoch_loss, epoch_cindex, test_loss)
                logger.info(msg)

                if test_loss < running_best_mse.get_best():
                    running_best_mse.update(test_loss)
                    if save_model:
                        save_model_dict(model, logger.get_model_dir(), msg)
                else:
                    count = running_best_mse.counter()
                    if count > early_stop_epoch:
                        logger.info(f"early stop in epoch {global_epoch}")
                        break_flag = True
                        break

if __name__ == "__main__":
    main()
