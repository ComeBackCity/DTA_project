# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import argparse
from tqdm import tqdm

from metrics import get_cindex, get_rm2
from dataset_new import *
from my_model import SimpleGATCrossModel
from utils import *

def val(model, criterion, dataloader, device):
    model.eval()
    running_loss = AverageMeter()

    pred_list = []
    label_list = []

    for data in tqdm(dataloader, desc="Testing", leave=False):
        data = [data_elem.to(device) for data_elem in data]

        with torch.no_grad():
            pred = model(data)
            label = data[2]
            loss = criterion(pred.view(-1), label.view(-1))
            pred_list.append(pred.view(-1).detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
            running_loss.update(loss.item(), label.size(0))

    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    epoch_cindex = get_cindex(label, pred)
    epoch_r2 = get_rm2(label, pred)
    epoch_loss = running_loss.get_average()
    running_loss.reset()

    return epoch_loss, epoch_cindex, epoch_r2

def main():
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--dataset', required=True, help='davis or kiba')
    parser.add_argument('--model_path', required=True, type=str, help='model path ready to load')
    args = parser.parse_args()

    data_root = "data"
    DATASET = args.dataset
    model_path = args.model_path

    # Load test dataset
    test_set = GNNDataset(DATASET, split='test')
    print("Number of test samples: ", len(test_set))
    test_loader = DataLoader(
        test_set, 
        batch_size=32, 
        shuffle=False, 
        num_workers=8,
        collate_fn=collate
    )

    device = torch.device('cuda:0')
    
    # Initialize model with same configuration as training
    # model = SimpleGATCrossModel(
    #     prot_feat_dim=1204, 
    #     drug_feat_dim=34, 
    #     prot_edge_dim=13,
    #     drug_edge_dim=8,
    #     hidden_dim=256,
    #     prot_layers=4,
    #     drug_layers=2, 
    #     out_dim=1,
    # ).to(device)
    
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

    criterion = nn.MSELoss()
    load_model_dict(model, model_path)
    
    # Run evaluation
    test_loss, test_cindex, test_r2 = val(model, criterion, test_loader, device)
    msg = "test_loss:%.4f, test_cindex:%.4f, test_r2:%.4f" % (test_loss, test_cindex, test_r2)
    print(msg)

if __name__ == "__main__":
    main()
