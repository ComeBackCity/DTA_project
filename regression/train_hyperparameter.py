import os
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch_geometric.loader import DataLoader

from dataset_new_gvp import (
    GNNDataset, collate,
    TupleCompose,
    MaskDrugNodeFeatures, PerturbDrugNodeFeatures, PerturbDrugEdgeAttr,
    MaskProteinNodeFeatures, PerturbProteinNodeFeatures, PerturbProteinEdgeAttr,
    DropProteinNodes, DropProteinEdges, AddRandomProteinEdges
)
from my_model_gvp import SimpleGATGVPCrossModel
from metrics import get_cindex
from utils import AverageMeter


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def val(model, criterion, dataloader, device):
    model.eval()
    running_loss = AverageMeter()
    all_targets, all_preds = [], []

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Validation", leave=False):
            data = [d.to(device) for d in data]
            y = data[3]
            pred = model(data)
            loss = criterion(pred.view(-1), y.view(-1))
            running_loss.update(loss.item(), y.size(0))
            all_targets.append(y.cpu().numpy().reshape(-1))
            all_preds.append(pred.cpu().numpy().reshape(-1))

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    cindex = get_cindex(all_targets, all_preds)
    return running_loss.get_average(), cindex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--prot_layers', type=int, default=6)
    parser.add_argument('--prot_gvp_layer', type=int, default=6)
    parser.add_argument('--drug_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--hidden_dim', type=int, default=256)
    args = parser.parse_args()

    setup_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transforms = TupleCompose([
        MaskDrugNodeFeatures(prob=0.05, mask_prob=0.05),
        PerturbDrugNodeFeatures(prob=0.05, noise_std=0.01),
        PerturbDrugEdgeAttr(prob=0.05, noise_std=0.01),
        MaskProteinNodeFeatures(prob=0.05, mask_prob=0.05),
        PerturbProteinNodeFeatures(prob=0.05, noise_std=0.01),
        PerturbProteinEdgeAttr(prob=0.05, noise_std=0.005),
        DropProteinNodes(prob=0.1, drop_prob=0.03),
        DropProteinEdges(prob=0.1, drop_prob=0.03),
        AddRandomProteinEdges(prob=0.05, add_prob=0.005)
    ])

    train_set = GNNDataset(args.dataset, split='train', transform=transforms)
    val_set = GNNDataset(args.dataset, split='valid')

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=8,
        pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=8,
        pin_memory=True, persistent_workers=True
    )

    model = SimpleGATGVPCrossModel(
        prot_feat_dim=1204,
        drug_feat_dim=34,
        prot_edge_dim=10,
        drug_edge_dim=8,
        hidden_dim=args.hidden_dim,
        prot_layers=args.prot_layers,
        prot_gvp_layer=args.prot_gvp_layer,
        drug_layers=args.drug_layers,
        out_dim=1,
        dropout=args.dropout
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float('inf')
    best_val_cindex = 0

    for epoch in range(args.epochs):
        model.train()
        for data in train_loader:
            data = [d.to(device) for d in data]
            y = data[3]
            pred = model(data)
            loss = criterion(pred.view(-1), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_loss, val_cindex = val(model, criterion, val_loader, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_cindex = val_cindex

    # Print for subprocess to parse
    print(f"VAL_LOSS: {best_val_loss}")
    print(f"VAL_CINDEX: {best_val_cindex}")
    return best_val_loss, best_val_cindex


if __name__ == '__main__':
    main()
