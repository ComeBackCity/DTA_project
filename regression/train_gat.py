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
from torch.utils.data import default_collate\
    # , DataLoader
from torch_geometric.data import Batch
# from torch.utils.data import BatchSampler
import torch.nn.functional as F
import argparse
import gc
from tqdm import tqdm

from torch_geometric import data as DATA

from metrics import get_cindex
# from dataset import *
from dataset_new import *
# from model_gat import MGraphDTA
from model import MGraphDTA
from my_model import DTAModel
from utils import *
from log.train_logger import TrainLogger
from torch.utils.tensorboard import SummaryWriter
import resource
from transformers import get_constant_schedule_with_warmup

def val(model, criterion, dataloader, device):
    model.eval()
    running_loss = AverageMeter()

    for data in dataloader:
        data = data.to(device)

        with torch.no_grad():
            pred = model(data)
            loss = criterion(pred.view(-1), data.y.view(-1))
            label = data.y
            running_loss.update(loss.item(), label.size(0))

    epoch_loss = running_loss.get_average()
    running_loss.reset()

    model.train()

    return epoch_loss

def setup_seed(seed):
    random.seed(seed)                          
    np.random.seed(seed)                       
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True  
    # torch.use_deterministic_algorithms(True)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def collate_fn(batch):
    # print(batch)
    batch_x = [item.x for item in batch]
    batch_edge_index = [item.edge_index for item in batch]
    batch_edge_attr = [item.edge_attr for item in batch]
    batch_y = [item.y for item in batch]
    batch_target = [item.target for item in batch]
    batch_prot5_embedding = [item.prot5_embedding for item in batch]

    # Determine the maximum length of prot5_embedding in the batch
    max_len = max([len(item) for item in batch_prot5_embedding])

    # Pad prot5_embedding to the maximum length and create a mask
    batch_prot5_embedding_padded = []
    batch_mask = []
    for item in batch_prot5_embedding:
        padding_len = max_len - len(item)
        padded_item = torch.nn.functional.pad(item, (0, 0, 0, padding_len), mode='constant', value=0)
        mask = (item.sum(dim=2) != 0).float()
        batch_prot5_embedding_padded.append(padded_item)
        batch_mask.append(mask)

    # Convert the batch to PyTorch tensors
    batch_x = torch.stack(batch_x)
    batch_edge_index = torch.stack(batch_edge_index)
    batch_edge_attr = torch.stack(batch_edge_attr)
    batch_y = torch.stack(batch_y)
    batch_target = torch.stack(batch_target)
    batch_prot5_embedding_padded = torch.stack(batch_prot5_embedding_padded)
    batch_mask = torch.stack(batch_mask)

    # Create a new Data object with the padded prot5_embedding and mask
    batch_data = DATA.Data(
        x=batch_x,
        edge_index=batch_edge_index,
        edge_attr=batch_edge_attr,
        y=batch_y,
        target=batch_target,
        prot5_embedding=batch_prot5_embedding_padded,
        mask=batch_mask
    )

    return batch_data

def custom_collate(batch):
    modified_batch = []
    for item in batch:
        prot5_embedding = item.prot5_embedding
        pad_length = 1200 - prot5_embedding.shape[0] 
        prot5_embedding = F.pad(prot5_embedding, (0, 0, 0, pad_length), 'constant', 0)
        prot5_embedding = torch.unsqueeze(prot5_embedding, dim=0)
        mask = (prot5_embedding.sum(dim=2) != 0).float()
        modified_item = item.clone()
        modified_item.prot5_embedding = prot5_embedding
        modified_item.mask = mask
        modified_batch.append(modified_item)

    data_collate = Batch.from_data_list(modified_batch)
    
    return data_collate


def main():
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

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

    train_set = GNNDataset(fpath, train=True)
    test_set = GNNDataset(fpath, train=False)
    
    n_workers = 16

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=n_workers)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=n_workers)

    device = torch.device('cuda:0')
    model = MGraphDTA(3, 25 + 1, embedding_size=128, filter_num=32, out_dim=1, dropout=0.5).to(device)
    # model = DTAModel(dropout=0.5, out_dim=1).to(device)

    epochs = 3000
    steps_per_epoch = 50
    num_iter = math.ceil((epochs * steps_per_epoch) / len(train_loader))
    break_flag = False

    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    criterion = nn.MSELoss()
    writer = SummaryWriter("./tensorboard")
    # scheduler = optim.lr_scheduler.CyclicLR(
    #     optimizer=optimizer,
    #     base_lr=1e-7,
    #     max_lr=1e-3,
    #     step_size_up=50,
    #     step_size_down=50,
    #     cycle_momentum=False
    # )

    # scheduler = get_constant_schedule_with_warmup(
    #     optimizer=optimizer,
    #     num_warmup_steps=50
    # )

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
            data = data.to(device)
            pred = model(data)

            loss = criterion(pred.view(-1), data.y.view(-1))
            cindex = get_cindex(data.y.detach().cpu().numpy().reshape(-1), pred.detach().cpu().numpy().reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.update(loss.item(), data.y.size(0)) 
            running_cindex.update(cindex, data.y.size(0))

            if global_step % steps_per_epoch == 0:

                global_epoch += 1

                epoch_loss = running_loss.get_average()
                epoch_cindex = running_cindex.get_average()
                running_loss.reset()
                running_cindex.reset()

                test_loss = val(model, criterion, test_loader, device)
                # scheduler.step()

                msg = "epoch-%d, loss-%.4f, cindex-%.4f, test_loss-%.4f" % (global_epoch, epoch_loss, epoch_cindex, test_loss)
                logger.info(msg)

                writer.add_scalar(f"{DATASET}/train/loss", epoch_loss, global_epoch)
                writer.add_scalar(f"{DATASET}/train/cindex", epoch_cindex, global_epoch)
                writer.add_scalar(f"{DATASET}/test/loss", test_loss, global_epoch)

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

        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

# %%

