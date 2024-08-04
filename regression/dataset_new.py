import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
import os.path as osp
import os
import pandas as pd

class GNNDataset(Dataset):

    def __init__(self, root, dataset_name, split='train'):
        super().__init__()
        self.split = split
        self.root = root
        self.dataset_name = dataset_name
        if self.dataset_name == "davis" or self.dataset_name == 'kiba':
            df = DTI(self.dataset_name)
            df.convert_to_log("binding")
            split = davis.get_split()
            self.df = split(self.split)
        elif self.dataset_name == "full_toxcast":
            self.df = pd.read_csv(f"./data/full_toxcast/raw/data_{self.split}.csv")
            
        with open("")

    def __len__(self):
        return df.shape[0]
    
    def __getitem__(self, idx):
        data = torch.load(osp.join(f"{self.root}/processed", f'{self.split}/processed_data_{self.split}_{idx}.pt'))
        data.protein_graph.x = torch.cat([
            data.protein_graph.one_hot_residues,
            data.protein_graph.meiler_features,
            data.protein_graph.esm_embeddings,
            data.protein_graph.beta_carbon_vectors,
            data.protein_graph.seq_neighbour_vector
        ], dim=1)
        return data.mol_graph, data.protein_graph, data.y
    
def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    
    batchA.x 
    # batchB.x[]
    return batchA, batchB


if __name__ == "__main__":
    dataset = GNNDataset('data/davis')


