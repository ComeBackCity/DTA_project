import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
import os.path as osp
import os

class GNNDataset(Dataset):

    def __init__(self, root, split='train'):
        self.split = split
        self.root = root
        super().__init__()

    def __len__(self):
        processed_dir = osp.join(self.root, 'processed', self.split)
        return len(os.listdir(processed_dir))
    
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
    # batchB.x[]
    return batchA, batchB


if __name__ == "__main__":
    dataset = GNNDataset('data/davis')


