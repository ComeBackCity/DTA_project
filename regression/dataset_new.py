import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
import os.path as osp
import os
import pandas as pd
from tdc.multi_pred import DTI
import pickle


class GNNDataset(Dataset):

    def __init__(self, dataset_name, split='train', prot_transform=None, mol_transform=None):
        super().__init__()
        self.split = split
        self.dataset_name = dataset_name
        self.prot_transform = prot_transform
        self.mol_transform = mol_transform
        
        if self.dataset_name == "davis" or self.dataset_name == 'kiba':
            self.df = pd.read_csv(f"./data/{self.dataset_name}/csvs/{self.dataset_name}_{self.split}_42.csv")
        elif self.dataset_name == "full_toxcast":
            self.df = pd.read_csv(f"./data/{self.dataset_name}/raw/data_{self.split}.csv")
            
        with open(f"./data/{self.dataset_name}_molecule.pkl", "rb") as f:
            self.mol_graphs = pickle.load(f)
            
        with open(f"./data/{self.dataset_name}_mapping.pkl", "rb") as f:
            self.prot_mapping = pickle.load(f)
            
        with open(f'./data/{self.dataset_name}_stats.pkl', 'rb') as f:
            self.prot_stats = pickle.load(f)
            
        with open(f'./data/{self.dataset_name}_mostats.pkl', 'rb') as f:
            self.mol_stats = pickle.load(f)   

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        # print(self.df.columns)
        if self.dataset_name == 'full_toxcast':
            protein_key, drug_key, label_key = "sequence", "smiles", 'label'
        else:
            protein_key, drug_key, label_key = "target_sequence", "compound_iso_smiles", "affinity"
            # protein_key, drug_key, label_key = "Target", "Drug", "Y"
            
        protein = self.df.at[idx, protein_key]
        drug = self.df.at[idx, drug_key]
        label = self.df.at[idx, label_key]
        
        mol_graph = self.mol_graphs[drug]
        mol_stats = self.mol_stats
        
        l = torch.log(mol_graph.x[:, -2:])
        l = standardize_tensor(l, mol_stats['x']['mean'], mol_stats['x']['std'])
        
        drug_graph = Data(
            x = torch.cat([mol_graph.x[:, :-2], l], dim=1),
            edge_index = mol_graph.edge_index,
            edge_attr = mol_graph.edge_attr
        )
        
        header = self.prot_mapping[protein]
    
        with open(f"./data/{self.dataset_name}/protein_graphs/{header}.pkl", "rb") as f:
            prot_graph = pickle.load(f)
            
        prot_stats = self.prot_stats
        protein_graph = Data()
        
        # protein_graph.x = torch.cat([
        #     prot_graph.one_hot_residues,
        #     standardize_tensor(prot_graph.meiler_features, prot_stats['meiler_features']['mean'], prot_stats['meiler_features']['std']),
        #     standardize_tensor(prot_graph.esm_embeddings, prot_stats['esm_embeddings']['mean'], prot_stats['esm_embeddings']['std']),
        #     standardize_tensor(prot_graph.beta_carbon_vectors, prot_stats['beta_carbon_vectors']['mean'], prot_stats['beta_carbon_vectors']['std']),
        #     standardize_tensor(prot_graph.seq_neighbour_vector, prot_stats['seq_neighbour_vector']['mean'], prot_stats['seq_neighbour_vector']['std'])
        # ], dim=1)
        
        
        protein_graph.x = torch.cat([
            prot_graph.one_hot_residues,
            standardize_tensor(prot_graph.meiler_features, 
                               prot_stats['meiler_features']['mean'], 
                               prot_stats['meiler_features']['std']),
            prot_graph.esm_embeddings,
            prot_graph.beta_carbon_vectors, 
            prot_graph.seq_neighbour_vector,
            standardize_tensor(prot_graph.b_factor.reshape(-1, 1), 
                               prot_stats['b_factor']['mean'], 
                               prot_stats['b_factor']['std']),
        ], dim=1)
        
        
        
        e = standardize_tensor(prot_graph.edge_attr[:, -2:], 
                               prot_stats['edge_attr']['mean'],
                               prot_stats['edge_attr']['std'])
        
        protein_graph.edge_index = prot_graph.edge_index
        protein_graph.edge_attr = torch.cat([prot_graph.edge_attr[:, :-2], e], dim=1)
            
        # if self.prot_transform is not None:
        #     protein_graph = self.prot_transform(protein_graph)
        #     prot_graph.x = torch.cat([prot_graph.x, prot_graph.pe], dim=1)
            
        # if self.mol_transform is not None:
        #     mol_graph = self.mol_transform(mol_graph)
        #     mol_graph.x = torch.cat([mol_graph.x, mol_graph.pe], dim=1)
        
        return drug_graph, protein_graph, torch.tensor(label, dtype=torch.float32)
        
def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    
    return batchA, batchB

def standardize_tensor(tensor, mean, std, epsilon=1e-8):
    """
    Standardize a tensor using the provided mean and standard deviation arrays.
    
    Args:
    - tensor (torch.Tensor): The input tensor to be standardized.
    - mean (torch.Tensor): The mean values for standardization.
    - std (torch.Tensor): The standard deviation values for standardization.
    
    Returns:
    - standardized_tensor (torch.Tensor): The standardized tensor.
    """
    # Ensure mean and std are tensors
    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).reshape(-1, tensor.shape[1])
    std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).reshape(-1, tensor.shape[1])
    
    # Standardize the tensor
    standardized_tensor = (tensor - mean) / (std + epsilon)
    
    return standardized_tensor
