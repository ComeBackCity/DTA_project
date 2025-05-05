import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData, Batch
import pandas as pd
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
        # Define the column names based on dataset
        if self.dataset_name == 'full_toxcast':
            protein_key, drug_key, label_key = "sequence", "smiles", 'label'
        else:
            protein_key, drug_key, label_key = "target_sequence", "compound_iso_smiles", "affinity"
        
        # Extract data for protein, drug, and label
        protein = self.df.at[idx, protein_key]
        drug = self.df.at[idx, drug_key]
        label = self.df.at[idx, label_key]
        
        # Load and process drug graph
        mol_graph = self.mol_graphs[drug]
        mol_stats = self.mol_stats
        drug_x = torch.cat([mol_graph.x[:, :-2], 
                            standardize_tensor(torch.log(mol_graph.x[:, -2:]), 
                                               mol_stats['x']['mean'], 
                                               mol_stats['x']['std'])], dim=1)
        
        # Load and process protein graph
        header = self.prot_mapping[protein]
        with open(f"./data/{self.dataset_name}/protein_graphs/with_embeddings/{header}.pkl", "rb") as f:
            prot_graph = pickle.load(f)
            
        prot_stats = self.prot_stats
        protein_x = torch.cat([
            prot_graph.one_hot_residues,
            standardize_tensor(prot_graph.meiler_features, prot_stats['meiler_features']['mean'], prot_stats['meiler_features']['std']),
            prot_graph.esm_embeddings,
            prot_graph.beta_carbon_vectors,
            prot_graph.seq_neighbour_vector,
            standardize_tensor(prot_graph.b_factor.reshape(-1, 1), prot_stats['b_factor']['mean'], prot_stats['b_factor']['std']),
        ], dim=1)
        
        # Standardize protein edge attributes
        e = standardize_tensor(prot_graph.edge_attr[:, -2:], 
                               prot_stats['edge_attr']['mean'],
                               prot_stats['edge_attr']['std'])
        protein_edge_attr = torch.cat([prot_graph.edge_attr[:, :-2], e], dim=1)
        
        # Create heterogeneous graph with supernode
        hetero_graph = HeteroData()
        
        # Add nodes
        hetero_graph['drug'].x = drug_x
        hetero_graph['protein'].x = protein_x
        hetero_graph['supernode'].x = torch.ones((1, drug_x.size(1)))  # Feature size matches drug/protein

        # Add edges for drug-drug and protein-protein interactions
        hetero_graph['drug', 'drug-drug', 'drug'].edge_index = mol_graph.edge_index
        hetero_graph['drug', 'drug-drug', 'drug'].edge_attr = mol_graph.edge_attr
        hetero_graph['protein', 'protein-protein', 'protein'].edge_index = prot_graph.edge_index
        hetero_graph['protein', 'protein-protein', 'protein'].edge_attr = protein_edge_attr

        # Add supernode edges (connect supernode to all nodes in drug and protein graphs)
        num_drug_nodes = drug_x.size(0)
        num_protein_nodes = protein_x.size(0)

        # Supernode to drug nodes (supernode is the source)
        hetero_graph['supernode', 'drug-supernode', 'drug'].edge_index = torch.stack(
            [torch.zeros(num_drug_nodes, dtype=torch.long), torch.arange(num_drug_nodes)]
        )

        # Drug nodes to supernode (drug nodes are the source)
        hetero_graph['drug', 'drug-supernode', 'supernode'].edge_index = torch.stack(
            [torch.arange(num_drug_nodes), torch.zeros(num_drug_nodes, dtype=torch.long)]
        )

        # Supernode to protein nodes (supernode is the source)
        hetero_graph['supernode', 'protein-supernode', 'protein'].edge_index = torch.stack(
            [torch.zeros(num_protein_nodes, dtype=torch.long), torch.arange(num_protein_nodes)]
        )

        # Protein nodes to supernode (protein nodes are the source)
        hetero_graph['protein', 'protein-supernode', 'supernode'].edge_index = torch.stack(
            [torch.arange(num_protein_nodes), torch.zeros(num_protein_nodes, dtype=torch.long)]
        )


        # Add label as graph-level attribute
        hetero_graph.y = torch.tensor([label], dtype=torch.float32)
                
        return hetero_graph

def collate(data_list):
    return Batch.from_data_list(data_list)

def standardize_tensor(t, mean, std, epsilon=1e-8):
    mean = torch.as_tensor(mean, dtype=t.dtype, device=t.device).reshape(-1, t.shape[1])
    std = torch.as_tensor(std, dtype=t.dtype, device=t.device).reshape(-1, t.shape[1])
    return (t - mean) / (std + epsilon)
