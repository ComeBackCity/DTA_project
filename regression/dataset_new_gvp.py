import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
import os.path as osp
import os
import pandas as pd
from tdc.multi_pred import DTI
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from collections import defaultdict
import math
import numpy as np
import random
from torch_geometric.utils import subgraph
import torch.nn.functional as F
from tqdm import tqdm
import re
import gc
import torch
from torch_geometric.transforms import BaseTransform

def standardize_tensor(tensor, mean, std):
    """
    Standardizes a tensor: (x - mean) / std
    Args:
        tensor (torch.Tensor): Input tensor.
        mean (list or torch.Tensor): Mean values.
        std (list or torch.Tensor): Std values.

    Returns:
        torch.Tensor: Standardized tensor.
    """
    mean = torch.tensor(mean, device=tensor.device, dtype=tensor.dtype)
    std = torch.tensor(std, device=tensor.device, dtype=tensor.dtype)
    return (tensor - mean) / (std + 1e-6)

# class GNNDataset(Dataset):
#     def __init__(self, dataset_name, split='train'):
#         super().__init__()
#         self.dataset_name = dataset_name
#         self.split = split

#         # Load dataframes
#         if dataset_name in ["davis", "kiba"]:
#             self.df = pd.read_csv(f"./data/{dataset_name}/csvs/{dataset_name}_{split}_42.csv")
#         elif dataset_name == "full_toxcast":
#             self.df = pd.read_csv(f"./data/{dataset_name}/raw/data_{split}.csv")
#         else:
#             raise ValueError(f"Unknown dataset {dataset_name}")

#         # Load graphs with memory mapping for large files
#         # with open(f"./data/{dataset_name}_molecule.pkl", "rb") as f:
#         #     self.mol_graphs = pickle.load(f)

#         with open(f"./data/{dataset_name}_molecule_graph_and_chemfm.pkl", "rb") as f:
#             self.mol_graphs = pickle.load(f)

#         # Load mappings and stats
#         with open(f"./data/{dataset_name}_mapping.pkl", "rb") as f:
#             self.prot_mapping = pickle.load(f)

#         with open(f"./data/{dataset_name}_molecule_stats.pkl", "rb") as f:
#             self.mol_stats = pickle.load(f)

#         with open(f"./data/{dataset_name}_stats.pkl", "rb") as f:
#             self.prot_stats = pickle.load(f)
            
#         # Pre-compute and cache protein graph paths
#         self.prot_paths = {}
#         for protein_id in self.df['target_sequence'].unique():
#             prot_file = self.prot_mapping[protein_id]
#             # self.prot_paths[protein_id] = f"./data/{self.dataset_name}/protein_graphs/with_embeddings/{prot_file}.pkl"
#             self.prot_paths[protein_id] = f"./data/{self.dataset_name}/protein_graphs_with_all_embeddings/{prot_file}.pkl"

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         if self.dataset_name == 'full_toxcast':
#             protein_key, drug_key, label_key = "sequence", "smiles", "label"
#         else:
#             protein_key, drug_key, label_key = "target_sequence", "compound_iso_smiles", "affinity"

#         # Load raw entries
#         protein_id = self.df.at[idx, protein_key]
#         drug_smiles = self.df.at[idx, drug_key]
#         label = torch.tensor(self.df.at[idx, label_key], dtype=torch.float32)

#         # ---------------------------
#         # Molecule Graph Processing
#         # ---------------------------
#         mol_data = self.mol_graphs[drug_smiles]
#         # print(mol_data)

#         # print(mol_data)
        
#         # Standardize continuous molecule features
#         x_onehot_cat = mol_data.x[:, :18]  # one-hot + categorical part
#         x_continuous = mol_data.x[:, 18:32]  # continuous part (12+2 features)
#         x_donor_acceptor = mol_data.x[:, 32:]  # donor/acceptor flags

#         x_continuous_std = standardize_tensor(
#             x_continuous,
#             mean=self.mol_stats['x_continuous']['mean'],
#             std=self.mol_stats['x_continuous']['std']
#         )


#         drug_graph = Data(
#             x = torch.cat([x_onehot_cat, x_continuous_std, x_donor_acceptor], dim=1),
#             edge_index = mol_data.edge_index,
#             edge_attr = standardize_tensor(
#                 mol_data.edge_attr,
#                 mean=self.mol_stats['edge_attr']['mean'],
#                 std=self.mol_stats['edge_attr']['std']
#             ),
#             # cls_embedding = mol_data.cls_embedding,
#             # token_embeddings = mol_data.token_embeddings
#         )

#         # ---------------------------
#         # Protein Graph Processing
#         # ---------------------------
#         # Load protein data using cached path
#         # print(self.prot_paths[protein_id])
#         with open(self.prot_paths[protein_id], "rb") as f:
#             prot_data = pickle.load(f)

#         # print(prot_data)
#         # exit()

#         # Assemble protein features more efficiently
#         prot_node_feats = torch.cat([
#             prot_data.one_hot_residues,
#             standardize_tensor(prot_data.meiler_features,
#                                mean=self.prot_stats['meiler_features']['mean'],
#                                std=self.prot_stats['meiler_features']['std']),
#             prot_data.residue_embeddings,
#             prot_data.beta_carbon_vector,
#             prot_data.seq_neighbour_vector,
#             standardize_tensor(prot_data.b_factor.reshape(-1, 1),
#                                mean=self.prot_stats['b_factor']['mean'],
#                                std=self.prot_stats['b_factor']['std']),
#             standardize_tensor(prot_data.physicochemical_feat,
#                                mean=self.prot_stats['physicochemical_feat']['mean'],
#                                std=self.prot_stats['physicochemical_feat']['std']),
#             standardize_tensor(prot_data.degree,
#                                mean=self.prot_stats['degree']['mean'],
#                                std=self.prot_stats['degree']['std']),
#             standardize_tensor(prot_data.betweenness,
#                                mean=self.prot_stats['betweenness']['mean'],
#                                std=self.prot_stats['betweenness']['std']),
#             standardize_tensor(prot_data.pagerank,
#                                mean=self.prot_stats['pagerank']['mean'],
#                                std=self.prot_stats['pagerank']['std']),
#             standardize_tensor(prot_data.contact_number,
#                                mean=self.prot_stats['contact_number']['mean'],
#                                std=self.prot_stats['contact_number']['std'])
#         ], dim=1)

#         protein_graph = Data(
#             x = prot_node_feats,
#             edge_index = prot_data.edge_index,
#             edge_attr = torch.cat([
#                 prot_data.edge_attr[:, :6],  # bond-type kinds
#                 standardize_tensor(
#                     prot_data.edge_attr[:, 6:],  # distance, angle, dx, dy, dz, seq_sep
#                     mean=self.prot_stats['edge_attr']['mean'],
#                     std=self.prot_stats['edge_attr']['std']
#                 )
#             ], dim=1),
#             # cls_embedding = prot_data.cls_embedding,
#             # residue_embeddings = prot_data.residue_embeddings
#         )

#         # ---------------------------
#         # AUGMENTATION (train split only)
#         # ---------------------------
#         if self.split == 'train':
#             # List of all possible non-subgraphing augmentations
#             non_subgraph_augs = [
#                 # Drug graph augmentations
#                 (lambda: mask_node_features(drug_graph.x, mask_prob=0.03), 0.2), 
#                 (lambda: perturb_features(drug_graph.x, noise_std=0.005), 0.2),
#                 (lambda: perturb_edge_attr(drug_graph.edge_attr, noise_std=0.005), 0.2),
#                 # Protein graph augmentations
#                 (lambda: mask_node_features(protein_graph.x, mask_prob=0.03), 0.2),
#                 (lambda: perturb_features(protein_graph.x, noise_std=0.005), 0.2),
#                 (lambda: perturb_edge_attr(protein_graph.edge_attr, noise_std=0.005), 0.2),
#                 # Node and edge operations
#                 (lambda: drop_nodes(protein_graph.x, protein_graph.edge_index, protein_graph.edge_attr, drop_prob=0.03), 0.25),
#                 (lambda: drop_edges(protein_graph.edge_index, protein_graph.edge_attr, drop_prob=0.03), 0.25),
#                 (lambda: add_random_edges(protein_graph.edge_index, protein_graph.edge_attr, protein_graph.x.size(0), add_prob=0.005), 0.2)
#             ]
            
#             # Randomly select up to 3 non-subgraphing augmentations
#             selected_augs = []
#             for aug_func, prob in non_subgraph_augs:
#                 if random.random() < prob:
#                     selected_augs.append(aug_func)
#                 if len(selected_augs) >= 3:
#                     break
            
#             # Apply selected augmentations
#             for aug_func in selected_augs:
#                 result = aug_func()
#                 if isinstance(result, tuple):
#                     if len(result) == 2:  # edge operations
#                         protein_graph.edge_index, protein_graph.edge_attr = result
#                     else:  # node operations
#                         protein_graph.x, protein_graph.edge_index, protein_graph.edge_attr = result
#                 else:
#                     if result.shape == drug_graph.x.shape:
#                         drug_graph.x = result
#                     elif result.shape == protein_graph.x.shape:
#                         protein_graph.x = result
#                     elif result.shape == drug_graph.edge_attr.shape:
#                         drug_graph.edge_attr = result
#                     elif result.shape == protein_graph.edge_attr.shape:
#                         protein_graph.edge_attr = result
            
#             # Subgraph sampling (30% chance) - independent of other augmentations
#             if random.random() < 0.3:
#                 # Randomly sample between 60% and 85% of the graph
#                 sample_ratio = random.uniform(0.6, 0.85)
#                 protein_graph.x, protein_graph.edge_index, protein_graph.edge_attr = sample_subgraph(
#                     protein_graph.x, protein_graph.edge_index, protein_graph.edge_attr, 
#                     sample_ratio=sample_ratio)

#         return drug_graph, protein_graph, label

#     def get_labels(self):
#         if self.dataset_name == 'full_toxcast':
#             return list(self.df["label"])
#         else:
#             return list(self.df["affinity"])

        
# def collate(data_list):
#     # Pre-allocate lists for better memory efficiency
#     drug_graphs = []
#     protein_graphs = []
#     labels = []
    
#     # Collect data
#     for data in data_list:
#         drug_graphs.append(data[0])
#         protein_graphs.append(data[1])
#         labels.append(data[2])
    
#     # Create batches
#     batchA = Batch.from_data_list(drug_graphs)
#     batchB = Batch.from_data_list(protein_graphs)
#     labels = torch.stack(labels)  # More efficient than torch.Tensor
    
#     return batchA, batchB, labels

def standardize_tensor(tensor, mean, std):
    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device)
    return (tensor - mean) / (std + 1e-8)

class GNNDataset(Dataset):
    def __init__(self, dataset_name, split='train', transform = None):
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split

        if dataset_name in ["davis", "kiba"]:
            self.df = pd.read_csv(f"./data/{dataset_name}/csvs/{dataset_name}_{split}_42.csv")
        elif dataset_name == "full_toxcast":
            self.df = pd.read_csv(f"./data/{dataset_name}/raw/data_{split}.csv")
        else:
            raise ValueError(f"Unknown dataset {dataset_name}")

        self.mol_graphs = torch.load(f"./data/{dataset_name}_molecule_graph_and_chemfm.pt", weights_only=False)

        import pickle
        with open(f"./data/{dataset_name}_mapping.pkl", "rb") as f:
            self.prot_mapping = pickle.load(f)

        self.mol_stats = torch.load(f"./data/{dataset_name}_molecule_stats.pt")
        with open(f"./data/{dataset_name}_stats_gvp.pkl", "rb") as f:
            self.prot_stats = pickle.load(f)

        self.prot_paths = {
            protein_id: f"./data/{dataset_name}/protein_graphs_with_all_embeddings_gvp/{self.prot_mapping[protein_id]}.pt"
            for protein_id in self.df['target_sequence'].unique()
        }

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        protein_key, drug_key, label_key = ("sequence", "smiles", "label") \
            if self.dataset_name == 'full_toxcast' \
            else ("target_sequence", "compound_iso_smiles", "affinity")

        protein_id = self.df.at[idx, protein_key]
        drug_smiles = self.df.at[idx, drug_key]
        label = torch.tensor(self.df.at[idx, label_key], dtype=torch.float32)

        mol_data = self.mol_graphs[drug_smiles]

        x_continuous_std = standardize_tensor(
            mol_data.x[:, 18:32],
            mean=self.mol_stats['x_continuous']['mean'],
            std=self.mol_stats['x_continuous']['std']
        )

        drug_graph = Data(
            x=torch.cat([mol_data.x[:, :18], x_continuous_std, mol_data.x[:, 32:]], dim=1),
            edge_index=mol_data.edge_index,
            edge_attr=standardize_tensor(
                mol_data.edge_attr,
                mean=self.mol_stats['edge_attr']['mean'],
                std=self.mol_stats['edge_attr']['std']
            )
        )

        prot_data = torch.load(self.prot_paths[protein_id], weights_only=False)

        prot_node_feats = torch.cat([
            prot_data.one_hot_residues,
            standardize_tensor(prot_data.meiler_features, self.prot_stats['meiler_features']['mean'], self.prot_stats['meiler_features']['std']),
            prot_data.residue_embeddings,
            prot_data.beta_carbon_vector,
            prot_data.seq_neighbour_vector,
            standardize_tensor(prot_data.b_factor.reshape(-1, 1), self.prot_stats['b_factor']['mean'], self.prot_stats['b_factor']['std']),
            standardize_tensor(prot_data.physicochemical_feat, self.prot_stats['physicochemical_feat']['mean'], self.prot_stats['physicochemical_feat']['std']),
            standardize_tensor(prot_data.degree, self.prot_stats['degree']['mean'], self.prot_stats['degree']['std']),
            standardize_tensor(prot_data.betweenness, self.prot_stats['betweenness']['mean'], self.prot_stats['betweenness']['std']),
            standardize_tensor(prot_data.pagerank, self.prot_stats['pagerank']['mean'], self.prot_stats['pagerank']['std']),
            standardize_tensor(prot_data.contact_number, self.prot_stats['contact_number']['mean'], self.prot_stats['contact_number']['std'])
        ], dim=1)


        protein_graph = Data(
            x=prot_node_feats,
            edge_index=prot_data.edge_index,
            edge_attr=torch.cat([
                prot_data.edge_attr[:, :6],
                standardize_tensor(prot_data.edge_attr[:, 6:9], self.prot_stats['edge_attr']['mean'], self.prot_stats['edge_attr']['std']),
                prot_data.edge_attr[:, 9:]
            ], dim=1)
        )
        
        protein_graph_gvp = Data(
            x=prot_data.x,
            node_s=prot_data.node_s,
            node_v=prot_data.node_v,
            edge_s=prot_data.edge_s,
            edge_v=prot_data.edge_v,
            edge_index=prot_data.edge_index
        )


        if self.transform is not None:
            drug_graph, protein_graph, protein_graph_gvp, label = self.transform((drug_graph, protein_graph, protein_graph_gvp, label))

        return drug_graph, protein_graph, protein_graph_gvp, label

    def get_labels(self):
        if self.dataset_name == 'full_toxcast':
            return list(self.df["label"])
        else:
            return list(self.df["affinity"])

def collate(data_list):
    drug_graphs, protein_graphs, protein_graphs_gvp, labels = zip(*data_list)
    return Batch.from_data_list(drug_graphs), Batch.from_data_list(protein_graphs), Batch.from_data_list(protein_graphs_gvp), torch.stack(labels)


class MaskDrugNodeFeatures(BaseTransform):
    def __init__(self, prob=0.2, mask_prob=0.03):
        self.prob = prob
        self.mask_prob = mask_prob

    def __call__(self, data_tuple):
        if torch.rand(1).item() < self.prob:
            drug_graph, protein_graph, protein_graph_gvp, label = data_tuple
            drug_graph = drug_graph.clone()
            mask = torch.rand(drug_graph.x.size(0), device=drug_graph.x.device) < self.mask_prob
            drug_graph.x[mask] = 0.0
            return drug_graph, protein_graph, protein_graph_gvp, label
        return data_tuple

class PerturbDrugNodeFeatures(BaseTransform):
    def __init__(self, prob=0.2, noise_std=0.005):
        self.prob = prob
        self.noise_std = noise_std

    def __call__(self, data_tuple):
        if torch.rand(1).item() < self.prob:
            drug_graph, protein_graph, protein_graph_gvp, label = data_tuple
            drug_graph = drug_graph.clone()
            drug_graph.x += torch.randn_like(drug_graph.x) * self.noise_std
            return drug_graph, protein_graph, protein_graph_gvp, label
        return data_tuple

class PerturbDrugEdgeAttr(BaseTransform):
    def __init__(self, prob=0.2, noise_std=0.005):
        self.prob = prob
        self.noise_std = noise_std

    def __call__(self, data_tuple):
        if torch.rand(1).item() < self.prob:
            drug_graph, protein_graph, protein_graph_gvp, label = data_tuple
            drug_graph = drug_graph.clone()
            drug_graph.edge_attr += torch.randn_like(drug_graph.edge_attr) * self.noise_std
            return drug_graph, protein_graph, protein_graph_gvp, label
        return data_tuple


class MaskProteinNodeFeatures(BaseTransform):
    def __init__(self, prob=0.2, mask_prob=0.03):
        self.prob = prob
        self.mask_prob = mask_prob

    def __call__(self, data_tuple):
        if torch.rand(1).item() < self.prob:
            drug_graph, protein_graph, protein_graph_gvp, label = data_tuple
            protein_graph = protein_graph.clone()
            mask = torch.rand(protein_graph.x.size(0), device=protein_graph.x.device) < self.mask_prob
            protein_graph.x[mask] = 0.0
            return drug_graph, protein_graph, protein_graph_gvp, label
        return data_tuple
    
class PerturbProteinNodeFeatures(BaseTransform):
    def __init__(self, prob=0.2, noise_std=0.005, shear_range=0.05):
        self.prob = prob
        self.noise_std = noise_std
        self.shear_range = shear_range

    def random_shear_matrix(self, dim=3):
        mat = torch.eye(dim)
        for i in range(dim):
            for j in range(dim):
                if i != j:
                    mat[i, j] = (torch.rand(1).item() - 0.5) * 2 * self.shear_range
        return mat

    def __call__(self, data_tuple):
        if torch.rand(1).item() < self.prob:
            drug_graph, protein_graph, protein_graph_gvp, label = data_tuple
            protein_graph = protein_graph.clone()
            if protein_graph_gvp is not None:
                protein_graph_gvp = protein_graph_gvp.clone()

            protein_graph.x += torch.randn_like(protein_graph.x) * self.noise_std

            # Apply random shear to GVP coordinates if present
            shear_mat = None
            coords_gvp = None
            if protein_graph_gvp is not None and hasattr(protein_graph_gvp, 'x') and protein_graph_gvp.x is not None:
                coords_gvp = protein_graph_gvp.x
                shear_mat = self.random_shear_matrix(dim=coords_gvp.size(1)).to(coords_gvp.device)
                new_coords_gvp = torch.matmul(coords_gvp, shear_mat)
                protein_graph_gvp.x = new_coords_gvp

                # Update all edge features that depend on coordinates in GVP
                if hasattr(protein_graph_gvp, 'edge_index') and protein_graph_gvp.edge_index is not None:
                    row, col = protein_graph_gvp.edge_index
                    # Update edge_v (direction vectors)
                    if hasattr(protein_graph_gvp, 'edge_v') and protein_graph_gvp.edge_v is not None:
                        diff = new_coords_gvp[row] - new_coords_gvp[col]
                        dist = diff.norm(dim=-1, keepdim=True) + 1e-8
                        direction = diff / dist
                        protein_graph_gvp.edge_v = direction.unsqueeze(1)
                    # Update edge_s if it contains distance as first column
                    if hasattr(protein_graph_gvp, 'edge_s') and protein_graph_gvp.edge_s is not None:
                        diff = new_coords_gvp[row] - new_coords_gvp[col]
                        dist = diff.norm(dim=-1)
                        if protein_graph_gvp.edge_s.size(1) > 0:
                            protein_graph_gvp.edge_s[:, 0] = dist

                # Update GAT edge_attr distance (7th column) using new_coords_gvp
                if hasattr(protein_graph, 'edge_attr') and protein_graph.edge_attr is not None and \
                   hasattr(protein_graph, 'edge_index') and protein_graph.edge_index is not None:
                    row, col = protein_graph.edge_index
                    diff = new_coords_gvp[row] - new_coords_gvp[col]
                    dist = diff.norm(dim=-1)
                    if protein_graph.edge_attr.size(1) > 6:
                        protein_graph.edge_attr[:, 6] = dist

            return drug_graph, protein_graph, protein_graph_gvp, label
        return data_tuple

class PerturbProteinEdgeAttr(BaseTransform):
    def __init__(self, prob=0.2, noise_std=0.005):
        self.prob = prob
        self.noise_std = noise_std

    def __call__(self, data_tuple):
        if torch.rand(1).item() < self.prob:
            drug_graph, protein_graph, protein_graph_gvp, label = data_tuple
            protein_graph = protein_graph.clone()
            protein_graph.edge_attr += torch.randn_like(protein_graph.edge_attr) * self.noise_std
            return drug_graph, protein_graph, protein_graph_gvp, label
        return data_tuple

class DropProteinNodes(BaseTransform):
    def __init__(self, prob=0.25, drop_prob=0.03):
        self.prob = prob
        self.drop_prob = drop_prob

    def __call__(self, data_tuple):
        if torch.rand(1).item() < self.prob:
            drug_graph, protein_graph, protein_graph_gvp, label = data_tuple
            protein_graph = protein_graph.clone()
            if protein_graph_gvp is not None:
                protein_graph_gvp = protein_graph_gvp.clone()

            keep = torch.rand(protein_graph.x.size(0), device=protein_graph.x.device) > self.drop_prob
            protein_graph.x = protein_graph.x[keep]

            mapping = torch.full((len(keep),), -1, device=protein_graph.x.device)
            mapping[keep] = torch.arange(keep.sum(), device=protein_graph.x.device)

            mask_edge = keep[protein_graph.edge_index[0]] & keep[protein_graph.edge_index[1]]
            protein_graph.edge_index = mapping[protein_graph.edge_index[:, mask_edge]]
            protein_graph.edge_attr = protein_graph.edge_attr[mask_edge]

            if protein_graph_gvp is not None:
                # Update all node features in GVP
                for attr in ['x', 'node_s', 'node_v']:
                    if hasattr(protein_graph_gvp, attr) and getattr(protein_graph_gvp, attr) is not None:
                        setattr(protein_graph_gvp, attr, getattr(protein_graph_gvp, attr)[keep])
                mask_edge_gvp = keep[protein_graph_gvp.edge_index[0]] & keep[protein_graph_gvp.edge_index[1]]
                protein_graph_gvp.edge_index = mapping[protein_graph_gvp.edge_index[:, mask_edge_gvp]]
                for attr in ['edge_s', 'edge_v']:
                    if hasattr(protein_graph_gvp, attr) and getattr(protein_graph_gvp, attr) is not None:
                        setattr(protein_graph_gvp, attr, getattr(protein_graph_gvp, attr)[mask_edge_gvp])
                # Update edge_v and edge_s using new coordinates
                if hasattr(protein_graph_gvp, 'x') and protein_graph_gvp.x is not None:
                    row, col = protein_graph_gvp.edge_index
                    if hasattr(protein_graph_gvp, 'edge_v') and protein_graph_gvp.edge_v is not None:
                        diff = protein_graph_gvp.x[row] - protein_graph_gvp.x[col]
                        dist = diff.norm(dim=-1, keepdim=True) + 1e-8
                        direction = diff / dist
                        protein_graph_gvp.edge_v = direction.unsqueeze(1)
                    if hasattr(protein_graph_gvp, 'edge_s') and protein_graph_gvp.edge_s is not None:
                        diff = protein_graph_gvp.x[row] - protein_graph_gvp.x[col]
                        dist = diff.norm(dim=-1)
                        if protein_graph_gvp.edge_s.size(1) > 0:
                            protein_graph_gvp.edge_s[:, 0] = dist
                    # Update GAT edge_attr distance (7th column) using new GVP coordinates
                    if hasattr(protein_graph, 'edge_attr') and protein_graph.edge_attr is not None:
                        diff = protein_graph_gvp.x[row] - protein_graph_gvp.x[col]
                        dist = diff.norm(dim=-1)
                        if protein_graph.edge_attr.size(1) > 6:
                            protein_graph.edge_attr[:, 6] = dist
            return drug_graph, protein_graph, protein_graph_gvp, label
        return data_tuple

class DropProteinEdges(BaseTransform):
    def __init__(self, prob=0.25, drop_prob=0.03):
        self.prob = prob
        self.drop_prob = drop_prob

    def __call__(self, data_tuple):
        if torch.rand(1).item() < self.prob:
            drug_graph, protein_graph, protein_graph_gvp, label = data_tuple
            protein_graph = protein_graph.clone()
            if protein_graph_gvp is not None:
                protein_graph_gvp = protein_graph_gvp.clone()
            num_edges = protein_graph.edge_index.size(1)
            keep = torch.rand(num_edges, device=protein_graph.edge_index.device) > self.drop_prob
            protein_graph.edge_index = protein_graph.edge_index[:, keep]
            protein_graph.edge_attr = protein_graph.edge_attr[keep]
            if protein_graph_gvp is not None:
                protein_graph_gvp.edge_index = protein_graph_gvp.edge_index[:, keep]
                for attr in ['edge_s', 'edge_v']:
                    if hasattr(protein_graph_gvp, attr) and getattr(protein_graph_gvp, attr) is not None:
                        setattr(protein_graph_gvp, attr, getattr(protein_graph_gvp, attr)[keep])
                # Update edge_v and edge_s using new coordinates
                if hasattr(protein_graph_gvp, 'x') and protein_graph_gvp.x is not None:
                    row, col = protein_graph_gvp.edge_index
                    if hasattr(protein_graph_gvp, 'edge_v') and protein_graph_gvp.edge_v is not None:
                        diff = protein_graph_gvp.x[row] - protein_graph_gvp.x[col]
                        dist = diff.norm(dim=-1, keepdim=True) + 1e-8
                        direction = diff / dist
                        protein_graph_gvp.edge_v = direction.unsqueeze(1)
                    if hasattr(protein_graph_gvp, 'edge_s') and protein_graph_gvp.edge_s is not None:
                        diff = protein_graph_gvp.x[row] - protein_graph_gvp.x[col]
                        dist = diff.norm(dim=-1)
                        if protein_graph_gvp.edge_s.size(1) > 0:
                            protein_graph_gvp.edge_s[:, 0] = dist
                    # Update GAT edge_attr distance (7th column) using new GVP coordinates
                    if hasattr(protein_graph, 'edge_attr') and protein_graph.edge_attr is not None:
                        diff = protein_graph_gvp.x[row] - protein_graph_gvp.x[col]
                        dist = diff.norm(dim=-1)
                        if protein_graph.edge_attr.size(1) > 6:
                            protein_graph.edge_attr[:, 6] = dist
            return drug_graph, protein_graph, protein_graph_gvp, label
        return data_tuple

class AddRandomProteinEdges(BaseTransform):
    def __init__(self, prob=0.2, add_prob=0.005):
        self.prob = prob
        self.add_prob = add_prob

    def __call__(self, data_tuple):
        if torch.rand(1).item() < self.prob:
            drug_graph, protein_graph, protein_graph_gvp, label = data_tuple
            protein_graph = protein_graph.clone()
            if protein_graph_gvp is not None:
                protein_graph_gvp = protein_graph_gvp.clone()
            num_nodes = protein_graph.x.size(0)
            num_possible = num_nodes * (num_nodes - 1)
            num_add = int(self.add_prob * num_possible)
            rand_src = torch.randint(0, num_nodes, (num_add,), device=protein_graph.x.device)
            rand_dst = torch.randint(0, num_nodes, (num_add,), device=protein_graph.x.device)
            new_edges = torch.stack([rand_src, rand_dst], dim=0)
            new_edge_attr = torch.zeros((num_add, protein_graph.edge_attr.size(1)), device=protein_graph.edge_attr.device)
            protein_graph.edge_index = torch.cat([protein_graph.edge_index, new_edges], dim=1)
            protein_graph.edge_attr = torch.cat([protein_graph.edge_attr, new_edge_attr], dim=0)
            if protein_graph_gvp is not None:
                protein_graph_gvp.edge_index = torch.cat([protein_graph_gvp.edge_index, new_edges], dim=1)
                for attr in ['edge_s', 'edge_v']:
                    if hasattr(protein_graph_gvp, attr) and getattr(protein_graph_gvp, attr) is not None:
                        pad_shape = list(getattr(protein_graph_gvp, attr).shape)
                        pad_shape[0] = num_add
                        pad = torch.zeros(pad_shape, device=getattr(protein_graph_gvp, attr).device, dtype=getattr(protein_graph_gvp, attr).dtype)
                        setattr(protein_graph_gvp, attr, torch.cat([getattr(protein_graph_gvp, attr), pad], dim=0))
                # Update edge_v and edge_s using new coordinates
                if hasattr(protein_graph_gvp, 'x') and protein_graph_gvp.x is not None:
                    row, col = protein_graph_gvp.edge_index
                    if hasattr(protein_graph_gvp, 'edge_v') and protein_graph_gvp.edge_v is not None:
                        diff = protein_graph_gvp.x[row] - protein_graph_gvp.x[col]
                        dist = diff.norm(dim=-1, keepdim=True) + 1e-8
                        direction = diff / dist
                        protein_graph_gvp.edge_v = direction.unsqueeze(1)
                    if hasattr(protein_graph_gvp, 'edge_s') and protein_graph_gvp.edge_s is not None:
                        diff = protein_graph_gvp.x[row] - protein_graph_gvp.x[col]
                        dist = diff.norm(dim=-1)
                        if protein_graph_gvp.edge_s.size(1) > 0:
                            protein_graph_gvp.edge_s[:, 0] = dist
                    # Update GAT edge_attr distance (7th column) using new GVP coordinates
                    if hasattr(protein_graph, 'edge_attr') and protein_graph.edge_attr is not None:
                        diff = protein_graph_gvp.x[row] - protein_graph_gvp.x[col]
                        dist = diff.norm(dim=-1)
                        if protein_graph.edge_attr.size(1) > 6:
                            protein_graph.edge_attr[:, 6] = dist
            return drug_graph, protein_graph, protein_graph_gvp, label
        return data_tuple

from torch_geometric.transforms import BaseTransform

class TupleCompose(BaseTransform):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data_tuple):
        for transform in self.transforms:
            data_tuple = transform(data_tuple)
        return data_tuple


def standardize_tensor_(tensor, mean, std, epsilon=1e-8):
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
    print(tensor.shape)
    print(mean.shape)
    print(std.shape)
    exit()
    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).reshape(-1, tensor.shape[1])
    std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).reshape(-1, tensor.shape[1])
    
    # Standardize the tensor
    standardized_tensor = (tensor - mean) / (std + epsilon)
    
    return standardized_tensor


class BalancedRegressionBatchSampler(Sampler):
    def __init__(self, labels, batch_size, minority_ratio=0.1, shuffle=True):
        """
        Custom batch sampler to ensure diversity in label values.

        Args:
            labels (List[float] or Tensor): Continuous labels.
            batch_size (int): Number of samples per batch.
            minority_ratio (float): Proportion of minority samples in each batch.
            shuffle (bool): Whether to shuffle samples within each group.
        """
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Define minority and majority labels
        label_counts = defaultdict(int)
        for label in self.labels:
            label_counts[label] += 1

        # Identify majority labels (most frequent)
        max_count = max(label_counts.values())
        self.majority_labels = [label for label, count in label_counts.items() if count == max_count]
        # Identify minority labels (less frequent)
        self.minority_labels = [label for label, count in label_counts.items() if count < max_count]

        # Organize indices by label
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.label_to_indices[label].append(idx)

        # Extract majority and minority indices
        self.majority_indices = []
        self.minority_indices = []
        for label, indices in self.label_to_indices.items():
            if label in self.minority_labels:
                self.minority_indices.extend(indices)
            else:
                self.majority_indices.extend(indices)

        # Shuffle if required
        if self.shuffle:
            random.shuffle(self.majority_indices)
            random.shuffle(self.minority_indices)

        # Calculate number of minority samples per batch
        self.minority_ratio = minority_ratio
        self.minority_per_batch = max(1, int(self.batch_size * self.minority_ratio))
        self.majority_per_batch = self.batch_size - self.minority_per_batch

    def __iter__(self):
        # Create copies to avoid modifying original lists
        majority = self.majority_indices.copy()
        minority = self.minority_indices.copy()

        if self.shuffle:
            random.shuffle(majority)
            random.shuffle(minority)

        # Calculate total number of batches
        num_minority_batches = math.ceil(len(minority) / self.minority_per_batch)
        num_majority_batches = math.ceil(len(majority) / self.majority_per_batch)
        num_batches = max(num_minority_batches, num_majority_batches)

        for _ in range(num_batches):
            batch = []
            # Add minority samples
            for _ in range(self.minority_per_batch):
                if len(minority) > 0:
                    batch.append(minority.pop())
                else:
                    # If minority samples are exhausted, reshuffle or skip
                    if self.shuffle:
                        random.shuffle(self.minority_indices)
                        minority.extend(self.minority_indices.copy())
                        if len(minority) > 0:
                            batch.append(minority.pop())
                    # Else, skip adding
            # Add majority samples
            for _ in range(self.majority_per_batch):
                if len(majority) > 0:
                    batch.append(majority.pop())
                else:
                    # If majority samples are exhausted, reshuffle or skip
                    if self.shuffle:
                        random.shuffle(self.majority_indices)
                        majority.extend(self.majority_indices.copy())
                        if len(majority) > 0:
                            batch.append(majority.pop())
                    # Else, skip adding
                    
            if len(batch) > 0:
                yield batch

    def __len__(self):
        return math.ceil(max(len(self.minority_indices)/self.minority_per_batch, len(self.majority_indices)/self.majority_per_batch))
    

class BalancedRegressionBatchSampler2(Sampler):
    def __init__(self, labels, batch_size, minority_ratio=0.1, shuffle=True):
        """
        Custom batch sampler to ensure diversity in label values.

        Args:
            labels (List[float] or Tensor): Continuous labels.
            batch_size (int): Number of samples per batch.
            minority_ratio (float): Proportion of minority samples in each batch.
            shuffle (bool): Whether to shuffle samples within each group.
        """
        self.labels = np.array(labels) if not isinstance(labels, np.ndarray) else labels
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Use numpy's unique to get counts more efficiently
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        max_count = np.max(counts)
        
        # Create masks for majority and minority labels
        majority_mask = counts == max_count
        self.majority_labels = unique_labels[majority_mask]
        self.minority_labels = unique_labels[~majority_mask]

        # Create label to indices mapping using numpy operations
        self.label_to_indices = {}
        for label in unique_labels:
            self.label_to_indices[label] = np.where(self.labels == label)[0]

        # Extract majority and minority indices
        self.majority_indices = np.concatenate([self.label_to_indices[label] for label in self.majority_labels])
        self.minority_indices = np.concatenate([self.label_to_indices[label] for label in self.minority_labels])

        # Pre-compute batch sizes
        self.minority_ratio = minority_ratio
        self.minority_per_batch = max(1, int(self.batch_size * self.minority_ratio))
        self.majority_per_batch = self.batch_size - self.minority_per_batch

        # Pre-compute number of batches
        self.num_minority_batches = math.ceil(len(self.minority_indices) / self.minority_per_batch)
        self.num_majority_batches = math.ceil(len(self.majority_indices) / self.majority_per_batch)
        self.num_batches = max(self.num_minority_batches, self.num_majority_batches)

        # Initialize pointers
        self.majority_ptr = 0
        self.minority_ptr = 0

        # Pre-shuffle if shuffle is enabled
        if self.shuffle:
            np.random.shuffle(self.majority_indices)
            np.random.shuffle(self.minority_indices)

    def __iter__(self):
        # Reset pointers
        self.majority_ptr = 0
        self.minority_ptr = 0

        # Pre-shuffle if shuffle is enabled
        if self.shuffle:
            np.random.shuffle(self.majority_indices)
            np.random.shuffle(self.minority_indices)

        for _ in range(self.num_batches):
            batch = []
            
            # Handle minority samples
            if self.minority_ptr + self.minority_per_batch > len(self.minority_indices):
                if self.shuffle:
                    np.random.shuffle(self.minority_indices)
                self.minority_ptr = 0
            
            end = min(self.minority_ptr + self.minority_per_batch, len(self.minority_indices))
            batch.extend(self.minority_indices[self.minority_ptr:end].tolist())
            self.minority_ptr += self.minority_per_batch

            # Handle majority samples
            if self.majority_ptr + self.majority_per_batch > len(self.majority_indices):
                if self.shuffle:
                    np.random.shuffle(self.majority_indices)
                self.majority_ptr = 0
            
            end = min(self.majority_ptr + self.majority_per_batch, len(self.majority_indices))
            batch.extend(self.majority_indices[self.majority_ptr:end].tolist())
            self.majority_ptr += self.majority_per_batch

            # Pad batch if necessary
            while len(batch) < self.batch_size:
                if len(self.minority_indices) > 0:
                    batch.append(self.minority_indices[0])
                elif len(self.majority_indices) > 0:
                    batch.append(self.majority_indices[0])
                else:
                    break

            yield batch

    def __len__(self):
        return self.num_batches

class AdaptiveBalancedSampler(torch.utils.data.Sampler):
    def __init__(self, labels, batch_size, n_clusters=5, shuffle=True, adaptive_ratio=True):
        super().__init__(None)
        self.labels = torch.tensor(labels, dtype=torch.float32)  # Ensure float32 for efficiency
        self.batch_size = batch_size
        self.n_clusters = n_clusters
        self.shuffle = shuffle
        self.adaptive_ratio = adaptive_ratio
        self.n_samples = len(labels)
        
        # Pre-compute clusters once
        self.clusters = self._compute_clusters()
        
        # Initialize cluster weights
        self.cluster_weights = torch.ones(n_clusters) / n_clusters
        
        # Performance tracking with more efficient storage
        self.cluster_performance = torch.zeros(n_clusters)
        self.cluster_counts = torch.zeros(n_clusters)
        
        # Pre-compute and store cluster indices
        self.cluster_indices = []
        for i in range(n_clusters):
            indices = torch.where(self.clusters == i)[0]
            self.cluster_indices.append(indices)
            
        # Pre-compute number of batches
        self.n_batches = self.n_samples // self.batch_size

    def _compute_clusters(self):
        # Use quantile-based clustering more efficiently
        quantiles = torch.quantile(self.labels, torch.linspace(0, 1, self.n_clusters + 1))
        clusters = torch.zeros_like(self.labels, dtype=torch.long)
        
        for i in range(self.n_clusters):
            mask = (self.labels >= quantiles[i]) & (self.labels < quantiles[i + 1])
            clusters[mask] = i
            
        return clusters

    def update_weights(self, predictions, targets):
        if not self.adaptive_ratio:
            return
            
        # Update cluster performance more efficiently
        for i in range(self.n_clusters):
            mask = self.clusters == i
            if mask.sum() > 0:
                cluster_preds = predictions[mask]
                cluster_targets = targets[mask]
                mse = F.mse_loss(cluster_preds, cluster_targets)
                self.cluster_performance[i] = mse
                self.cluster_counts[i] += 1
        
        # Update weights based on performance more efficiently
        if self.cluster_counts.sum() > 0:
            perf = self.cluster_performance / (self.cluster_counts + 1e-6)
            weights = 1.0 / (perf + 1e-6)
            self.cluster_weights = weights / weights.sum()

    def __iter__(self):
        # Reset cluster indices
        current_indices = [indices.clone() for indices in self.cluster_indices]
        
        if self.shuffle:
            for indices in current_indices:
                perm = torch.randperm(len(indices))
                indices.copy_(indices[perm])
        
        # Pre-compute available clusters
        available_clusters = [i for i in range(self.n_clusters) if len(current_indices[i]) > 0]
        
        for _ in range(self.n_batches):
            batch_indices = []
            
            while len(batch_indices) < self.batch_size:
                if not available_clusters:
                    # If no clusters have indices, reshuffle all
                    for i in range(self.n_clusters):
                        current_indices[i] = self.cluster_indices[i].clone()
                        if self.shuffle:
                            perm = torch.randperm(len(current_indices[i]))
                            current_indices[i].copy_(current_indices[i][perm])
                    available_clusters = [i for i in range(self.n_clusters) if len(current_indices[i]) > 0]
                
                # Sample cluster based on weights more efficiently
                cluster_weights = self.cluster_weights[available_clusters]
                cluster_weights = cluster_weights / cluster_weights.sum()
                cluster_idx = torch.multinomial(cluster_weights, 1).item()
                cluster = available_clusters[cluster_idx]
                
                # Get index from selected cluster
                idx = current_indices[cluster][0].item()
                batch_indices.append(idx)
                
                # Update cluster indices more efficiently
                current_indices[cluster] = current_indices[cluster][1:]
                if len(current_indices[cluster]) == 0:
                    available_clusters.remove(cluster)

            yield batch_indices

    def __len__(self):
        return self.n_batches

# =====================
# Graph Augmentation Utils
# =====================
def mask_node_features(x, mask_prob=0.05):
    """Randomly mask node features with given probability."""
    mask = torch.rand(x.shape) < mask_prob
    x = x.clone()
    x[mask] = 0
    return x

def perturb_features(x, noise_std=0.01):
    """Add Gaussian noise to node features."""
    noise = torch.randn_like(x) * noise_std
    return x + noise

def perturb_edge_attr(edge_attr, noise_std=0.01):
    """Add Gaussian noise to edge attributes."""
    if edge_attr is None:
        return None
    noise = torch.randn_like(edge_attr) * noise_std
    return edge_attr + noise

def drop_edges(edge_index, edge_attr, drop_prob=0.05):
    """Randomly drop edges from edge_index and corresponding edge attributes."""
    num_edges = edge_index.size(1)
    keep_mask = torch.rand(num_edges) > drop_prob
    edge_index = edge_index[:, keep_mask]
    if edge_attr is not None:
        edge_attr = edge_attr[keep_mask]
    return edge_index, edge_attr

def add_random_edges(edge_index, edge_attr, num_nodes, add_prob=0.01):
    """Randomly add edges between nodes and corresponding edge attributes."""
    num_possible = num_nodes * (num_nodes - 1) // 2
    num_add = int(num_possible * add_prob)
    added = 0
    new_edges = []
    
    # Create new edge attributes with same shape as existing ones
    if edge_attr is not None:
        edge_dim = edge_attr.size(1)
        new_edge_attr = torch.zeros((num_add, edge_dim), 
                                  dtype=edge_attr.dtype, 
                                  device=edge_attr.device)
    else:
        new_edge_attr = None
    
    while added < num_add:
        i = torch.randint(0, num_nodes, (1,)).item()
        j = torch.randint(0, num_nodes, (1,)).item()
        if i != j:
            new_edges.append([i, j])
            added += 1
    
    if new_edges:
        new_edges = torch.tensor(new_edges, dtype=edge_index.dtype, device=edge_index.device).t()
        edge_index = torch.cat([edge_index, new_edges], dim=1)
        if new_edge_attr is not None:
            edge_attr = torch.cat([edge_attr, new_edge_attr], dim=0)
    
    return edge_index, edge_attr

def drop_nodes(x, edge_index, edge_attr, drop_prob=0.05):
    """Randomly drop nodes and their edges from the graph."""
    num_nodes = x.size(0)
    keep_mask = torch.rand(num_nodes) > drop_prob
    keep_indices = torch.where(keep_mask)[0]
    
    if len(keep_indices) == 0:
        return x, edge_index, edge_attr  # Avoid empty graph
        
    x = x[keep_indices]
    
    # Remap edge indices
    idx_map = -1 * torch.ones(num_nodes, dtype=torch.long, device=x.device)
    idx_map[keep_indices] = torch.arange(len(keep_indices), device=x.device)
    
    # Filter edges where both nodes are kept
    mask_edges = keep_mask[edge_index[0]] & keep_mask[edge_index[1]]
    edge_index = edge_index[:, mask_edges]
    edge_index = idx_map[edge_index]
    
    if edge_attr is not None:
        edge_attr = edge_attr[mask_edges]
    
    return x, edge_index, edge_attr

def sample_subgraph(x, edge_index, edge_attr, sample_ratio=0.1):
    """Custom subgraph sampling that maintains graph connectivity."""
    num_nodes = x.size(0)
    num_sample = max(2, int(num_nodes * sample_ratio))
    
    # Start with a random node
    start_node = torch.randint(0, num_nodes, (1,)).item()
    sampled_nodes = {start_node}
    
    # Keep sampling until we have enough nodes
    while len(sampled_nodes) < num_sample:
        # Get all edges connected to currently sampled nodes
        mask = torch.isin(edge_index[0], torch.tensor(list(sampled_nodes))) | \
               torch.isin(edge_index[1], torch.tensor(list(sampled_nodes)))
        connected_edges = edge_index[:, mask]
        
        # Get unique nodes from these edges
        connected_nodes = torch.unique(connected_edges)
        
        # Add a random connected node that we haven't sampled yet
        available_nodes = [n.item() for n in connected_nodes if n.item() not in sampled_nodes]
        if not available_nodes:
            break  # No more connected nodes to sample
        new_node = random.choice(available_nodes)
        sampled_nodes.add(new_node)
    
    # Convert to tensor and sort for consistency
    sampled_nodes = torch.tensor(sorted(list(sampled_nodes)), device=x.device)
    
    # Get the subgraph
    x_sub = x[sampled_nodes]
    
    # Create mapping for edge indices
    idx_map = torch.full((num_nodes,), -1, dtype=torch.long, device=x.device)
    idx_map[sampled_nodes] = torch.arange(len(sampled_nodes), device=x.device)
    
    # Filter edges where both nodes are in sampled set
    mask = torch.isin(edge_index[0], sampled_nodes) & torch.isin(edge_index[1], sampled_nodes)
    edge_index_sub = edge_index[:, mask]
    edge_index_sub = idx_map[edge_index_sub]
    
    # Get corresponding edge attributes
    if edge_attr is not None:
        edge_attr_sub = edge_attr[mask]
    else:
        edge_attr_sub = None
    
    return x_sub, edge_index_sub, edge_attr_sub


