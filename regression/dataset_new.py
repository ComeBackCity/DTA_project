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

class GNNDataset(Dataset):
    def __init__(self, dataset_name, split='train'):
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split

        # Load dataframes
        if dataset_name in ["davis", "kiba"]:
            self.df = pd.read_csv(f"./data/{dataset_name}/csvs/{dataset_name}_{split}_42.csv")
        elif dataset_name == "full_toxcast":
            self.df = pd.read_csv(f"./data/{dataset_name}/raw/data_{split}.csv")
        else:
            raise ValueError(f"Unknown dataset {dataset_name}")

        # Load graphs
        with open(f"./data/{dataset_name}_molecule.pkl", "rb") as f:
            self.mol_graphs = pickle.load(f)

        with open(f"./data/{dataset_name}_mapping.pkl", "rb") as f:
            self.prot_mapping = pickle.load(f)

        with open(f"./data/{dataset_name}_molecule_stats.pkl", "rb") as f:
            self.mol_stats = pickle.load(f)

        with open(f"./data/{dataset_name}_stats.pkl", "rb") as f:
            self.prot_stats = pickle.load(f)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.dataset_name == 'full_toxcast':
            protein_key, drug_key, label_key = "sequence", "smiles", "label"
        else:
            protein_key, drug_key, label_key = "target_sequence", "compound_iso_smiles", "affinity"

        # Load raw entries
        protein_id = self.df.at[idx, protein_key]
        drug_smiles = self.df.at[idx, drug_key]
        label = torch.tensor(self.df.at[idx, label_key], dtype=torch.float32)

        # ---------------------------
        # Molecule Graph Processing
        # ---------------------------
        mol_data = self.mol_graphs[drug_smiles]
        
        # Standardize continuous molecule features
        x_onehot_cat = mol_data.x[:, :18]  # one-hot + categorical part
        x_continuous = mol_data.x[:, 18:32]  # continuous part (12+2 features)
        x_donor_acceptor = mol_data.x[:, 32:]  # donor/acceptor flags

        x_continuous_std = standardize_tensor(
            x_continuous,
            mean=self.mol_stats['x_continuous']['mean'],
            std=self.mol_stats['x_continuous']['std']
        )

        drug_graph = Data(
            x = torch.cat([x_onehot_cat, x_continuous_std, x_donor_acceptor], dim=1),
            edge_index = mol_data.edge_index,
            edge_attr = standardize_tensor(
                mol_data.edge_attr,
                mean=self.mol_stats['edge_attr']['mean'],
                std=self.mol_stats['edge_attr']['std']
            )
        )

        # ---------------------------
        # Protein Graph Processing
        # ---------------------------
        prot_file = self.prot_mapping[protein_id]
        with open(f"./data/{self.dataset_name}/protein_graphs/with_embeddings/{prot_file}.pkl", "rb") as f:
            prot_data = pickle.load(f)

        # Assemble protein features
        prot_node_feats = torch.cat([
            prot_data.one_hot_residues,
            standardize_tensor(prot_data.meiler_features,
                               mean=self.prot_stats['meiler_features']['mean'],
                               std=self.prot_stats['meiler_features']['std']),
            prot_data.esm_embeddings,
            prot_data.beta_carbon_vector,
            prot_data.seq_neighbour_vector,
            standardize_tensor(prot_data.b_factor.reshape(-1, 1),
                               mean=self.prot_stats['b_factor']['mean'],
                               std=self.prot_stats['b_factor']['std']),
            standardize_tensor(prot_data.physicochemical_feat,
                               mean=self.prot_stats['physicochemical_feat']['mean'],
                               std=self.prot_stats['physicochemical_feat']['std']),
            standardize_tensor(prot_data.degree,
                               mean=self.prot_stats['degree']['mean'],
                               std=self.prot_stats['degree']['std']),
            standardize_tensor(prot_data.betweenness,
                               mean=self.prot_stats['betweenness']['mean'],
                               std=self.prot_stats['betweenness']['std']),
            standardize_tensor(prot_data.pagerank,
                               mean=self.prot_stats['pagerank']['mean'],
                               std=self.prot_stats['pagerank']['std']),
            standardize_tensor(prot_data.contact_number,
                               mean=self.prot_stats['contact_number']['mean'],
                               std=self.prot_stats['contact_number']['std']),
            # Relative position usually kept as-is, not normalized
        ], dim=1)

        protein_graph = Data(
            x = prot_node_feats,
            edge_index = prot_data.edge_index,
            edge_attr = torch.cat([
                prot_data.edge_attr[:, :6],  # bond-type kinds
                standardize_tensor(
                    prot_data.edge_attr[:, 6:],  # distance, angle, dx, dy, dz, seq_sep
                    mean=self.prot_stats['edge_attr']['mean'],
                    std=self.prot_stats['edge_attr']['std']
                )
            ], dim=1)
        )

        return drug_graph, protein_graph, label

    def get_labels(self):
        if self.dataset_name == 'full_toxcast':
            return list(self.df["label"])
        else:
            return list(self.df["affinity"])

        
def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    labels = torch.Tensor([data[2] for data in data_list])
    
    return batchA, batchB, labels

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
    

class BalancedRegressionBatchSampler2(torch.utils.data.Sampler):
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
        """
        An adaptive balanced sampler that uses clustering and dynamic ratios for better sampling.

        Args:
            labels (List[float] or Tensor): Continuous labels.
            batch_size (int): Number of samples per batch.
            n_clusters (int): Number of clusters to divide the label space into.
            shuffle (bool): Whether to shuffle samples within each cluster.
            adaptive_ratio (bool): Whether to use adaptive sampling ratios.
        """
        self.labels = np.array(labels) if not isinstance(labels, np.ndarray) else labels
        self.batch_size = batch_size
        self.n_clusters = n_clusters
        self.shuffle = shuffle
        self.adaptive_ratio = adaptive_ratio

        # Create clusters using quantile-based binning
        self.cluster_edges = np.quantile(self.labels, np.linspace(0, 1, n_clusters + 1))
        self.cluster_edges[-1] += 1e-6  # Ensure the last edge includes the maximum value
        
        # Assign each sample to a cluster
        self.cluster_assignments = np.digitize(self.labels, self.cluster_edges) - 1
        
        # Initialize cluster indices and filter out empty clusters
        self.cluster_indices = {}
        self.cluster_sizes = {}
        self.valid_clusters = []
        
        for i in range(n_clusters):
            mask = self.cluster_assignments == i
            indices = np.where(mask)[0]
            if len(indices) > 0:  # Only keep non-empty clusters
                self.cluster_indices[i] = indices
                self.cluster_sizes[i] = len(indices)
                self.valid_clusters.append(i)
        
        if not self.valid_clusters:
            raise ValueError("No valid clusters found. Try reducing the number of clusters.")
            
        # Initialize performance tracking
        self.cluster_performance = {i: 1.0 for i in self.valid_clusters}
        self.cluster_counts = {i: 0 for i in self.valid_clusters}
        
        # Calculate initial sampling ratios based on cluster sizes
        total_samples = sum(self.cluster_sizes.values())
        self.sampling_ratios = {
            i: size / total_samples for i, size in self.cluster_sizes.items()
        }
        
        # Initialize cluster pointers and shuffle states
        self.cluster_pointers = {i: 0 for i in self.valid_clusters}
        self.cluster_shuffle_states = {
            i: np.arange(size) for i, size in self.cluster_sizes.items()
        }
        
        # Pre-shuffle if enabled
        if self.shuffle:
            for i in self.valid_clusters:
                np.random.shuffle(self.cluster_shuffle_states[i])
        
        # Calculate number of batches
        self.num_batches = math.ceil(total_samples / batch_size)

    def update_performance(self, cluster_id, performance):
        """Update the performance metric for a specific cluster."""
        if self.adaptive_ratio and cluster_id in self.valid_clusters:
            # Update performance with exponential moving average
            alpha = 0.1  # Smoothing factor
            self.cluster_performance[cluster_id] = (
                (1 - alpha) * self.cluster_performance[cluster_id] + 
                alpha * performance
            )
            self.cluster_counts[cluster_id] += 1
            
            # Update sampling ratios based on inverse performance
            total_inverse_perf = sum(1.0 / (p + 1e-6) for p in self.cluster_performance.values())
            self.sampling_ratios = {
                i: (1.0 / (p + 1e-6)) / total_inverse_perf 
                for i, p in self.cluster_performance.items()
            }

    def get_cluster_batch_size(self, cluster_id):
        """Calculate the number of samples to take from a specific cluster."""
        if cluster_id not in self.valid_clusters:
            return 0
        return max(1, int(self.batch_size * self.sampling_ratios[cluster_id]))

    def __iter__(self):
        # Reset cluster pointers
        self.cluster_pointers = {i: 0 for i in self.valid_clusters}
        
        # Reshuffle if enabled
        if self.shuffle:
            for i in self.valid_clusters:
                np.random.shuffle(self.cluster_shuffle_states[i])

        for _ in range(self.num_batches):
            batch = []
            remaining_slots = self.batch_size

            # Try to fill batch from each valid cluster
            for cluster_id in self.valid_clusters:
                if remaining_slots <= 0:
                    break

                # Calculate how many samples to take from this cluster
                n_samples = min(
                    self.get_cluster_batch_size(cluster_id),
                    remaining_slots
                )

                if n_samples <= 0:
                    continue

                # Get indices from the cluster
                indices = self.cluster_indices[cluster_id]
                shuffle_state = self.cluster_shuffle_states[cluster_id]
                
                # Add samples to batch
                for _ in range(n_samples):
                    if self.cluster_pointers[cluster_id] >= len(indices):
                        if self.shuffle:
                            np.random.shuffle(shuffle_state)
                        self.cluster_pointers[cluster_id] = 0
                    
                    idx = indices[shuffle_state[self.cluster_pointers[cluster_id]]]
                    batch.append(idx)
                    self.cluster_pointers[cluster_id] += 1
                    remaining_slots -= 1

            # If we still have slots, fill with samples from the largest valid cluster
            if remaining_slots > 0:
                largest_cluster = max(
                    self.cluster_sizes.items(), 
                    key=lambda x: x[1]
                )[0]
                
                indices = self.cluster_indices[largest_cluster]
                shuffle_state = self.cluster_shuffle_states[largest_cluster]
                
                for _ in range(remaining_slots):
                    if self.cluster_pointers[largest_cluster] >= len(indices):
                        if self.shuffle:
                            np.random.shuffle(shuffle_state)
                        self.cluster_pointers[largest_cluster] = 0
                    
                    idx = indices[shuffle_state[self.cluster_pointers[largest_cluster]]]
                    batch.append(idx)
                    self.cluster_pointers[largest_cluster] += 1

            yield batch

    def __len__(self):
        return self.num_batches