import os
import pickle
import torch
import collections

def calculate_mean_std_per_key(data_dir, file_name):
    """
    Calculate mean and std for continuous features in 'x' and 'edge_attr' of a PyG molecule dataset.

    Args:
        data_dir (str): Directory where the molecule .pkl is stored.
        file_name (str): Filename of the molecule .pkl.

    Returns:
        stats (dict): Dictionary with 'mean' and 'std' tensors per feature group.
    """
    all_tensors = collections.defaultdict(list)

    file_path = os.path.join(data_dir, file_name)
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f)

    for data in data_dict.values():
        if hasattr(data, 'x') and torch.is_tensor(data.x):
            # Structure:
            # x = [one-hot (12), categorical (6), continuous (12 + 2 from 3D features), donor/acceptor (2)]
            #   => total = 12 + 6 + 14 + 2 = 34 features
            # Continuous + 3D-derived features = indices 18:32
            continuous_part = data.x[:, 18:32]  # 14 columns (12 + 2 new)
            all_tensors['x_continuous'].append(continuous_part)

        if hasattr(data, 'edge_attr') and torch.is_tensor(data.edge_attr):
            # New edge_attr = 8 features
            # [bond_type, conjugated, aromatic, stereoZ, stereoE, bond length, bond angle, torsion angle]
            all_tensors['edge_attr'].append(data.edge_attr)

    stats = {}

    for key, tensors in all_tensors.items():
        concatenated = torch.cat(tensors, dim=0)
        mean = concatenated.mean(dim=0)
        std = concatenated.std(dim=0)
        stats[key] = {'mean': mean.tolist(), 'std': std.tolist()}

    return stats

# ------------------------------------------------------------------------------
# Example Usage
if __name__ == "__main__":
    for dataset in ['davis', 'kiba']:
        print(f"Processing {dataset} dataset...")
        data_dir = "./data/"
        file_name = f"{dataset}_molecule.pkl"

        stats = calculate_mean_std_per_key(data_dir, file_name)

        # Save to file
        with open(f'./data/{dataset}_molecule_stats.pkl', 'wb') as f:
            pickle.dump(stats, f)

        # Print stats
        for key, value in stats.items():
            print(f"Feature Group: {key}")
            print(f"Mean: {value['mean']}")
            print(f"Std: {value['std']}")
