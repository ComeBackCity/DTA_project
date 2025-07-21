import os
import torch
import collections

def calculate_mean_std_per_key(data_dir, file_name):
    """
    Calculate mean and std for continuous features in 'x' and 'edge_attr' of a PyG molecule dataset.

    Args:
        data_dir (str): Directory where the molecule .pt is stored.
        file_name (str): Filename of the molecule .pt.

    Returns:
        stats (dict): Dictionary with 'mean' and 'std' tensors per feature group.
    """
    all_tensors = collections.defaultdict(list)

    file_path = os.path.join(data_dir, file_name)
    data_dict = torch.load(file_path, weights_only=False)

    for data in data_dict.values():
        if hasattr(data, 'x') and torch.is_tensor(data.x):
            # x: [one-hot (12), categorical (6), continuous (12 + 2 from 3D features), donor/acceptor (2)]
            continuous_part = data.x[:, 18:32]  # 14 columns
            all_tensors['x_continuous'].append(continuous_part)

        if hasattr(data, 'edge_attr') and torch.is_tensor(data.edge_attr):
            # edge_attr: 8 features
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
        file_name = f"{dataset}_molecule_graph_and_chemfm.pt"  # <- Changed from .pkl to .pt

        stats = calculate_mean_std_per_key(data_dir, file_name)

        # Save to .pt
        output_file = f'./data/{dataset}_molecule_stats.pt'
        torch.save(stats, output_file)

        # Print stats
        for key, value in stats.items():
            print(f"Feature Group: {key}")
            print(f"Mean: {value['mean']}")
            print(f"Std: {value['std']}")
