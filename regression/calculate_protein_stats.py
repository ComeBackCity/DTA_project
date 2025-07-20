import os
import pickle
import torch
import collections

def calculate_mean_std_per_key(data_dir):
    """
    Calculate mean and std for each feature tensor (per column) 
    in a directory of saved PyG Data objects.
    
    Features like one-hot, clustering, relative position are skipped.
    Edge attributes: only last 6 columns normalized.
    """
    # Keys that we should NOT normalize
    keys_to_skip = {
        'one_hot_residues', 'relative_position', 'clustering', 'edge_index', 'pos', 'esm_embeddings'
    }

    all_tensors = collections.defaultdict(list)

    for filename in os.listdir(data_dir):
        if filename.endswith('.pkl'):
            with open(os.path.join(data_dir, filename), 'rb') as f:
                data = pickle.load(f)

                for key, value in data.items():
                    if torch.is_tensor(value):
                        if key in keys_to_skip:
                            continue
                        if key == 'edge_attr':
                            # Only take last 6 columns: distance, angle, dx, dy, dz, seq_sep
                            all_tensors[key].append(value[:, 6:])
                        else:
                            all_tensors[key].append(value)

    # Now calculate mean and std per feature key
    stats = {}

    for key, tensors in all_tensors.items():
        concatenated = torch.cat(tensors, dim=0)
        mean = concatenated.mean(dim=0)
        std = concatenated.std(dim=0)
        stats[key] = {'mean': mean, 'std': std}

    return stats

# Example usage
if __name__ == "__main__":
    for dataset in ['davis', 'kiba', 'full_toxcast']:
        # keep only davis and kiba
        if dataset not in ['davis', 'kiba']:
            continue
        print(f"Processing {dataset}")
        data_directory = f"./data/{dataset}/protein_graphs_with_all_embeddings"
        stats = calculate_mean_std_per_key(data_directory)

        with open(f'./data/{dataset}_stats.pkl', 'wb') as f:
            pickle.dump(stats, f)

        for key, value in stats.items():
            print(f"\nKey: {key}")
            print(f"Mean: {value['mean']}")
            print(f"Std: {value['std']}")
