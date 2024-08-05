import os
import pickle
import torch
import collections

def calculate_mean_std_per_key(data_dir):
    """
    Calculate the mean and standard deviation per tensor key per column in a directory of PyG Data objects.
    
    Args:
    - data_dir (str): The directory containing the PyG Data objects as .pkl files.
    
    Returns:
    - stats (dict): A dictionary containing the mean and std for each tensor key.
    """
    # Initialize a dictionary to store all tensor values by key
    all_tensors = collections.defaultdict(list)

    for filename in os.listdir(data_dir):
        if filename.endswith('.pkl'):
            with open(os.path.join(data_dir, filename), 'rb') as file:
                data = pickle.load(file)
                
                for key, value in data.items():
                    if key not in {'edge_index', 'pos', 'one_hot_residues'} and torch.is_tensor(value):
                        if key == 'edge_attr':
                            all_tensors[key].append(value[:, -2:])  # Only last two columns
                        else:
                            all_tensors[key].append(value)
    
    # Initialize a dictionary to store the mean and std per key
    stats = {}

    for key, tensors in all_tensors.items():
        concatenated_tensor = torch.cat(tensors, dim=0)
        mean = concatenated_tensor.mean(dim=0)
        std = concatenated_tensor.std(dim=0)
        stats[key] = {'mean': mean, 'std': std}
    
    return stats

# Example usage
for dataset in ['davis', 'kiba', 'full_toxcast']:
    print(dataset)
    data_directory = f"./data/{dataset}/protein_graphs"
    stats = calculate_mean_std_per_key(data_directory)

    # Save stats to a file
    with open(f'./data/{dataset}_stats.pkl', 'wb') as f:
        pickle.dump(stats, f)

    # Print the stats
    for key, value in stats.items():
        print(f"Key: {key}")
        print(f"Mean: {value['mean']}")
        print(f"Std: {value['std']}")
