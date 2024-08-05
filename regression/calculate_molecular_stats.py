import os
import pickle
import torch
import collections

def calculate_mean_std_per_key(data_dir, file_name):
    """
    Calculate the mean and standard deviation per tensor key per column in a PyG Data object stored in a single .pkl file.
    Specifically focuses on the last 9 columns of the tensor 'x' and the last 2 columns of 'edge_attr'.
    
    Args:
    - data_dir (str): The directory containing the PyG Data object as a .pkl file.
    - file_name (str): The name of the .pkl file containing the PyG Data object.
    
    Returns:
    - stats (dict): A dictionary containing the mean and std for each tensor key.
    """
    # Initialize a dictionary to store all tensor values by key
    all_tensors = collections.defaultdict(list)

    file_path = os.path.join(data_dir, file_name)
    with open(file_path, 'rb') as file:
        data_dict = pickle.load(file)
        
        for data in data_dict.values():
            for key, value in data.items():
                if key == 'x' and torch.is_tensor(value):
                    # Append only the last 9 columns of 'x'
                    all_tensors[key].append(torch.log(value[:, -2:]))
                elif key not in {'edge_index', 'edge_attr'} and torch.is_tensor(value):
                    # Append other tensor types (excluding the ones to ignore)
                    all_tensors[key].append(value)
    
    # Initialize a dictionary to store the mean and std per key
    stats = {}

    for key, tensors in all_tensors.items():
        concatenated_tensor = torch.cat(tensors, dim=0)
        mean = concatenated_tensor.mean(dim=0)
        std = concatenated_tensor.std(dim=0)
        stats[key] = {'mean': mean.tolist(), 'std': std.tolist()}
    
    return stats

# Example usage
for dataset in ['davis', 'kiba', 'full_toxcast']:
    print(f"Processing dataset: {dataset}")
    data_directory = f"./data/"
    file_name = f"{dataset}_molecule.pkl"
    stats = calculate_mean_std_per_key(data_directory, file_name)

    # Save stats to a file
    with open(f'./data/{dataset}_mostats.pkl', 'wb') as f:
        pickle.dump(stats, f)

    # Print the stats
    for key, value in stats.items():
        print(f"Key: {key}")
        print(f"Mean: {value['mean']}")
        print(f"Std: {value['std']}")
