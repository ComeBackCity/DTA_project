import itertools
import subprocess
import sys

# Define the grid of hyperparameters
param_grid = {
    'lr': [5e-5, 1e-5, 5e-4, 1e-4, 5e-4, 1e-3],
    'hidden_dim': [64, 128, 256 ,512],
    'prot_layers': [3, 4, 5, 6, 7],
    'prot_gvp_layer': [3, 4, 5, 6, 7],
    'drug_layers': [2, 3, 4, 5, 6],
    'dropout': [0.1, 0.15, 0.2],
    'weight_decay': [0.001, 0.005, 0.01]
}

# Generate all combinations
keys, values = zip(*param_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

results = []

for i, params in enumerate(combinations):
    cmd = [
        sys.executable, 'train_hyperparameter.py',
        '--dataset', 'davis',
        '--lr', str(params['lr']),
        '--weight_decay', str(params['weight_decay']),
        '--batch_size', str(params['batch_size']),
        '--epochs', '500',
        '--prot_layers', str(params['prot_layers']),
        '--prot_gvp_layer', str(params['prot_gvp_layer']),
        '--drug_layers', str(params['drug_layers']),
        '--dropout', str(params['dropout']),
        '--save_model',
        # Add more args as needed
    ]
    print(f"Running combination {i+1}/{len(combinations)}: {params}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    val_loss = None
    for line in result.stdout.splitlines():
        if 'VAL_LOSS:' in line:
            try:
                val_loss, val_cindex = float(line.split('VAL_LOSS:')[1].strip())
            except Exception:
                pass
    results.append((params, val_loss))
    print(f"  Result: VAL_LOSS={val_loss}")

# save all the hyper parameter combinations
with open('hyperparameter_results.txt', 'w') as f:
    for params, val_loss in results:
        f.write(f"{params}, {val_loss}\n")  