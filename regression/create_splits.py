import pandas as pd
import argparse
import os


def create_fold(df, fold_seed, frac):
    """Standard random split

    Args:
        df (pd.DataFrame): dataset dataframe
        fold_seed (int): random seed
        frac (list): [train_fraction, valid_fraction, test_fraction]

    Returns:
        dict: dict with keys 'train', 'valid', 'test' mapping to respective DataFrames
    """
    train_frac, val_frac, test_frac = frac
    test = df.sample(frac=test_frac, replace=False, random_state=fold_seed)
    train_val = df.drop(test.index)
    val = train_val.sample(frac=val_frac / (1 - test_frac), replace=False, random_state=fold_seed)
    train = train_val.drop(val.index)

    return {
        "train": train.reset_index(drop=True),
        "valid": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="davis")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="../Data")

    args = parser.parse_args()

    # Load dataset
    csv_path = os.path.join(args.data_dir, f"{args.dataset_name}.csv")
    df = pd.read_csv(csv_path)

    # Create standard fold
    fold = create_fold(df, args.seed, [0.8, 0.1, 0.1])

    # Save splits
    for split in ['train', 'valid', 'test']:
        out_path = os.path.join(args.data_dir, f"{args.dataset_name}_{split}_{args.seed}.csv")
        fold[split].to_csv(out_path, index=False)

    print(f"{args.dataset_name} standard split done! Shapes - "
          f"Train: {fold['train'].shape}, Valid: {fold['valid'].shape}, Test: {fold['test'].shape}")
