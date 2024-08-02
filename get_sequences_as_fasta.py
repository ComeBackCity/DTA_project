import os
import pandas as pd
from Bio import SeqIO

def save_sequences_as_fasta(dataframe, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sequences = set(dataframe['sequence'])
    for i, seq in enumerate(sequences):
        with open(os.path.join(output_dir, f"sequence_{i}.fasta"), "w") as f:
            f.write(f">sequence_{i}\n{seq}")

def main(train_path, test_path, output_dir):
    # Read dataframe
    df1 = pd.read_csv(train_path, usecols=['target_sequence'])
    print(len(df1))
    df2 = pd.read_csv(test_path, usecols=['target_sequence'])
    print(len(df2))
    df = pd.concat([df1, df2])

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Save sequences as FASTA files
    # save_sequences_as_fasta(df, output_dir)

if __name__ == "__main__":
    train_path = "regression/data/kiba/raw/data_train.csv"
    test_path = "regression/data/kiba/raw/data_test.csv"
    output_dir = "regression/data/fastas/full_toxcast"
    main(train_path, test_path, output_dir)
