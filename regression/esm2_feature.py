import os
import pandas as pd
from Bio import SeqIO

def save_sequences_as_fasta(dataframe, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sequences = set(dataframe['Sequence'])
    for i, seq in enumerate(sequences):
        with open(os.path.join(output_dir, f"sequence_{i}.fasta"), "w") as f:
            f.write(f">sequence_{i}\n{seq}")

def main(dataframe_path, output_dir):
    # Read dataframe
    df = pd.read_csv(dataframe_path)

    # Drop duplicates
    df.drop_duplicates(subset=['Sequence'], inplace=True)

    # Save sequences as FASTA files
    save_sequences_as_fasta(df, output_dir)

if __name__ == "__main__":
    dataframe_path = "regression/data/full_toxcast/raw/data.csv"
    output_dir = "path/to/output/directory"
    main(dataframe_path, output_dir)
