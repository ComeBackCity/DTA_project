import pandas as pd
from Bio import SeqIO

def load_fasta(fasta_file):
    """Load a protein sequence from a FASTA file."""
    with open(fasta_file, "r") as file:
        for record in SeqIO.parse(file, "fasta"):
            return str(record.seq)
    return None

def load_dataframe(file_path, file_type='csv'):
    """Load a dataframe from a CSV or TSV file."""
    if file_type == 'csv':
        return pd.read_csv(file_path)
    elif file_type == 'tsv':
        return pd.read_csv(file_path, sep='\t')

def check_sequence_in_set(sequence, df, set_name, file_type="csv"):
    """Check if the sequence is in the dataframe and return the result with row number."""
    if file_type == "csv":
        col = 'target_sequence'
    else:
        col = 'X2'
        
    match = df[df[col] == sequence]
    print(match)
    if not match.empty:
        row_number = match.index[0]
        return f"The sequence is in the {set_name} set at row {row_number}.", match
    else:
        return f"The sequence is not in the {set_name} set.", None

def main():
    fasta_file = "regression/data/fastas/davis/sequence_348.fasta"  # Change this to your FASTA file path
    train_csv = "regression/data/davis/raw/data_train.csv"  # Change this to your training CSV file path
    test_csv = "regression/data/davis/raw/data_test.csv"  # Change this to your testing CSV file path
    tsv_file = "regression/data/data/davis.tab"  # Change this to your TSV file path

    # Load the protein sequence from the FASTA file
    protein_sequence = load_fasta(fasta_file)
    if protein_sequence is None:
        print("No sequence found in the FASTA file.")
        return

    # Load the training, testing, and TSV dataframes
    train_df = load_dataframe(train_csv, 'csv')
    test_df = load_dataframe(test_csv, 'csv')
    tsv_df = load_dataframe(tsv_file, 'tsv')
    
    print(tsv_df['ID2'])

    # Check if the sequence is in the train or test set
    train_result, _ = check_sequence_in_set(protein_sequence, train_df, "train")
    test_result, _ = check_sequence_in_set(protein_sequence, test_df, "test")

    # Check if the sequence is in the TSV file
    tsv_result, tsv_match = check_sequence_in_set(protein_sequence, tsv_df, "tsv", "tsv")

    # Print the results
    print(train_result)
    print(test_result)
    print(tsv_result)
    if tsv_match is not None:
        print("Matching row in TSV file:")
        print(tsv_match)

if __name__ == "__main__":
    main()
