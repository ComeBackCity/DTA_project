import os
import pickle

def read_single_sequence_fasta(file_path):
    sequence = ""
    header = None
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                header = line[1:]  # Remove '>' character
            else:
                sequence += line
    
    return header, sequence

if __name__ == "__main__":
    all_fastas = os.listdir("fastas/full_toxcast")
    mapping = {}
    for file_name in all_fastas:
        header, seq = read_single_sequence_fasta(os.path.join("fastas/full_toxcast", file_name))
        mapping = {seq : header}
        
    with open("full_toxcast_mapping.pkl", "wb") as f:
        pickle.dump(mapping, f)
        
        
