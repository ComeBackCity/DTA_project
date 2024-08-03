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
    all_fastas = os.listdir("fastas/kiba")
    mapping = {}
    for file_name in all_fastas:
        header, seq = read_single_sequence_fasta(os.path.join("fastas/kiba", file_name))
        mapping.update({seq : header})
        
    with open("kiba_mapping.pkl", "wb") as f:
        pickle.dump(mapping, f)
        
        
