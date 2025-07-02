import os.path as osp
import numpy as np
import torch
import os
import pandas as pd
from torch_geometric.data import InMemoryDataset
from torch_geometric import data as DATA
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from tqdm import tqdm
import re
# from tdc.multi_pred import DTI
import sys
sys.path.append("../")
from convert_pdb_to_pyg import uniprot_id_to_structure
import gc
import pickle
import gc
import random
from esm2_feature import load_esm2_model, get_esm2_embeddings
from esmc_feature import load_model, get_embeddings
fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)


'''
Note that training and test datasets are the same as GraphDTA
Please see: https://github.com/thinng/GraphDTA
'''

def setup_seed(seed):
    random.seed(seed)                          
    np.random.seed(seed)                       
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True  

def get_pattern(dataset, header):
    if dataset == "davis" and "358" in header:
        file_pattern = re.compile(rf"sequence_358_92648_relaxed_rank_001_alphafold2_ptm_model_2_seed_000.pdb")
    else: 
        file_pattern = re.compile(rf"{header}_unrelaxed_rank_001_.*\.pdb")
        
    return file_pattern

def remove_X_from_sequence(sequence):
    # Replace 'X' with an empty string
    cleaned_sequence = sequence.replace('X', '')
    return cleaned_sequence

def process_data(dataset, model_type="esm2"):
    """
    Process protein sequences and save them as graphs with embeddings.
    
    Args:
    - dataset (str): Name of the dataset
    - model_type (str): Either "esm2" or "esmc" to specify which model to use
    """
    # Load the appropriate model
    model_components = load_model(model_type)

    with open(f"data/{dataset}_mapping.pkl", "rb") as f:
        mapping = pickle.load(f)
        
    save_dir = f"./data/{dataset}/protein_graphs_with_all_embeddings"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print(f"Processing {dataset} protein sequences and saving graphs with embeddings to {save_dir}")

    for sequence, header in tqdm(mapping.items()):
        directory = f"./data/alphafold2/{dataset}/{header}/"
        
        if not os.path.exists(directory):
            print(f"Warning: Directory not found for {header}: {directory}. Skipping.")
            continue
            
        files = os.listdir(directory) 
        save_path = os.path.join(save_dir, f"{header}.pkl")
        
        # if os.path.exists(save_path):
        #     continue
        
        file_pattern = get_pattern(dataset, header)
        file_name = None
        for f in files:
            if file_pattern.match(f):
                file_name = os.path.join(directory, f)
                break  
                
        if file_name is None:
            print(f"Warning: PDB file not found for {header} with pattern {file_pattern.pattern}. Skipping.")
            continue
            
        print(f"Extracting embeddings for {header}...")
        # get_embeddings now returns residue_embeddings and cls_embedding
        residue_embeddings, cls_embedding = get_embeddings(remove_X_from_sequence(sequence), model_components, model_type)
        print(f"Done extracting embeddings for {header}.")


        print(f"Processing PDB for {header}...")
        # Pass residue_embeddings and cls_embedding separately
        protein_graph = uniprot_id_to_structure(pdb_path=file_name, residue_embeddings=residue_embeddings.numpy(), cls_embedding=cls_embedding.numpy() if cls_embedding is not None else None) # convert to numpy
        print(f"Processed PDB for {header}.")
        
        if protein_graph is not None:
            with open(save_path, "wb") as f:
                pickle.dump(protein_graph, f)
            
            print(f"Saved graph for {header} to {save_path}")
        else:
            print(f"Warning: Could not generate protein graph for {header}. Skipping save.")
            
        gc.collect()


if __name__ == "__main__":
    setup_seed(100)
    # process_data("davis", "esm2") # Example for ESM2
    # process_data("kiba", "esm2") # Example for ESM2
    process_data("davis", "esmc")
    process_data("kiba", "esmc")
    # process_data("full_toxcast", "esm2")
    # process_data("full_toxcast", "esmc")
    
    
