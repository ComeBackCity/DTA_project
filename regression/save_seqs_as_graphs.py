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
from tdc.multi_pred import DTI
import sys
sys.path.append("../")
from convert_pdb_to_pyg import uniprot_id_to_structure
import gc
import pickle
import gc
import random
from esm2_feature import load_esm2_model, get_esm2_embeddings
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
        file_pattern = re.compile(rf"AF-Q9BRL4-F1-model_v4.pdb")
    else: 
        file_pattern = re.compile(rf"{header}_unrelaxed_rank_001_.*\.pdb")
        
    return file_pattern

def remove_X_from_sequence(sequence):
    # Replace 'X' with an empty string
    cleaned_sequence = sequence.replace('X', '')
    return cleaned_sequence

def process_data(dataset):
    
    tokenizer, model, device = load_esm2_model()
    
    with open(f"data/{dataset}_mapping.pkl", "rb") as f:
        mapping = pickle.load(f)
        
    save_dir = f"./data/{dataset}/protein_graphs/"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for sequence, header in mapping.items():
        
        directory = f"./data/alphafold2/{dataset}/{header}/"
        files = os.listdir(directory) 
        save_path = os.path.join(save_dir, f"{header}.pkl")
        
        if os.path.exists(save_path):
            continue
        
        file_pattern = get_pattern(dataset, header)
        for f in files:
            if file_pattern.match(f):
                file_name = os.path.join(directory, f)
                break  
            
        if header == "sequence_348":
            embeddings = get_esm2_embeddings(remove_X_from_sequence(sequence), tokenizer, model, device)
        else:
            embeddings = get_esm2_embeddings(remove_X_from_sequence(sequence), tokenizer, model, device)
        
        protein_graph = uniprot_id_to_structure(file_path=file_name, embeddings = embeddings)
    
        with open(save_path, "wb") as f:
            pickle.dump(protein_graph, f)
    
        gc.collect()            


if __name__ == "__main__":
    setup_seed(100)
    process_data("davis")
    process_data("kiba")
    process_data("full_toxcast")
    
    
