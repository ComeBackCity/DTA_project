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
import pickle
import random
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

def process(dataset_name):
    if dataset_name == "davis" or dataset_name == 'kiba':
        dataset = DTI(dataset_name)
        dataset.convert_to_log("binding")
        split = dataset.get_split()
        df_train = split['train']
        df_val = split['valid']
        df_test = split['test']
        df = pd.concat([df_train, df_val, df_test])
    elif dataset_name == 'full_toxcast':
        df_train = pd.read_csv("./data/full_toxcast/raw/data_train.csv")
        df_test = pd.read_csv("./data/full_toxcast/raw/data_test.csv")
        df = pd.concat([df_train, df_test])
        
    key_name = "smiles" if dataset_name == "full_toxcast" else "Drug"

    smiles = df[key_name].unique()
    graph_dict = dict()
    for smile in tqdm(smiles, total=len(smiles)):
        mol = Chem.MolFromSmiles(smile)
        g = mol2graph(mol)
        g = DATA.Data(x=g[0], edge_index=g[1], edge_attr=g[2])
        graph_dict[smile] = g

    with open(f"./data/{dataset_name}_molecule.pkl", "wb" ) as f:
        pickle.dump(graph_dict, f)

def get_nodes(g):
    feat = []
    for n, d in g.nodes(data=True):
        h_t = []
        h_t += [int(d['a_type'] == x) for x in ['H', 'C', 'N', 'O', 'F', 'Cl', 'S', 'Br', 'I', ]]
        h_t.append(d['a_num'])
        h_t.append(d['acceptor'])
        h_t.append(d['donor'])
        h_t.append(int(d['aromatic']))
        h_t += [int(d['hybridization'] == x) \
                for x in (Chem.rdchem.HybridizationType.SP, \
                            Chem.rdchem.HybridizationType.SP2,
                            Chem.rdchem.HybridizationType.SP3)]
        h_t.append(d['num_h'])
        # 5 more
        h_t.append(d['ExplicitValence'])
        h_t.append(d['FormalCharge'])
        h_t.append(d['ImplicitValence'])
        h_t.append(d['NumExplicitHs'])
        h_t.append(d['NumRadicalElectrons'])
        feat.append((n, h_t))
    feat.sort(key=lambda item: item[0])
    node_attr = torch.FloatTensor([item[1] for item in feat])

    return node_attr

def get_edges(g):
    e = {}
    for n1, n2, d in g.edges(data=True):
        e_t = [int(d['b_type'] == x)
                for x in (Chem.rdchem.BondType.SINGLE, \
                            Chem.rdchem.BondType.DOUBLE, \
                            Chem.rdchem.BondType.TRIPLE, \
                            Chem.rdchem.BondType.AROMATIC)]

        e_t.append(int(d['IsConjugated'] == False))
        e_t.append(int(d['IsConjugated'] == True))
        e[(n1, n2)] = e_t

    if len(e) == 0:
        return torch.LongTensor([[0], [0]]), torch.FloatTensor([[0, 0, 0, 0, 0, 0]])

    edge_index = torch.LongTensor(list(e.keys())).transpose(0, 1)
    edge_attr = torch.FloatTensor(list(e.values()))
    return edge_index, edge_attr

def mol2graph(mol):
    if mol is None:
        return None
    feats = chem_feature_factory.GetFeaturesForMol(mol)
    g = nx.DiGraph()

    # Create nodes
    for i in range(mol.GetNumAtoms()):
        atom_i = mol.GetAtomWithIdx(i)
        g.add_node(i,
                    a_type=atom_i.GetSymbol(),
                    a_num=atom_i.GetAtomicNum(),
                    acceptor=0,
                    donor=0,
                    aromatic=atom_i.GetIsAromatic(),
                    hybridization=atom_i.GetHybridization(),
                    num_h=atom_i.GetTotalNumHs(),

                    # 5 more node features
                    ExplicitValence=atom_i.GetExplicitValence(),
                    FormalCharge=atom_i.GetFormalCharge(),
                    ImplicitValence=atom_i.GetImplicitValence(),
                    NumExplicitHs=atom_i.GetNumExplicitHs(),
                    NumRadicalElectrons=atom_i.GetNumRadicalElectrons(),
                    )

    for i in range(len(feats)):
        if feats[i].GetFamily() == 'Donor':
            node_list = feats[i].GetAtomIds()
            for n in node_list:
                g.nodes[n]['donor'] = 1
        elif feats[i].GetFamily() == 'Acceptor':
            node_list = feats[i].GetAtomIds()
            for n in node_list:
                g.nodes[n]['acceptor'] = 1

    # Read Edges
    for i in range(mol.GetNumAtoms()):
        for j in range(mol.GetNumAtoms()):
            e_ij = mol.GetBondBetweenAtoms(i, j)
            if e_ij is not None:
                g.add_edge(i, j,
                            b_type=e_ij.GetBondType(),
                            # 1 more edge features 2 dim
                            IsConjugated=int(e_ij.GetIsConjugated()),
                            )

    node_attr = get_nodes(g)
    edge_index, edge_attr = get_edges(g)

    return node_attr, edge_index, edge_attr

if __name__ == "__main__":
    setup_seed(100)
    process("davis")
    process("kiba")
    process("full_toxcast")
    
    
