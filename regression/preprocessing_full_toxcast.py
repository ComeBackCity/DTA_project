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
fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)


'''
Note that training and test datasets are the same as GraphDTA
Please see: https://github.com/thinng/GraphDTA
'''

VOCAB_PROTEIN = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, 
				"F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, 
				"O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, 
				"U": 19, "T": 20, "W": 21, 
				"V": 22, "Y": 23, "X": 24, 
				"Z": 25 }


def seqs2int(target):

    return [VOCAB_PROTEIN[s] for s in target] 


class GNNDataset(InMemoryDataset):

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):  
        processed_dir_train = osp.join(self.root, 'processed', 'train')
        processed_dir_val = osp.join(self.root, 'processed', 'val')
        processed_dir_test = osp.join(self.root, 'processed', 'test')
        os.makedirs(processed_dir_train, exist_ok=True)
        os.makedirs(processed_dir_val, exist_ok=True)
        os.makedirs(processed_dir_test, exist_ok=True)
        return [osp.join(processed_dir_train, filename) for filename in os.listdir(processed_dir_train)] + \
                [osp.join(processed_dir_val, filename) for filename in os.listdir(processed_dir_val)] + \
                [osp.join(processed_dir_test, filename) for filename in os.listdir(processed_dir_test)]  

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass


    def process_data(self, df, graph_dict, split, save_dir):
        for itr , row in tqdm(df.iterrows()):
            smi = row['smiles']
            sequence = row['sequence']
            label = row['label']

            mol_graph = graph_dict[smi]
            
            with open("data/full_toxcast_mapping.pkl", "rb") as f:
                mapping = pickle.load(f)
                
            header = mapping[sequence]
            file_pattern = re.compile(rf"{header}_unrelaxed_rank_001_.*\.pdb")

            directory = f"./data/alphafold2/davis/{header}/"
            files = os.listdir(directory)
            
            for f in files:
                if file_pattern.match(f):
                    file_name = os.path.join(directory, f)
                    break       

            protein_graph = uniprot_id_to_structure(file_name)
                        
            # Get Labels
            try:
                data = DATA.Data(
                    mol_graph = mol_graph,
                    protein_graph = protein_graph,
                    y = torch.tensor(label)
                )
            except:
                    print("unable to process: ", smi)
                    
                    
            if self.pre_filter is not None:
                if not self.pre_filter(data):
                    continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, f'{save_dir}/processed_data_{split}_{itr}.pt')
            itr+=1
            # exit()

            

    def process(self):
        df_train = pd.read_csv("./data/full_toxcast/raw/data_train.csv")
        df_test = pd.read_csv("./data/full_toxcast/raw/data_test.csv")
        df = pd.concat([df_train, df_test])

        smiles = df['smiles'].unique()
        graph_dict = dict()
        for smile in tqdm(smiles, total=len(smiles)):
            mol = Chem.MolFromSmiles(smile)
            g = self.mol2graph(mol)
            g = DATA.Data(x=g[0], edge_index=g[1], edge_attr=g[2])
            graph_dict[smile] = g

        save_dir_train = osp.join(self.root, 'processed', 'train')
        os.makedirs(save_dir_train, exist_ok=True)

        save_dir_test = osp.join(self.root, 'processed', 'test')
        os.makedirs(save_dir_test, exist_ok=True)

        self.process_data(df_train, graph_dict, 'train', save_dir_train)
        self.process_data(df_test, graph_dict, 'test', save_dir_test)

    def get_nodes(self, g):
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

    def get_edges(self, g):
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

    def mol2graph(self, mol):
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

        node_attr = self.get_nodes(g)
        edge_index, edge_attr = self.get_edges(g)

        return node_attr, edge_index, edge_attr

if __name__ == "__main__":
    GNNDataset('data/full_toxcast')
    
    
    
