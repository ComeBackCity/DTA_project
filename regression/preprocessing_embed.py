import os.path as osp
import os
import numpy as np
import torch
import pandas as pd
from torch_geometric.data import InMemoryDataset, Dataset, Data
import torch.nn.functional as F
from torch_geometric import data as DATA
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from tqdm import tqdm
import re
import gc
from transformers import T5EncoderModel, T5Tokenizer
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

# def getProteinLanguageModelEmbeddings(target):
#     pass

# def get_t5_model(device):
#     model = T5EncoderModel.from_pretrained("../protT5/protT5_checkpoint/", torch_dtype=torch.float16)
#     model = model.to(device) # move model to GPU
#     model = model.eval() # set model to evaluation model
#     tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False ) 

#     return model, tokenizer

#@title Load encoder-part of ProtT5 in half-precision. { display-mode: "form" }
# Load ProtT5 in half-precision (more specifically: the encoder-part of ProtT5-XL-U50) 
# def get_T5_model_enc(device):
#     model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
#     model = model.to(device) # move model to GPU
#     model = model.eval() # set model to evaluation model
#     tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

#     return model, tokenizer


class GNNDataset(Dataset):

    def __init__(self, root, train=True, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        if train:
            self.data_div = 'train'
            # self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.data_div = 'test'
            # self.data, self.slices = torch.load(self.processed_paths[1])


    @property
    def raw_file_names(self):
        return ['data_train.csv', 'data_test.csv', 'prot5.pth', 'esm2.pth']

    @property
    def processed_file_names(self):

        # return ['train/processed_data_train_1.pt', 'train/processed_data_train_2.pt',
        #         'test/processed_data_test_1.pt', 'test/processed_data_test_2.pt', ...]
    

        processed_dir_train = osp.join(self.root, 'processed', 'train')
        processed_dir_test = osp.join(self.root, 'processed', 'test')
        os.makedirs(processed_dir_train, exist_ok=True)
        os.makedirs(processed_dir_test, exist_ok=True)
        return [osp.join(processed_dir_train, filename) for filename in os.listdir(processed_dir_train)] + \
                [osp.join(processed_dir_test, filename) for filename in os.listdir(processed_dir_test)] 
    
        # if self.data_div == 'train':
        #     return ['train/processed_data_train_1.pt', 'train/processed_data_train_2.pt', ...,]
        # else:
        #     return ['train/processed_data_train_1.pt', 'train/processed_data_train_2.pt', ...,]

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def len(self) -> int:
        return len()
    
    def get(self, idx: int) -> Data:
        return super().get(idx)

    def process_data(self, data_path, raw_data_dir, graph_dict, mode, save_dir):
        df = pd.read_csv(data_path)
        prot5_path = self.raw_paths[2]
        esm2_path = self.raw_paths[3]
        prot5_map = torch.load(prot5_path)
        esm2_map = torch.load(esm2_path)
        print(f"length of esm2 map is = {len(esm2_map)}")

        # data_list = []
        itr = 0
        for _ , row in tqdm(df.iterrows()):
            smi = row['compound_iso_smiles']
            sequence = row['target_sequence']
            label = row['affinity']

            x, edge_index, edge_attr = graph_dict[smi]

            # caution
            x = (x - x.min()) / (x.max() - x.min())

            target = seqs2int(sequence)
            target_len = 1200
            if len(target) < target_len:
                target = np.pad(target, (0, target_len- len(target)))
            else:
                target = target[:target_len]

            prot5_embedding = prot5_map[sequence]
            pad_length = target_len - prot5_embedding.shape[0] 
            prot5_embedding = F.pad(prot5_embedding, (0, 0, 0, pad_length), 'constant', 0)
            prot5_embedding = torch.unsqueeze(prot5_embedding, dim=0).to(torch.float16)
            mask = (prot5_embedding.sum(dim=2) != 0).to(torch.float16)

            esm2_embedding = esm2_map[sequence]

            # print(prot5_embedding.shape)
            # print(esm2_embedding.shape)

            try:
                data = DATA.Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=torch.FloatTensor([label]),
                    target=torch.LongTensor([target]),
                    prot5_embedding = prot5_embedding,
                    esm2_embedding = esm2_embedding,
                    mask=mask
                )

            except:
                    print("unable to process: ", smi)


            if self.pre_filter is not None:
                if not self.pre_filter(data):
                    continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, f'{save_dir}/processed_data_{mode}_{itr}.pt')
            itr+=1

            # data_list.append(data)

            # del data, mask, prot5_embedding
            # gc.collect(1)

        # return data_list

    def process(self):
        df_train = pd.read_csv(self.raw_paths[0])
        df_test = pd.read_csv(self.raw_paths[1])
        df = pd.concat([df_train, df_test])

        smiles = df['compound_iso_smiles'].unique()
        graph_dict = dict()
        for smile in tqdm(smiles, total=len(smiles)):
            mol = Chem.MolFromSmiles(smile)
            g = self.mol2graph(mol)
            graph_dict[smile] = g

        save_dir_train = osp.join(self.root, 'processed', 'train')
        os.makedirs(save_dir_train, exist_ok=True)

        save_dir_test = osp.join(self.root, 'processed', 'test')
        os.makedirs(save_dir_test, exist_ok=True)

        self.process_data(self.raw_paths[0], graph_dict, 'train', save_dir_train)
        self.process_data(self.raw_paths[1], graph_dict, 'test', save_dir_test)

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
    GNNDataset('data/davis')
    # GNNDataset('data/kiba')
