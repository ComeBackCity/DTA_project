import os
import os.path as osp
import random
import pickle
import numpy as np
import pandas as pd
import torch
from torch_geometric import data as DATA
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures, AllChem
from rdkit import RDConfig
from transformers import AutoTokenizer, AutoModel

# ChemFM model load
model_name = "ChemFM/ChemFM-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModel.from_pretrained(model_name).eval().to("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def extract_chemfm_features(smiles: str):
    inputs = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].cpu()
    token_embeddings = outputs.last_hidden_state[:, 1:, :].squeeze(0).cpu()
    return token_embeddings, cls_embedding

# Chemical feature factory
fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

# Precomputed properties (same as before)
electronegativity = { 'H': 2.20, 'B': 2.04, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
                      'Si': 1.90, 'P': 2.19, 'S': 2.58, 'Cl': 3.16, 'Br': 2.96, 'I': 2.66 }

vdw_radius = { 'H': 1.20, 'B': 2.00, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47,
               'Si': 2.10, 'P': 1.80, 'S': 1.80, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98 }

atomic_properties = {
    'H': {'atomic_mass': 1.008, 'a_num': 1},
    'B': {'atomic_mass': 10.81, 'a_num': 5},
    'C': {'atomic_mass': 12.011, 'a_num': 6},
    'N': {'atomic_mass': 14.007, 'a_num': 7},
    'O': {'atomic_mass': 15.999, 'a_num': 8},
    'F': {'atomic_mass': 18.998, 'a_num': 9},
    'Si': {'atomic_mass': 28.085, 'a_num': 14},
    'P': {'atomic_mass': 30.974, 'a_num': 15},
    'S': {'atomic_mass': 32.06, 'a_num': 16},
    'Cl': {'atomic_mass': 35.45, 'a_num': 17},
    'Br': {'atomic_mass': 79.904, 'a_num': 35},
    'I': {'atomic_mass': 126.904, 'a_num': 53}
}

# Helper functions
def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def compute_gasteiger(mol):
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except:
        pass

def float_flag(x): return 1.0 if x else 0.0

def distance(xyz1, xyz2):
    return np.linalg.norm(np.array(xyz1) - np.array(xyz2))

def angle(xyz1, xyz2, xyz3):
    v1 = np.array(xyz1) - np.array(xyz2)
    v2 = np.array(xyz3) - np.array(xyz2)
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.arccos(np.clip(cosine_angle, -1.0, 1.0))

def torsion(xyz1, xyz2, xyz3, xyz4):
    p0 = np.array(xyz1)
    p1 = np.array(xyz2)
    p2 = np.array(xyz3)
    p3 = np.array(xyz4)
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1 /= (np.linalg.norm(b1) + 1e-8)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.arctan2(y, x)

def get_atom_features(atom, mol, coords, centroid):
    sym = atom.GetSymbol()
    idx = atom.GetIdx()
    props = atomic_properties.get(sym, {'atomic_mass': 0.0, 'a_num': 0})
    features = [float_flag(sym == x) for x in atomic_properties.keys()]
    features.extend([
        float_flag(atom.GetIsAromatic()),
        float_flag(atom.IsInRing()),
        float_flag(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP),
        float_flag(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP2),
        float_flag(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3),
        float_flag(atom.HasProp('_CIPCode')),
        float(atom.GetTotalNumHs()),
        float(atom.GetDegree()),
        float(atom.GetExplicitValence()),
        float(atom.GetFormalCharge()),
        float(atom.GetImplicitValence()),
        float(atom.GetNumRadicalElectrons()),
        float(props['atomic_mass']),
        float(props['a_num']),
        float(electronegativity.get(sym, 0.0)),
        float(vdw_radius.get(sym, 0.0)),
        float(atom.GetProp('_GasteigerCharge')) if atom.HasProp('_GasteigerCharge') else 0.0,
        float(mol.GetRingInfo().NumAtomRings(idx)),
        distance(coords[idx], centroid),
        sum(distance(coords[idx], coords[j]) < 3.0 for j in range(len(coords)) if j != idx),
        0.0, 0.0  # Donor, Acceptor placeholders
    ])
    return features

def get_bond_features(bond, coords):
    u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    bond_length = distance(coords[u], coords[v])
    bond_angle = 0.0
    torsion_angle = 0.0
    u_neighbors = [n.GetIdx() for n in bond.GetBeginAtom().GetNeighbors() if n.GetIdx() != v]
    v_neighbors = [n.GetIdx() for n in bond.GetEndAtom().GetNeighbors() if n.GetIdx() != u]
    if u_neighbors:
        bond_angle = angle(coords[u_neighbors[0]], coords[u], coords[v])
    if len(u_neighbors) > 0 and len(v_neighbors) > 0:
        torsion_angle = torsion(coords[u_neighbors[0]], coords[u], coords[v], coords[v_neighbors[0]])
    return [
        float(bond.GetBondTypeAsDouble()),
        float_flag(bond.GetIsConjugated()),
        float_flag(bond.GetIsAromatic()),
        float_flag(bond.GetStereo() == Chem.rdchem.BondStereo.STEREOZ),
        float_flag(bond.GetStereo() == Chem.rdchem.BondStereo.STEREOE),
        bond_length,
        bond_angle,
        torsion_angle
    ]

def mol2graph(mol):
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol, randomSeed=42)
        coords = mol.GetConformer().GetPositions()
    except:
        return None
    centroid = np.mean(coords, axis=0)
    compute_gasteiger(mol)
    import networkx as nx
    g = nx.Graph()
    for i, atom in enumerate(mol.GetAtoms()):
        g.add_node(i, features=get_atom_features(atom, mol, coords, centroid))
    feats = chem_feature_factory.GetFeaturesForMol(mol)
    for f in feats:
        family = f.GetFamily()
        for atom_id in f.GetAtomIds():
            if family == 'Donor':
                g.nodes[atom_id]['features'][-2] = 1.0
            elif family == 'Acceptor':
                g.nodes[atom_id]['features'][-1] = 1.0
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        g.add_edge(u, v, features=get_bond_features(bond, coords))
    x = torch.tensor([g.nodes[i]['features'] for i in g.nodes()], dtype=torch.float)
    if g.edges():
        edge_index = torch.tensor([[e[0], e[1]] for e in g.edges()], dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor([e[2]['features'] for e in g.edges(data=True)], dtype=torch.float)
    else:
        edge_index = torch.LongTensor([[0], [0]])
        edge_attr = torch.FloatTensor([[0] * 8])
    return DATA.Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# Process Entry Point
def process(dataset_name):
    if dataset_name in ["davis", "kiba"]:
        df_train = pd.read_csv(f"./data/{dataset_name}/csvs/{dataset_name}_train_42.csv")
        df_valid = pd.read_csv(f"./data/{dataset_name}/csvs/{dataset_name}_valid_42.csv")
        df_test = pd.read_csv(f"./data/{dataset_name}/csvs/{dataset_name}_test_42.csv")
        df = pd.concat([df_train, df_valid, df_test])
        smiles_list = df["compound_iso_smiles"].unique()
    elif dataset_name == "full_toxcast":
        df_train = pd.read_csv("./data/full_toxcast/raw/data_train.csv")
        df_test = pd.read_csv("./data/full_toxcast/raw/data_test.csv")
        df = pd.concat([df_train, df_test])
        smiles_list = df["smiles"].unique()
    else:
        raise ValueError("Unsupported dataset")

    graph_dict = {}

    print(f"Processing {dataset_name} molecules...")
    for smile in tqdm(smiles_list):
        mol = Chem.MolFromSmiles(smile)
        if mol:
            graph_data = mol2graph(mol)
            if graph_data is not None:
                try:
                    token_embeddings, cls_embedding = extract_chemfm_features(smile)
                    graph_data.token_embeddings = token_embeddings
                    graph_data.cls_embedding = cls_embedding
                    graph_dict[smile] = graph_data
                except Exception as e:
                    print(f"Error processing SMILES {smile}: {e}")
                    continue

    # Save the dictionary as .pt file
    output_path = f"./data/{dataset_name}_molecule_graph_and_chemfm.pt"
    print(f"Saving processed molecules to {output_path}")
    torch.save(graph_dict, output_path)

# Run
if __name__ == "__main__":
    setup_seed(100)
    process("davis")
    process("kiba")
    # process("full_toxcast") # Uncomment if needed
