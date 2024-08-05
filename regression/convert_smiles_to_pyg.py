import networkx as nx
import torch
from torch_geometric.data import Data
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import graphein.molecule as gm

from functools import partial

def smiles_to_structure():
    config = gm.MoleculeGraphConfig(
        node_metadata_functions=[
            gm.atom_type_one_hot,
            gm.atomic_mass,
            gm.degree,
            gm.total_degree,
            gm.total_valence,
            gm.explicit_valence,
            gm.implicit_valence,
            gm.num_explicit_h,
            gm.num_implicit_h,
            gm.total_num_h,
            gm.num_radical_electrons,
            gm.formal_charge,
            gm.hybridization,
            gm.is_aromatic,
            gm.is_isotope,
            gm.is_ring,
            gm.chiral_tag,
            partial(gm.is_ring_size, ring_size=5),
            partial(gm.is_ring_size, ring_size=7)
        ],
        edge_metadata_functions=[
            gm.add_bond_type,
            gm.bond_is_aromatic,
            gm.bond_is_conjugated,
            gm.bond_is_in_ring,
            gm.bond_stereo,
            partial(gm.bond_is_in_ring_size, ring_size=5),
            partial(gm.bond_is_in_ring_size, ring_size=7)
        ]
    )

    g = gm.construct_graph(smiles="CC(=O)OC1=CC=CC=C1C(=O)O", config=config)
    for n, d in g.nodes(data=True):
        print(d)
    for u, v, d in g.edges(data=True):
        print(d)
        

if __name__ == "__main__":
    smiles_to_structure()