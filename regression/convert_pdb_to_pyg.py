#!/usr/bin/env python3
"""
Protein Graph Builder with Enhanced Node and Edge Features

This script constructs a protein graph from a PDB file using Graphein,
augments each residue node with biophysical, topological, and evolutionary features,
and enriches edge features with spatial and sequence-aware information.

Usage:
    python protein_graph_feature_loader.py --pdb <path_to_pdb> --esm <path_to_esm_embeddings>
"""
import os
import json
import numpy as np
import torch
from torch_geometric.data import Data
from functools import partial

import networkx as nx
from Bio.PDB import PDBParser
# from Bio.SubsMat.MatrixInfo import blosum62 as blosum62_dict

from graphein.protein.config import ProteinGraphConfig
from graphein.protein.edges.distance import (
    add_peptide_bonds,
    add_hydrogen_bond_interactions,
    add_cation_pi_interactions,
    add_ionic_interactions,
    add_disulfide_interactions,
    add_distance_to_edges,
    add_k_nn_edges,
)
from graphein.protein.features.nodes.amino_acid import amino_acid_one_hot, meiler_embedding
from graphein.protein.graphs import construct_graph
from graphein.protein.features.nodes.geometry import add_beta_carbon_vector, add_sequence_neighbour_vector

# Load residue properties and build vocab
with open('./data/residue_features.json', 'r') as f:
    residue_features = json.load(f)

STANDARD_RES = list(residue_features.keys())

# BLOSUM62 = np.zeros((20, 20), dtype=float)
# for i, aa1 in enumerate(STANDARD_RES):
#     for j, aa2 in enumerate(STANDARD_RES):
#         if (aa1, aa2) in blosum62_dict:
#             BLOSUM62[i, j] = blosum62_dict[(aa1, aa2)]
#         elif (aa2, aa1) in blosum62_dict:
#             BLOSUM62[i, j] = blosum62_dict[(aa2, aa1)]
#         else:
#             BLOSUM62[i, j] = 0.0

def one_hot_encode_residue(resname: str) -> np.ndarray:
    vec = np.zeros(len(STANDARD_RES), dtype=float)
    if resname in STANDARD_RES:
        vec[STANDARD_RES.index(resname)] = 1.0
    return vec

def calculate_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

def add_graph_centrality_features(g: nx.Graph):
    deg = dict(g.degree())
    clust = nx.clustering(g)
    btw = nx.betweenness_centrality(g)
    pr = nx.pagerank(g)
    for n, attr in g.nodes(data=True):
        attr['degree'] = float(deg[n])
        attr['clustering'] = clust[n]
        attr['betweenness'] = btw[n]
        attr['pagerank'] = pr[n]

def add_contact_number(g: nx.Graph, threshold: float = 9.0):
    coords = {n: np.array(d['coords']) for n, d in g.nodes(data=True)}
    for n, attr in g.nodes(data=True):
        count = sum(
            np.linalg.norm(coords[n] - coords[m]) < threshold
            for m in coords if m != n
        )
        attr['contact_number'] = float(count)

def add_relative_position(g: nx.Graph):
    N = g.number_of_nodes()
    for idx, (n, attr) in enumerate(g.nodes(data=True)):
        attr['relative_position'] = float(idx) / (N - 1) if N > 1 else 0.0

# def add_blosum62_feature(g: nx.Graph):
#     for n, attr in g.nodes(data=True):
#         aa = attr['residue_name']
#         idx = STANDARD_RES.index(aa) if aa in STANDARD_RES else None
#         attr['blosum62'] = BLOSUM62[idx].tolist() if idx is not None else np.zeros(20).tolist()

def build_edge_features(g: nx.Graph) :
    edge_indices = []
    edge_attrs = []
    LONG_RANGE_THRESHOLD = 12

    node_map = {n: i for i, n in enumerate(g.nodes())}
    coords = {n: np.array(d['coords']) for n, d in g.nodes(data=True)}
    seq_indices = {n: idx for idx, (n, _) in enumerate(g.nodes(data=True))}

    for u, v, a in g.edges(data=True):
        coord_u = coords[u]
        coord_v = coords[v]
        diff = coord_v - coord_u
        dist = np.linalg.norm(diff) + 1e-8
        direction = diff / dist

        idx_u = seq_indices[u]
        idx_v = seq_indices[v]
        seq_sep = abs(idx_v - idx_u)
        is_long_range = float(seq_sep > LONG_RANGE_THRESHOLD)

        kinds = a.get('kind', [])

        edge_indices.append([node_map[u], node_map[v]])
        edge_attrs.append([
            float('knn' in kinds),
            float('peptide_bond' in kinds),
            float('disulfide' in kinds),
            float('hbond' in kinds),
            float('ionic' in kinds),
            float('cation_pi' in kinds),
            a.get('distance', 0.0),
            calculate_angle(diff, coord_u),
            direction[0], direction[1], direction[2],
            seq_sep,
            is_long_range
        ])

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    return edge_index, edge_attr

def convert_nx_to_pyg(g: nx.Graph) -> Data:
    one_hots, meilers, coords = [], [], []
    betas, seqs, bfs = [], [], []
    physchem, degree, clustering = [], [], []
    betweenness, pagerank, contact_num = [], [], []
    relpos = []

    for idx, (n, attr) in enumerate(g.nodes(data=True)):
        one_hots.append(one_hot_encode_residue(attr['residue_name']))
        meilers.append(attr['meiler'])
        coords.append(attr['coords'])
        betas.append(attr['c_beta_vector'])
        seqs.append(attr['sequence_neighbour_vector_n_to_c'])
        bfs.append(attr['b_factor'])
        physchem.append(residue_features[attr['residue_name']])

        degree.append(attr['degree'])
        clustering.append(attr['clustering'])
        betweenness.append(attr['betweenness'])
        pagerank.append(attr['pagerank'])
        contact_num.append(attr['contact_number'])
        relpos.append(attr['relative_position'])

    edge_index, edge_attr = build_edge_features(g)

    return Data(
        one_hot_residues = torch.tensor(one_hots, dtype=torch.float),
        meiler_features  = torch.tensor(meilers, dtype=torch.float),
        x_coords         = torch.tensor(coords, dtype=torch.float),  # EGNN compatible
        pos              = torch.tensor(coords, dtype=torch.float),
        beta_carbon_vector = torch.tensor(betas, dtype=torch.float),
        seq_neighbour_vector = torch.tensor(seqs, dtype=torch.float),
        b_factor         = torch.tensor(bfs, dtype=torch.float),
        physicochemical_feat = torch.tensor(physchem, dtype=torch.float),
        degree           = torch.tensor(degree, dtype=torch.float).unsqueeze(1),
        clustering       = torch.tensor(clustering, dtype=torch.float).unsqueeze(1),
        betweenness      = torch.tensor(betweenness, dtype=torch.float).unsqueeze(1),
        pagerank         = torch.tensor(pagerank, dtype=torch.float).unsqueeze(1),
        contact_number   = torch.tensor(contact_num, dtype=torch.float).unsqueeze(1),
        relative_position = torch.tensor(relpos, dtype=torch.float).unsqueeze(1),
        edge_index       = edge_index,
        edge_attr        = edge_attr
    )

def uniprot_id_to_structure(pdb_path: str, residue_embeddings: np.ndarray, cls_embedding: np.ndarray | None) -> Data | None:
    config = ProteinGraphConfig(
        granularity="CA",
        edge_construction_functions=[
            add_peptide_bonds,
            add_hydrogen_bond_interactions,
            add_cation_pi_interactions,
            add_ionic_interactions,
            add_disulfide_interactions,
            partial(add_k_nn_edges, k=5, long_interaction_threshold=0),
            add_distance_to_edges,
        ],
        node_metadata_functions=[
            amino_acid_one_hot,
            meiler_embedding,
        ],
    )
    g = construct_graph(config=config, path=pdb_path)

    if g is None:
        return None

    add_beta_carbon_vector(g)
    add_sequence_neighbour_vector(g)
    add_graph_centrality_features(g)
    add_contact_number(g)
    add_relative_position(g)
    # add_blosum62_feature(g)

    pyg_graph = convert_nx_to_pyg(g)

    if residue_embeddings is not None:
        pyg_graph.residue_embeddings = torch.tensor(residue_embeddings, dtype=torch.float)
    if cls_embedding is not None:
        pyg_graph.cls_embedding = torch.tensor(cls_embedding, dtype=torch.float)

    return pyg_graph
