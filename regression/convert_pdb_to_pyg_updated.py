#!/usr/bin/env python3
"""
Protein Graph Builder with Biophysical, Topological, and GVP Features.

This module constructs protein graphs from PDB files using Graphein,
adding both traditional residue features and GVP-compatible geometric features.

Outputs include:
- Biophysical & centrality features
- Sequence-aware features
- Geometric Vector Perceptron (GVP) node & edge features

Usage:
    python protein_graph_feature_loader.py --pdb <path_to_pdb> --esm <path_to_esm_embeddings>
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from functools import partial
import networkx as nx

from graphein.protein.config import ProteinGraphConfig
from graphein.protein.edges.distance import (
    add_peptide_bonds, add_hydrogen_bond_interactions, add_cation_pi_interactions,
    add_ionic_interactions, add_disulfide_interactions, add_distance_to_edges, add_k_nn_edges
)
from graphein.protein.features.nodes.amino_acid import amino_acid_one_hot, meiler_embedding
from graphein.protein.graphs import construct_graph
from graphein.protein.features.nodes.geometry import add_beta_carbon_vector, add_sequence_neighbour_vector

# ============================
# Load Residue Features
# ============================

with open('./data/residue_features.json', 'r') as f:
    residue_features = json.load(f)

STANDARD_RES = list(residue_features.keys())

# ============================
# Helper Functions (Your Originals)
# ============================

def one_hot_encode_residue(resname: str) -> np.ndarray:
    """One-hot encode a residue name."""
    vec = np.zeros(len(STANDARD_RES), dtype=float)
    if resname in STANDARD_RES:
        vec[STANDARD_RES.index(resname)] = 1.0
    return vec

def calculate_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate angle between two 3D vectors."""
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

def add_graph_centrality_features(g: nx.Graph):
    """Add degree, clustering, betweenness, and pagerank to node attributes."""
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
    """Add contact number feature based on spatial distance."""
    coords = {n: np.array(d['coords']) for n, d in g.nodes(data=True)}
    for n, attr in g.nodes(data=True):
        count = sum(np.linalg.norm(coords[n] - coords[m]) < threshold for m in coords if m != n)
        attr['contact_number'] = float(count)

def add_relative_position(g: nx.Graph):
    """Add normalized relative position in sequence order to nodes."""
    N = g.number_of_nodes()
    for idx, (n, attr) in enumerate(g.nodes(data=True)):
        attr['relative_position'] = float(idx) / (N - 1) if N > 1 else 0.0

# ============================
# GVP-Compatible Feature Builders
# ============================

def _normalize(x):
    """L2 normalize a tensor along the last dimension."""
    return F.normalize(x, dim=-1)

def radial_basis(d, D_min=0., D_max=20., D_count=16):
    """Radial basis encoding of distances."""
    D_mu = torch.linspace(D_min, D_max, D_count, device=d.device)
    D_sigma = (D_max - D_min) / D_count
    return torch.exp(-((d.unsqueeze(-1) - D_mu) / D_sigma) ** 2)

def positional_encoding(offsets, num_embeddings=16):
    """Sinusoidal positional encoding for sequence separation."""
    frequency = torch.exp(torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=offsets.device) * -(np.log(10000.0) / num_embeddings))
    angles = offsets.unsqueeze(-1) * frequency
    return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)

# ============================
# Edge Feature Construction
# ============================

def build_edge_features(g: nx.Graph):
    """
    Build scalar and vector edge features for GVP.
    - Scalar: interaction types, distance, angle, sequence separation
    - Vector: normalized edge direction
    """
    edge_indices = []
    edge_scalar_feats = []
    edge_vector_feats = []

    node_map = {n: i for i, n in enumerate(g.nodes())}
    coords = {n: np.array(d['coords']) for n, d in g.nodes(data=True)}
    seq_indices = {n: idx for idx, (n, _) in enumerate(g.nodes(data=True))}

    LONG_RANGE_THRESHOLD = 12

    for u, v, a in g.edges(data=True):
        coord_u, coord_v = coords[u], coords[v]
        diff = coord_v - coord_u
        dist = np.linalg.norm(diff) + 1e-8
        direction = diff / dist

        idx_u, idx_v = seq_indices[u], seq_indices[v]
        seq_sep = abs(idx_v - idx_u)
        is_long_range = float(seq_sep > LONG_RANGE_THRESHOLD)

        kinds = a.get('kind', [])

        edge_indices.append([node_map[u], node_map[v]])
        edge_scalar_feats.append([
            float('knn' in kinds),
            float('peptide_bond' in kinds),
            float('disulfide' in kinds),
            float('hbond' in kinds),
            float('ionic' in kinds),
            float('cation_pi' in kinds),
            dist,
            calculate_angle(diff, coord_u),
            seq_sep,
            is_long_range
        ])
        edge_vector_feats.append(direction)

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_s = torch.tensor(edge_scalar_feats, dtype=torch.float)
    edge_v = torch.tensor(edge_vector_feats, dtype=torch.float).unsqueeze(1)

    return edge_index, edge_s, edge_v

# ============================
# Graph Conversion Function
# ============================

def convert_nx_to_pyg(g: nx.Graph) -> Data:
    """
    Converts a NetworkX protein graph to PyG Data object.
    Adds both traditional residue features and GVP-compatible geometric features.
    """

    # === Node features ===
    one_hots, meilers, coords = [], [], []
    betas, seqs, bfs = [], [], []
    physchem, degree, clustering = [], [], []
    betweenness, pagerank, contact_num = [], [], []
    relpos = []

    for _, attr in g.nodes(data=True):
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

    coords_tensor = torch.tensor(coords, dtype=torch.float)
    betas_tensor = torch.tensor(betas, dtype=torch.float)
    seqs_tensor = torch.tensor(seqs, dtype=torch.float)

    # === GVP Node features ===
    forward = F.pad(_normalize(coords_tensor[1:] - coords_tensor[:-1]), (0,0,0,1))
    backward = F.pad(_normalize(coords_tensor[:-1] - coords_tensor[1:]), (0,0,1,0))
    sidechain = betas_tensor

    node_v = torch.stack([forward, backward, sidechain], dim=1)  # (n_nodes, 3, 3)
    dihedrals = torch.zeros((len(coords), 3))  # Placeholder for dihedral angles
    node_s = torch.cat([torch.cos(dihedrals), torch.sin(dihedrals)], dim=-1)

    # === Edge features ===
    edge_index, raw_edge_s, edge_v = build_edge_features(g)

    diff = coords_tensor[edge_index[0]] - coords_tensor[edge_index[1]]
    dists = diff.norm(dim=-1)
    rbf = radial_basis(dists)

    seq_offset = edge_index[0] - edge_index[1]
    pos_emb = positional_encoding(seq_offset)

    edge_s = torch.cat([rbf, pos_emb], dim=-1)

    # === Scalar features (for pooling etc.) ===
    # scalar_feats = torch.cat([
    #     torch.tensor(one_hots, dtype=torch.float),
    #     torch.tensor(meilers, dtype=torch.float),
    #     torch.tensor(physchem, dtype=torch.float),
    #     torch.tensor(degree, dtype=torch.float).unsqueeze(1),
    #     torch.tensor(clustering, dtype=torch.float).unsqueeze(1),
    #     torch.tensor(betweenness, dtype=torch.float).unsqueeze(1),
    #     torch.tensor(pagerank, dtype=torch.float).unsqueeze(1),
    #     torch.tensor(contact_num, dtype=torch.float).unsqueeze(1),
    #     torch.tensor(relpos, dtype=torch.float).unsqueeze(1)
    # ], dim=-1)

    return Data(
        # GAT features
        one_hot_residues=torch.tensor(one_hots, dtype=torch.float),
        meiler_features=torch.tensor(meilers, dtype=torch.float),
        # x_coords=coords_tensor,
        pos=coords_tensor,
        beta_carbon_vector=betas_tensor,
        seq_neighbour_vector=seqs_tensor,
        b_factor=torch.tensor(bfs, dtype=torch.float),
        physicochemical_feat=torch.tensor(physchem, dtype=torch.float),
        degree=torch.tensor(degree, dtype=torch.float).unsqueeze(1),
        clustering=torch.tensor(clustering, dtype=torch.float).unsqueeze(1),
        betweenness=torch.tensor(betweenness, dtype=torch.float).unsqueeze(1),
        pagerank=torch.tensor(pagerank, dtype=torch.float).unsqueeze(1),
        contact_number=torch.tensor(contact_num, dtype=torch.float).unsqueeze(1),
        relative_position=torch.tensor(relpos, dtype=torch.float).unsqueeze(1),
        edge_attr=raw_edge_s,

        # GVP features
        x=coords_tensor,
        node_s=node_s,
        node_v=node_v,
        edge_s=edge_s,
        edge_v=edge_v,
        edge_index=edge_index,
        # scalar_features=scalar_feats
    )

# ============================
# Protein Graph Builder (Main Entry)
# ============================

def uniprot_id_to_structure(pdb_path: str, residue_embeddings: np.ndarray, cls_embedding: np.ndarray | None) -> Data | None:
    """
    Given a PDB file, construct a PyG Data object with biophysical and GVP features.
    Optionally add ESM embeddings.
    """
    config = ProteinGraphConfig(
        granularity="CA",
        edge_construction_functions=[
            add_peptide_bonds, add_hydrogen_bond_interactions,
            add_cation_pi_interactions, add_ionic_interactions,
            add_disulfide_interactions, partial(add_k_nn_edges, k=5, long_interaction_threshold=0),
            add_distance_to_edges,
        ],
        node_metadata_functions=[amino_acid_one_hot, meiler_embedding],
    )

    g = construct_graph(config=config, path=pdb_path)
    if g is None:
        return None

    # Add extra features
    add_beta_carbon_vector(g)
    add_sequence_neighbour_vector(g)
    add_graph_centrality_features(g)
    add_contact_number(g)
    add_relative_position(g)

    data = convert_nx_to_pyg(g)

    if residue_embeddings is not None:
        data.residue_embeddings = torch.tensor(residue_embeddings, dtype=torch.float)
    if cls_embedding is not None:
        data.cls_embedding = torch.tensor(cls_embedding, dtype=torch.float)

    return data
