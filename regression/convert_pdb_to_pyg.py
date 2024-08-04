import networkx as nx
import torch
from torch_geometric.data import Data
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.edges.distance import add_hydrogen_bond_interactions, \
    add_peptide_bonds, \
    add_distance_to_edges, \
    add_cation_pi_interactions, \
    add_k_nn_edges
    
from graphein.protein.features.sequence.embeddings import esm_sequence_embedding, \
    biovec_sequence_embedding
    
from graphein.protein.features.nodes.amino_acid import amino_acid_one_hot, meiler_embedding
from graphein.protein.features.nodes.dssp import add_dssp_feature
from graphein.protein import esm_residue_embedding
from graphein.protein.features.nodes.geometry import add_beta_carbon_vector, add_sequence_neighbour_vector

from functools import partial
from graphein.protein.graphs import construct_graphs_mp, construct_graph

# Define the full list of standard amino acid residues
standard_residues = [
    'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
    'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR'
]

# Define a function to one-hot encode residue types
def one_hot_encode_residue(residue_name):
    encoder = OneHotEncoder(categories=[standard_residues], sparse_output=False)
    one_hot = encoder.fit_transform([[residue_name]])
    return one_hot.flatten()

# Function to calculate the angle between two vectors
def calculate_angle(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    cos_theta = dot_product / (norm_vec1 * norm_vec2)
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Ensure the value is within the valid range for arccos
    return angle

def convert_nx_to_pyg(nx_graph):
    # Extract node attributes and create node features
    one_hot_residues = []
    meiler_features = []
    esm_embeddings = []
    coords = []
    beta_carbon_vectors = []
    seq_neighbour_vector = []
    
    node_indices = {}
    for i, (node, attr) in enumerate(nx_graph.nodes(data=True)):
        one_hot_residue = one_hot_encode_residue(attr['residue_name'])
        one_hot_residues.append(one_hot_residue)
        meiler_features.append(attr['meiler'])
        esm_embeddings.append(attr['esm_embedding'])
        coords.append(attr['coords'])
        beta_carbon_vectors.append(attr['c_beta_vector'])
        seq_neighbour_vector.append(attr['sequence_neighbour_vector_n_to_c'])
        node_indices[node] = i

    one_hot_residues = torch.tensor(one_hot_residues, dtype=torch.float)
    meiler_features = torch.tensor(meiler_features, dtype=torch.float)
    esm_embeddings = torch.tensor(esm_embeddings, dtype=torch.float)
    coords = torch.tensor(coords, dtype=torch.float)
    beta_carbon_vectors = torch.tensor(beta_carbon_vectors, dtype=torch.float)
    seq_neighbour_vector = torch.tensor(seq_neighbour_vector, dtype=torch.float)
    
    # Extract edge attributes and create edge indices
    edge_indices = []
    edge_attrs = []
    for u, v, attr in nx_graph.edges(data=True):
        knn_bond = 1.0 if 'knn' in attr['kind'] else 0.0
        peptide_bond = 1.0 if 'peptide_bond' in attr['kind'] else 0.0
        disulfide_bond = 1.0 if 'disulfide_bond' in attr['kind'] else 0.0
        hydrogen_bond = 1.0 if 'hydrogen_bond' in attr['kind'] else 0.0
        
        # Calculate the edge vector and angle relative to the origin
        coord_u = nx_graph.nodes[u]['coords']
        coord_v = nx_graph.nodes[v]['coords']
        edge_vector = np.array(coord_v) - np.array(coord_u)
        origin_vector = np.array(coord_u)  # Using one end of the edge for angle calculation
        angle = calculate_angle(edge_vector, origin_vector)
        
        edge_indices.append([node_indices[u], node_indices[v]])
        edge_attrs.append([knn_bond, peptide_bond, disulfide_bond, hydrogen_bond, attr['distance'], angle])
    
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    # Create PyG Data object
    data = Data(one_hot_residues=one_hot_residues, meiler_features=meiler_features,
                esm_embeddings=esm_embeddings, edge_index=edge_index, edge_attr=edge_attr, pos=coords, 
                beta_carbon_vectors=beta_carbon_vectors, seq_neighbour_vector=seq_neighbour_vector)
    
    return data


def uniprot_id_to_structure(file_path):
    params_to_change = {
        "granularity": "CA",
        "edge_construction_functions": [
            add_peptide_bonds, 
            add_hydrogen_bond_interactions,
            add_cation_pi_interactions,
            partial(add_k_nn_edges, k=8),
            add_distance_to_edges,
        ],
        "graph_metadata_functions": [
            partial(esm_residue_embedding, model_name="esm2_t33_650M_UR50D"),
            # biovec_sequence_embedding
        ],
        "node_metadata_functions": [
            amino_acid_one_hot,
            meiler_embedding,
        ]
    }

    config = ProteinGraphConfig(
        **params_to_change
    )
    config.dict()

    g = construct_graph(config=config, path=file_path)
    # g = construct_graph(config=config, path=f"./data/alphafold_structures/kiba/AF-{id}-F1-model_v4.pdb")
    add_beta_carbon_vector(g)
    add_sequence_neighbour_vector(g)
    pg = convert_nx_to_pyg(g)
    
    return pg