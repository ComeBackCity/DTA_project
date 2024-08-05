import torch
import torch.nn as nn
import math
from torch_geometric.data import Batch

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)   
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

def apply_positional_encoding_to_batch(x, batch, pos_enc):
    """
    Apply positional encoding to each graph in the batch.
    
    Args:
        batch (Batch): A batch of graphs.
        pos_enc (PositionalEncoding): The positional encoding instance.
        
    Returns:
        Batch: A batch of graphs with positional encoding applied.
    """
    node_features = x
    dim = node_features.size(1)
    batch_size = batch.max().item() + 1

    encoded_features = torch.zeros_like(node_features)

    for i in range(batch_size):
        # Get the node indices for the current graph
        graph_node_indices = (batch == i).nonzero(as_tuple=True)[0]
        num_graph_nodes = graph_node_indices.size(0)

        # Slice the node features for the current graph
        graph_features = node_features[graph_node_indices].unsqueeze(1).transpose(0, 1)
        
        # Apply positional encoding
        encoded_graph_features = pos_enc(graph_features).transpose(0, 1).squeeze(1)
        
        # Store the encoded features
        encoded_features[graph_node_indices] = encoded_graph_features

    # Return the updated batch with positional encoded features
    return encoded_features

# Example usage
# dim = 16  # Dimension of the node features
# max_len = 1000  # Maximum sequence length
# pos_enc = PositionalEncoding(d_model=dim, max_len=max_len)

# # Create a batch of graphs (for demonstration purposes)
# node_features = torch.randn(20, dim)
# edge_index = torch.randint(0, 20, (2, 40))
# edge_attr = torch.randn(40, dim)
# batch = Batch(x=node_features, edge_index=edge_index, edge_attr=edge_attr, batch=torch.randint(0, 5, (20,)))

# # Apply positional encoding to the batch
# encoded_batch = apply_positional_encoding_to_batch(batch, pos_enc)

# print("Original Node Features:")
# print(node_features)
# print("Positional Encoded Node Features:")
# print(encoded_batch)

