import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, global_mean_pool
from torch_geometric.nn import GATConv, BatchNorm, GINConv, GATv2Conv, HeteroBatchNorm
from pe import apply_positional_encoding_to_batch, PositionalEncoding
from torch.nn.modules.batchnorm import _BatchNorm
from mask_test import create_attention_mask

class NodeLevelBatchNorm(_BatchNorm):
    r"""
    Applies Batch Normalization over a batch of graph data.
    Shape:
        - Input: [batch_nodes_dim, node_feature_dim]
        - Output: [batch_nodes_dim, node_feature_dim]
    batch_nodes_dim: all nodes of a batch graph
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(NodeLevelBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError('expected 2D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        return torch.functional.F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

    def extra_repr(self):
        return 'num_features={num_features}, eps={eps}, ' \
               'affine={affine}'.format(**self.__dict__)


class egretblock_hetero(nn.Module):
    def __init__(self, in_dim) -> None:
        super().__init__()

        # GIN Conv requires a message function and an update function
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim)
        )

        # Initialize GINConv with the message and update functions
        self.gin_layer = GINConv(nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim)
        ), train_eps=True)

        self.attn_drop = nn.Dropout(0.2)

        self.encoder = nn.Sequential(
            NodeLevelBatchNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )

    def forward(self, x, edge_index):
        # Normalizing the input features
        # print(x[0].shape)
        # print(x[1].shape)
        
        # Compute features using GINConv
        gin_feat = self.gin_layer(x, edge_index)
        # print(gin_feat.shape)
        
        # Residual connection with attention dropout
        z = x[1] + self.attn_drop(gin_feat)
        
        # Pass through the encoder
        z1 = self.encoder(z)
        
        # Final residual connection
        z = z + z1
        
        return z


class egretblock(nn.Module):
    def __init__(self, in_dim, edge_dim, num_heads=1) -> None:
        super().__init__()
                
        out_dim = in_dim // num_heads
        self.egret_layer = GATv2Conv(in_channels=in_dim, 
                                        out_channels=out_dim,
                                        heads=num_heads,
                                        concat=True, 
                                        negative_slope=0.2, 
                                        dropout=0.2, add_self_loops = True, 
                                        edge_dim=edge_dim, fill_value = 'mean', bias = True)
        
        self.bn1 = NodeLevelBatchNorm(in_dim)
        # self.bn1 = HeteroBatchNorm(in_dim, num_types=3)
        self.attn_drop = nn.Dropout(0.2)
        
        self.encoder = nn.Sequential(
            NodeLevelBatchNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )
                
    def forward(self, x, edge_index, edge_attr):
        
        # print(x)
        attn_feat = self.egret_layer(self.bn1(x), edge_index, edge_attr)
        z = x + self.attn_drop(attn_feat)
        z1 = self.encoder(z)
        z = z + z1
        
        return z

class HeteroGraphEncoder(nn.Module):
    def __init__(self, protein_in_dim, drug_in_dim, super_feat_dim, hidden_dim, 
                 protein_edge_dim, drug_edge_dim, num_heads=1, num_layers=4):
        super().__init__()

        # Input projections to align feature dimensions to hidden_dim
        self.protein_proj = nn.Linear(protein_in_dim, hidden_dim)
        self.drug_proj = nn.Linear(drug_in_dim, hidden_dim)
        self.super_proj = nn.Linear(super_feat_dim, hidden_dim)

        self.layers = nn.ModuleList()
        # for _ in range(num_layers):
        #     conv = HeteroConv({
        #         ('protein', 'protein-protein', 'protein'): GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads),
        #         ('drug', 'drug-drug', 'drug'): GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads),
        #         ('protein', 'protein-supernode', 'supernode'): GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads),
        #         ('drug', 'drug-supernode', 'supernode'): GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads)
        #     }, aggr='mean')
        #     self.layers.append(conv)
            
        for _ in range(num_layers):
            conv = HeteroConv({
                ('protein', 'protein-protein', 'protein'): egretblock(hidden_dim, protein_edge_dim, num_heads=num_heads),
                ('drug', 'drug-drug', 'drug'): egretblock(hidden_dim, drug_edge_dim, num_heads=num_heads),
                ('protein', 'protein-supernode', 'supernode'): egretblock_hetero(hidden_dim),
                ('drug', 'drug-supernode', 'supernode'): egretblock_hetero(hidden_dim),
                ('supernode', 'supernode-protein', 'protein'): egretblock_hetero(hidden_dim),
                ('supernode', 'supernode-drug', 'drug'): egretblock_hetero(hidden_dim)
            }, aggr='mean')
            self.layers.append(conv)

        self.dropout = nn.Dropout(0.2)
        self.batch_norm = nn.ModuleDict({
            'protein': BatchNorm(hidden_dim),
            'drug': BatchNorm(hidden_dim),
            'supernode': BatchNorm(hidden_dim)
        })
        
        # self.batch_norm = HeteroBatchNorm(
        #     in_channels=hidden_dim,
        #     num_types=3
        # )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # Project protein and drug node features to the hidden dimension
        x_dict['protein'] = self.protein_proj(x_dict['protein'])
        x_dict['drug'] = self.drug_proj(x_dict['drug'])
        x_dict['supernode'] = self.super_proj(x_dict['supernode'])

        for conv in self.layers:
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            for node_type in x_dict:
                x_dict[node_type] = self.batch_norm[node_type](x_dict[node_type])
                x_dict[node_type] = self.dropout(x_dict[node_type])
        return x_dict


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        z = self.fc1(x)
        z = self.act(z)
        z = self.drop(z)
        z = self.fc2(z)
        z = self.drop(z)
        return z
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model * 2)
        
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, x1, x2, mask=None):
        N = x2.shape[0]
        
        q = self.linear1(x1)
        z = self.linear2(x2)
        z = torch.reshape(z, (2, N, self.d_model))
        k, v = z[0], z[1]
                
        # Reshape into (batch_size, num_heads, seq_len, d_k)
        query = q.view(-1, self.num_heads, self.d_k).transpose(0, 1)
        key = k.view(-1, self.num_heads, self.d_k).transpose(0, 1)
        value = v.view(-1, self.num_heads, self.d_k).transpose(0, 1)
                
        # Scaled dot-product attention
        scale = torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        scores = torch.matmul(query, key.transpose(2, 1)) / scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = self.softmax(scores)
        attention_weights = self.attn_drop(attention_weights)
        attention_output = torch.matmul(attention_weights, value)
        
        # Concatenate heads
        attention_output = attention_output.transpose(0, 1).contiguous().view(-1, self.d_model)
        
        # Final linear layer
        output = self.proj_drop(self.out_linear(attention_output))
        
        return output, attention_weights

class MGraphDTA_Hetero(nn.Module):
    def __init__(self, protein_in_dim, drug_in_dim, super_feat_dim, hidden_dim, 
                 protein_edge_dim, drug_edge_dim, out_dim=1, num_heads=4, num_layers=4):
        super().__init__()

        self.encoder = HeteroGraphEncoder(
            protein_in_dim=protein_in_dim,
            drug_in_dim=drug_in_dim,
            super_feat_dim=super_feat_dim,
            hidden_dim=hidden_dim,
            protein_edge_dim=protein_edge_dim,
            drug_edge_dim=drug_edge_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )

        self.prot_pe = PositionalEncoding(d_model=hidden_dim, dropout=0.2, max_len=2800)
        self.mol_pe = PositionalEncoding(d_model=hidden_dim, dropout=0.2, max_len=30)

        self.cross_attn1 = MultiHeadAttention(d_model=hidden_dim, num_heads=num_heads, dropout=0.2)
        self.cross_attn2 = MultiHeadAttention(d_model=hidden_dim, num_heads=num_heads, dropout=0.2)

        self.classifier = Mlp(
            in_features=hidden_dim * 3,  # Combine protein, drug, and supernode embeddings
            hidden_features=8,
            out_features=out_dim,
            act_layer=nn.LeakyReLU,
            drop=0.2
        )

    def forward(self, data):       
        
        # print(data)
        # print(data.x_dict)
        # print(data.batch_dict)
        # print(data.edge_attr_dict)
        # print(data.edge_index_dict) 
        # Assume `data` has `x_dict`, `edge_index_dict`, `edge_attr_dict`, and `batch_dict` for heterogeneous graph
        x_dict, edge_index_dict, edge_attr_dict, batch_dict = (
            data.x_dict, data.edge_index_dict, data.edge_attr_dict, data.batch_dict
        )

        # Pass through heterogeneous GNN layers
        x_dict = self.encoder(x_dict, edge_index_dict, edge_attr_dict)
        
        # Encode features with positional encoding where needed
        x_dict['protein'] = apply_positional_encoding_to_batch(x_dict['protein'], batch_dict['protein'], self.prot_pe)
        x_dict['drug'] = apply_positional_encoding_to_batch(x_dict['drug'], batch_dict['drug'], self.mol_pe)

        attn_mask1 = create_attention_mask(batch_dict['drug'], batch_dict['protein'])
        attn_mask2 = create_attention_mask(batch_dict['protein'], batch_dict['drug'])
           
        attn_feat1, _ = self.cross_attn1(x_dict['protein'], x_dict['drug'], mask=attn_mask1)       
        attn_feat2, _ = self.cross_attn2(x_dict['drug'], x_dict['protein'], mask=attn_mask2)

        # Pool features separately for each node type
        prot_feat = global_mean_pool(x_dict['protein'], batch_dict['protein'])
        drug_feat = global_mean_pool(x_dict['drug'], batch_dict['drug'])
        supernode_feat = global_mean_pool(x_dict['supernode'], batch_dict['supernode'])
        attn_feat1 = global_mean_pool(attn_feat1, batch_dict['protein'])
        attn_feat2 = global_mean_pool(attn_feat2, batch_dict['drug'])

        # Concatenate attention features and original pooled features
        combined_feat = torch.cat([supernode_feat, attn_feat1, attn_feat2], dim=-1)
        out = self.classifier(combined_feat)

        return out