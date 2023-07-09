from typing import Any
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, pool as pyg_pool
from torch_geometric.utils import to_dense_batch
from torch.nn.modules.batchnorm import _BatchNorm

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

class GraphGATConvBn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GATConv(in_channels, out_channels, heads=1, concat=False)
        self.norm = NodeLevelBatchNorm(out_channels)

    def forward(self, data):
        x, edge_index, edge_features = data.x, data.edge_index, data.edge_attr
        data.x = F.relu(self.norm(self.conv(x, edge_index, edge_features)))

        return data

class DrugEncoder(nn.Module):
    def __init__(self, input_features, out_dim, dropout) -> Any:
        super().__init__()
        self.conv1 = GraphGATConvBn(
            in_channels=input_features,
            out_channels= out_dim * 3
        )

        self.conv2 = GraphGATConvBn(
            in_channels= out_dim * 3,
            out_channels= out_dim
        )

        self.dropout = nn.Dropout(dropout)


    def forward(self, data):
        data = self.conv1(data)
        data.x = self.dropout(data.x)
        data = self.conv2(data)
        graph_featurees = pyg_pool.global_mean_pool(data.x, data.batch)

        return graph_featurees, data.x

class ProteinEncoderWithSelfAttention(nn.Module):
    def __init__(self, input_dim, dropout) -> Any:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=input_dim,
                      out_channels=32,
                      kernel_size=7, stride=1,
                      padding=7 // 2, dilation=1, groups=1,
                      bias=True, padding_mode='zeros'
            ),
            nn.LeakyReLU(negative_slope=.01),
            nn.BatchNorm1d(num_features=32,
                           eps=1e-05, momentum=0.1, affine=True),
            nn.Dropout(dropout)
        )

        self.attention_module = nn.MultiheadAttention(
            embed_dim=32,
            num_heads=1,
            dropout=0.3
        )

        self.weight = nn.Parameter(torch.ones(2))
        self.weight.requires_grad = True

    def forward(self, x):
        x = torch.permute(x , (0, 2, 1))
        x = self.encoder(x)
        x = torch.permute(x , (0, 2, 1))
        x_attn, _ = self.attention_module(x, x, x)
        weights = self.weight / torch.sum(self.weight)
        x = x * weights[0] + x_attn * weights[1] 
        out = torch.squeeze(x)
        return out

class DTAModel(nn.Module):
    def __init__(self, dropout, out_dim) -> Any:
        super().__init__()

        self.drug_encoder = DrugEncoder(
            input_features=22,
            out_dim = 32,
            dropout=dropout
        )

        self.protein_encoder_1 = ProteinEncoderWithSelfAttention(
            input_dim = 1024,
            dropout=dropout
        )

        self.protein_encoder_2 = ProteinEncoderWithSelfAttention(
            input_dim = 480,
            dropout=dropout
        )

        self.cross_attn_1 = nn.MultiheadAttention(
            embed_dim=32,
            num_heads=1,
            dropout=0.5,
            batch_first=True
        )

        self.cross_attn_2 = nn.MultiheadAttention(
            embed_dim=32,
            num_heads=1,
            dropout=0.5,
            batch_first=True
        )

        self.regressor = nn.Sequential(
            nn.Linear(32 * 5 , 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim)
        )

        self.pooling1 = nn.AdaptiveAvgPool1d(1)
        self.pooling2 = nn.AdaptiveAvgPool1d(1)
        self.pooling3 = nn.AdaptiveAvgPool1d(1)
        self.pooling4 = nn.AdaptiveAvgPool1d(1)

    def forward(self, data):
        prot5_embeddings = data.prot5_embedding
        prot5_embeddings = prot5_embeddings.type(torch.float32)
        esm2_embeddings = data.esm2_embedding
        graph_features, node_features = self.drug_encoder(data)
        prot5_feature = self.protein_encoder_1(prot5_embeddings)
        esm2_feature = self.protein_encoder_2(esm2_embeddings)
        node_level_features, node_mask = to_dense_batch(node_features, data.batch)

        cross_attn_features_prot5, _ = self.cross_attn_1(
            node_level_features, 
            prot5_feature, 
            prot5_feature
        )

        cross_attn_features_esm2, _ = self.cross_attn_1(
            node_level_features, 
            esm2_feature, 
            esm2_feature
        )

        seq_mask = torch.unsqueeze(data.mask, dim=2)
        node_mask = torch.unsqueeze(node_mask, dim=2)

        prot5_feature = torch.permute((prot5_feature * seq_mask), dims=(0, 2, 1))
        esm2_feature = torch.permute((esm2_feature * seq_mask), dims=(0, 2, 1))
        cross_attn_features_prot5 = torch.permute((cross_attn_features_prot5 * node_mask), dims=(0, 2, 1))
        cross_attn_features_esm2 = torch.permute((cross_attn_features_esm2 * node_mask), dims=(0, 2, 1))

        prot5_feature = self.pooling1(prot5_feature).squeeze()
        esm2_feature = self.pooling2(esm2_feature).squeeze()
        cross_attn_features_prot5 = self.pooling3(cross_attn_features_prot5).squeeze()
        cross_attn_features_esm2 = self.pooling3(cross_attn_features_esm2).squeeze()

        features = torch.cat([
            graph_features,
            prot5_feature,
            esm2_feature,
            cross_attn_features_esm2,
            cross_attn_features_prot5
        ], dim=1)

        out = self.regressor(features)

        return out
