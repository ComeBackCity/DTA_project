import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch.nn.modules.batchnorm import _BatchNorm
import torch_geometric.nn as gnn
from torch import Tensor
from collections import OrderedDict
from torch_geometric.utils import to_dense_batch


'''
MGraphDTA: Deep Multiscale Graph Neural Network for Explainable Drug-target binding affinity Prediction
'''


class Conv1dReLU(nn.Module):
    '''
    kernel_size=3, stride=1, padding=1
    kernel_size=5, stride=1, padding=2
    kernel_size=7, stride=1, padding=3
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.inc(x)

class LayerNormalizer(nn.Module):
    def __init__(self, feature_dim) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
        x = self.layer_norm(x)
        x = torch.permute(x, (0, 2, 1))
        return x

class LinearReLU(nn.Module):
    def __init__(self,in_features, out_features, bias=True):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features, bias=bias),
            nn.ReLU()
        )

    def forward(self, x):
        
        return self.inc(x)

class StackCNNwithSelfAttention(nn.Module):
    def __init__(self, layer_num, seq_length, in_channels, out_channels, kernel_size, stride=1, padding=0, dropout=0.5):
        super().__init__()

        self.inc = nn.Sequential(OrderedDict([('conv_layer0', Conv1dReLU(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))]))
        for layer_idx in range(layer_num - 1):
            self.inc.add_module(f'non_linearity{layer_idx}', nn.LeakyReLU())
            # self.inc.add_module(f'batch_norm{layer_idx}', nn.BatchNorm1d(num_features=96))
            self.inc.add_module(f'layer_norm{layer_idx}', LayerNormalizer(96))
            self.inc.add_module(f'dropout{layer_idx}', nn.Dropout(dropout))
            self.inc.add_module('conv_layer%d' % (layer_idx + 1), Conv1dReLU(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
            
        # self.inc2 = nn.Sequential()
        # self.inc.add_module(f'sa_block', nn.MultiheadAttention(embed_dim=out_channels, num_heads=1, dropout=0.2))
        # self.inc.add_module('pool_layer', nn.AdaptiveMaxPool1d(1))
        # self.inc.add_module('non_linearity', nn.LeakyReLU())
        # self.inc.add_module('batch_norm', nn.BatchNorm1d(num_features=96))
        # self.inc.add_module('dropout', nn.Dropout(0.2))

        # self.pooling_layer = nn.AdaptiveMaxPool1d(1)
        self.weight = nn.Parameter(torch.ones(2))
        self.weight.requires_grad = True
        self.attn_layer = nn.MultiheadAttention(embed_dim=out_channels, num_heads=1, dropout=dropout, batch_first=True)

        self.linears = nn.Sequential(
            nn.Linear(seq_length * 96, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 96),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):

        x = torch.permute(x, (0, 2, 1))
        x = self.inc(x)
        x = torch.permute(x, (0, 2, 1))
        if mask is not None:
            x_attn, _ = self.attn_layer(x, x, x, mask)
        else:
            x_attn, _ = self.attn_layer(x, x, x)
        weights = self.weight / torch.sum(self.weight)
        x = x * weights[0] + x_attn * weights[1]
        x = torch.permute(x, (0, 2, 1))
        seq_features = x
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.linears(x)
        x = torch.reshape(x , (x.shape[0], -1, 1))
        # x = self.pooling_layer(x)
        # x = torch.permute(x, (0, 2, 1))
        # attn_score, _ = self.attn_layer(query=x, key=x, value=x)
        # weights = self.weight / torch.sum(self.weight)
        # x = weights[0] * attn_score + weights[1] * x
        # x = torch.permute(x, (0, 2, 1))
        # x = self.pooling_layer(x).squeeze(-1)
        x = x.squeeze(-1)

        return x, seq_features

class TargetRepresentation(nn.Module):
    def __init__(self, block_num, vocab_size, embedding_num):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_num, padding_idx=0)
        self.block_list = nn.ModuleList()
        for block_idx in range(block_num):
            self.block_list.append(
                StackCNNwithSelfAttention(block_idx+1, embedding_num, 96, 3)
            )

        self.linear = nn.Linear(block_num * 96, 96)
        
    def forward(self, x):
        x = self.embed(x).permute(0, 2, 1)
        feats, seq_feats = [], []
        for block in self.block_list:
            features, seq_features = block(x)
            feats.append(features)
            seq_feats.append(seq_features)
        # feats = [block(x) for block in self.block_list]
        x = torch.cat(feats, -1)
        x = self.linear(x)

        seq_features = torch.mean(seq_feats)

        return x, seq_features


class ProteinRepresentation(nn.Module):
    def __init__(self, block_num, embedding_num, input_features, seq_length, dropout):
        super().__init__()
        self.reshaper = nn.Linear(in_features=input_features, out_features=embedding_num)
        self.block_list = nn.ModuleList()
        for block_idx in range(block_num):
            self.block_list.append(
                StackCNNwithSelfAttention(
                    block_idx+1, 
                    seq_length, 
                    embedding_num, 
                    96, 
                    3, 
                    padding = 3//2, 
                    dropout = dropout
                )
            )

        self.weights = nn.Parameter(torch.ones(block_num))
        self.weights.requires_grad = True
        self.linear = nn.Linear(block_num * 96, 96)
        
    def forward(self, x, mask=None):
        # if mask is not None:
        mask = mask.to(torch.float32)
        x = F.relu(self.reshaper(x.type(torch.float32)))
        feats, seq_features = [], []
        for block in self.block_list:
            feature, seq_feature = block(x, mask)
            feats.append(feature)
            seq_features.append(seq_feature)

        seq_feat = torch.zeros(seq_features[-1].shape).to("cuda")
        weights = self.weights / torch.sum(self.weights)
        for weight, seq_feature in zip(weights, seq_features):
            seq_feat += weight * seq_feature
        
        x = torch.cat(feats, -1)
        x = self.linear(x)

        return x, seq_feat

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
        self.conv = gnn.GATConv(in_channels, out_channels, heads=1, concat=False)
        self.norm = NodeLevelBatchNorm(out_channels)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        data.x = F.relu(self.norm(self.conv(x, edge_index, edge_attr)))

        return data
    
# class GraphGATConvBn2(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv = gnn.GATConv(in_channels, out_channels, heads=4, concat=False)
#         self.lin = gnn.Linear(
#             in_channels = out_channels * 4,
#             out_channels = out_channels
#         )
#         self.norm = NodeLevelBatchNorm(out_channels)

#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#         data.x = F.relu(self.norm(self.conv(x, edge_index)))
#         data.x = F.relu(self.lin(data.x))

#         return data

class DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate=32, bn_size=4):
        super().__init__()
        self.conv1 = GraphGATConvBn(num_input_features, int(growth_rate * bn_size))
        self.conv2 = GraphGATConvBn(int(growth_rate * bn_size), growth_rate)

    def bn_function(self, data):
        concated_features = torch.cat(data.x, 1)
        data.x = concated_features

        data = self.conv1(data)

        return data
    
    def forward(self, data):
        if isinstance(data.x, Tensor):
            data.x = [data.x]

        data = self.bn_function(data)
        data = self.conv2(data)

        return data

class DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, growth_rate=32, bn_size=4):
        super().__init__()
        for i in range(num_layers):
            layer = DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size)
            self.add_module('layer%d' % (i + 1), layer)


    def forward(self, data):
        features = [data.x]
        for name, layer in self.items():
            data = layer(data)
            features.append(data.x)
            data.x = features

        data.x = torch.cat(data.x, 1)

        return data


class GraphDenseNet(nn.Module):
    def __init__(self, num_input_features, out_dim, growth_rate=32, block_config = (3, 3, 3, 3), bn_sizes=[2, 3, 4, 4]):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([('conv0', GraphGATConvBn(num_input_features, 32))]))
        num_input_features = 32

        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers, num_input_features, growth_rate=growth_rate, bn_size=bn_sizes[i]
            )
            self.features.add_module('block%d' % (i+1), block)
            num_input_features += int(num_layers * growth_rate)

            trans = GraphGATConvBn(num_input_features, num_input_features // 2)
            self.features.add_module("transition%d" % (i+1), trans)
            num_input_features = num_input_features // 2

        self.classifer = nn.Linear(num_input_features, out_dim)

    def forward(self, data):
        data = self.features(data)
        node_features = self.classifer(data.x)
        x = gnn.global_mean_pool(node_features, data.batch)

        return node_features, x
    
class ProteinEncoder(nn.Module):
    def __init__(self, filter_num=32) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1024,
                      out_channels=32,
                      kernel_size=7, stride=1,
                      padding=7 // 2, dilation=1, groups=1,
                      bias=True, padding_mode='zeros'
            ),
            nn.LeakyReLU(negative_slope=.01),
            nn.BatchNorm1d(num_features=32,
                           eps=1e-05, momentum=0.1, affine=True),
            nn.Dropout(.5)
        )

    def forward(self, x):
        x = torch.unsqueeze(x.type(torch.float32), dim=2)
        x = self.encoder(x)
        out = torch.squeeze(x)
        return out


class MGraphDTA(nn.Module):
    def __init__(self, block_num, vocab_protein_size, embedding_size=128, filter_num=32, out_dim=1, dropout = 0.2):
        super().__init__()
        # self.protein_encoder_2 = TargetRepresentation(block_num, vocab_protein_size, embedding_size)
        
        self.ligand_encoder = GraphDenseNet(num_input_features=22, out_dim=filter_num*3, block_config=[8, 8, 8], bn_sizes=[2, 2, 2])
        self.protein_encoder = ProteinRepresentation(block_num, embedding_size, input_features=1024, seq_length=1200, dropout=dropout)
        # self.protein_encoder_2 = ProteinRepresentation(block_num, embedding_size, input_features=480, seq_length=1200)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim = filter_num * 3,
            num_heads=1,
            dropout=dropout,
            batch_first=True
        )

        self.pooling = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(filter_num * 3 * 3, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, out_dim)
        )

    def forward(self, data):
        # print('In forward')
        protein_features, seq_features = self.protein_encoder(data.prot5_embedding, mask=data.mask)
        node_features, ligand_features = self.ligand_encoder(data)
        seq_features = torch.permute(seq_features, (0, 2, 1))
        node_features, node_mask = to_dense_batch(node_features, data.batch)
        mask = data.mask.to(torch.float32)
        cross_attn_feature, _ = self.cross_attention(
            node_features,
            seq_features,
            seq_features,
            mask
        )

        node_mask = torch.unsqueeze(node_mask, dim=2)
        cross_attn_feature = torch.permute(cross_attn_feature * node_mask, (0, 2, 1))
        cross_attn_feature = self.pooling(cross_attn_feature)
        cross_attn_feature = torch.squeeze(cross_attn_feature)
        # esm_features = self.protein_encoder_2(data.esm2_embedding, mask = data.mask)

        # cross_attn_features, _ = self.cross_attention(ligand_features, protein_features, protein_features)
        # cross_attn_features, _ = self.cross_attention(protein_features, ligand_features, ligand_features)

        # x = torch.cat([protein_features, esm_features, ligand_features], dim=-1)
        x = torch.cat([protein_features, cross_attn_feature, ligand_features], dim=-1)
        x = self.classifier(x)

        return x

def contains_nan(x):
    has_nan = torch.isnan(x)
    if has_nan.any():
        print("Tensor contains NaN values")
    else:
        print("Tensor does not contain NaN values")

def contains_inf(x):
    has_nan = torch.isinf(x)
    if has_nan.any():
        print("Tensor contains Inf values")
    else:
        print("Tensor does not contain Inf values")
