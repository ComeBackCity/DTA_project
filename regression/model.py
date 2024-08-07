import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch.nn.modules.batchnorm import _BatchNorm
import torch_geometric.nn as gnn
from torch import Tensor
from collections import OrderedDict
from egret import EGRETLayer, MultiHeadEGRETLayer, config_dict
from mask_test import create_attention_mask
from pe import apply_positional_encoding_to_batch, PositionalEncoding

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

class LinearReLU(nn.Module):
    def __init__(self,in_features, out_features, bias=True):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features, bias=bias),
            nn.ReLU()
        )

    def forward(self, x):
        
        return self.inc(x)

class StackCNN(nn.Module):
    def __init__(self, layer_num, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.inc = nn.Sequential(OrderedDict([('conv_layer0', Conv1dReLU(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))]))
        for layer_idx in range(layer_num - 1):
            self.inc.add_module('conv_layer%d' % (layer_idx + 1), Conv1dReLU(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))

        self.inc.add_module('pool_layer', nn.AdaptiveMaxPool1d(1))

    def forward(self, x):
        return self.inc(x).squeeze(-1)

class TargetRepresentation(nn.Module):
    def __init__(self, block_num, vocab_size, embedding_num):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_num, padding_idx=0)
        self.block_list = nn.ModuleList()
        for block_idx in range(block_num):
            self.block_list.append(
                StackCNN(block_idx+1, embedding_num, 96, 3)
            )

        self.linear = nn.Linear(block_num * 96, 96)
        
    def forward(self, x):
        x = self.embed(x).permute(0, 2, 1)
        feats = [block(x) for block in self.block_list]
        x = torch.cat(feats, -1)
        x = self.linear(x)

        return x

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

class GraphConvBn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = gnn.GraphConv(in_channels, out_channels)
        self.norm = NodeLevelBatchNorm(out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        data.x = F.relu(self.norm(self.conv(x, edge_index)))

        return data

class DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate=32, bn_size=4):
        super().__init__()
        self.conv1 = GraphConvBn(num_input_features, int(growth_rate * bn_size))
        self.conv2 = GraphConvBn(int(growth_rate * bn_size), growth_rate)

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
        self.features = nn.Sequential(OrderedDict([('conv0', GraphConvBn(num_input_features, out_dim))]))

        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers, num_input_features, growth_rate=growth_rate, bn_size=bn_sizes[i]
            )
            self.features.add_module('block%d' % (i+1), block)
            num_input_features += int(num_layers * growth_rate)

            trans = GraphConvBn(num_input_features, num_input_features // 2)
            self.features.add_module("transition%d" % (i+1), trans)
            num_input_features = num_input_features // 2

        # self.classifer = nn.Linear(num_input_features, out_dim)
        self.sa = nn.MultiheadAttention(
            embed_dim=out_dim, 
            num_heads=4, 
            dropout=0.2
        )
        

    def forward(self, data):
        data = self.features(data)
        attn_feat, _ = self.sa(data.x, data.x, data.x)
        x = data.x + attn_feat

        return x

class egretblock(nn.Module):
    def __init__(self, in_dim, edge_dim, num_heads=1) -> None:
        super().__init__()
        
        # config_dict['feat_drop'] = 0.5
        # config_dict['edge_feat_drop'] = 0.5
        # config_dict['attn_drop'] = 0.5
        
        # self.egret_layer = MultiHeadEGRETLayer(in_dim, in_dim, edge_dim, num_heads, use_bias=True,
        #                                        merge='mean', config_dict=config_dict)
        
        self.egret_layer = gnn.GATv2Conv(in_channels=in_dim, 
                                        out_channels=in_dim, 
                                        heads=num_heads,
                                        concat=False, 
                                        negative_slope=0.2, 
                                        dropout=0.2, add_self_loops = True, 
                                        edge_dim=edge_dim, fill_value = 0, bias = True)
        
        self.bn1 = NodeLevelBatchNorm(in_dim)
        self.weight1 = nn.Parameter(torch.randn(2))
        
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=in_dim, stride=1, padding=1, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.5),
        )
        
        self.bn2 = NodeLevelBatchNorm(in_dim)
        self.weight2 = nn.Parameter(torch.randn(2))
        
    def forward(self, x, edge_index, edge_attr):
        
        # print(x)
        attn_feat = self.egret_layer(x, edge_index, edge_attr)
        # print(attn_feat)
        w1 = self.weight1.softmax(-1)
        z = w1[0] * x + w1[1] * attn_feat
        # print(z)
        z = self.bn1(z)
        z1 = torch.unsqueeze(z, 0)
        z1 = torch.permute(z1, (0, 2, 1))
        z1 = self.encoder(z1)
        z1 = torch.permute(z1, (0, 2, 1))
        z1 = torch.squeeze(z1)
        w2 = self.weight2.softmax(-1)
        z = w2[0] * z + w2[1] * z1
        # print(z)
        z = self.bn2(z)
        # print(z)
        
        return z
         
    
class GraphEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim) -> None:
        super().__init__()
        
        self.reshaper = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.5)
        )
        
        self.em1 = egretblock(in_dim=1024, edge_dim=edge_dim, num_heads=4)
        self.em2 = egretblock(in_dim=1024, edge_dim=edge_dim, num_heads=4)
        self.em3 = egretblock(in_dim=1024, edge_dim=edge_dim, num_heads=4)
        
        # self.weight1 = nn.Parameter(torch.randn(2))
        # self.weight2 = nn.Parameter(torch.randn(2))
        # self.weight3 = nn.Parameter(torch.randn(2))
        
        self.reducer = nn.Sequential(
            nn.Linear(1024, out_dim),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.5)
        )
        
    def forward(self, x, edge_index, edge_attr): 
        
        # print(x)
        z = self.reshaper(x)
        z = self.em1(z, edge_index, edge_attr)
        z = self.em2(z, edge_index, edge_attr)
        z = self.em2(z, edge_index, edge_attr)
        z = self.reducer(z)
        # print(z)
        return z
    
class LigandEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim) -> None:
        super().__init__()
        
        self.reshaper = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2)
        )
        
        self.em1 = egretblock(in_dim=1024, out_dim=512, edge_dim=edge_dim, num_heads=4)
        self.em2 = egretblock(in_dim=512, out_dim=out_dim, edge_dim=edge_dim, num_heads=4)
        
        
    def forward(self, x, edge_index, edge_attr): 
        
        z = self.reshaper(x)
        z = self.em1(z, edge_index, edge_attr)
        z = self.em2(z, edge_index, edge_attr)
        
        return z

class MGraphDTA(nn.Module):
    def __init__(self, protein_feat_dim, drug_feat_dim, protein_edge_dim, drug_edge_dim, filter_num=32, out_dim=1):
        super().__init__()
        # self.protein_encoder = GraphDenseNet(num_input_features=protein_feat_dim, out_dim=filter_num*3, block_config=[8, 8, 8], bn_sizes=[2, 2, 2])
        # self.ligand_encoder = GraphDenseNet(num_input_features=drug_feat_dim, out_dim=filter_num*3, block_config=[8, 8, 8], bn_sizes=[2, 2, 2])
        
        self.protein_encoder = GraphEncoder(protein_feat_dim, filter_num, protein_edge_dim)
        self.ligand_encoder = GraphEncoder(drug_feat_dim, filter_num, drug_edge_dim)
        self.cross_attn1 = nn.MultiheadAttention(
            embed_dim=filter_num, 
            num_heads=4, 
            dropout=0.2
        )
        
        self.cross_attn2 = nn.MultiheadAttention(
            embed_dim=filter_num, 
            num_heads=4, 
            dropout=0.2
        )

        self.classifier = nn.Sequential(
            nn.Linear(filter_num * 4, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 8),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(8, out_dim)
        )
        
        self.prot_pe = PositionalEncoding(d_model = filter_num, dropout = 0.2, max_len = 2800)
        self.mol_pe = PositionalEncoding(d_model = filter_num, dropout = 0.2, max_len = 30)

    def forward(self, data):
        
        protein = data[1]
        drug = data[0]
        
        # print(protein.x)
        # print(drug.x)
                
        prot_x = self.protein_encoder(protein.x, protein.edge_index, protein.edge_attr)
        ligand_x = self.ligand_encoder(drug.x, drug.edge_index, drug.edge_attr)
        
        # print(prot_x)
        # print(ligand_x)
        
        prot_x = apply_positional_encoding_to_batch(prot_x, protein.batch, self.prot_pe)
        mol_x = apply_positional_encoding_to_batch(ligand_x, drug.batch, self.mol_pe)
        
        # print(prot_x)
        # print(ligand_x)
        
        attn_mask1 = create_attention_mask(drug.batch, protein.batch)
        attn_mask2 = create_attention_mask(protein.batch, drug.batch)
        
        # print("masks")
        # print(attn_mask1)
        # print(attn_mask2)
        
        # print(attn_mask1)
        # print(attn_mask2)
        
        attn_feat1, _ = self.cross_attn1(prot_x, mol_x, mol_x, attn_mask=attn_mask1)
        attn_feat2, _ = self.cross_attn1(mol_x, prot_x, prot_x, attn_mask=attn_mask2)
        
        # print(attn_feat1)
        # print(attn_feat2)
        
        feat1 = gnn.global_mean_pool(attn_feat1, protein.batch)
        feat2 = gnn.global_mean_pool(attn_feat2, drug.batch)
        feat3 = gnn.global_mean_pool(prot_x, protein.batch)
        feat4 = gnn.global_mean_pool(ligand_x, drug.batch)
        
        # print(feat1)
        # print(feat2)
        # print(feat3)
        # print(feat4)

        x = torch.cat([feat1, feat2, feat3, feat4], dim=-1)
        
        # print(x)
        # print(x.shape)
        x = self.classifier(x)
        
        # print(x)
        # exit()

        return x


