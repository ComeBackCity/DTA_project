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
                
        out_dim = in_dim // num_heads
        self.egret_layer = gnn.GATv2Conv(in_channels=in_dim, 
                                        out_channels=out_dim,
                                        heads=num_heads,
                                        concat=True, 
                                        negative_slope=0.2, 
                                        dropout=0.2, add_self_loops = True, 
                                        edge_dim=edge_dim, fill_value = 'mean', bias = True)
        
        self.bn1 = NodeLevelBatchNorm(in_dim)
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
         
    
class GraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, edge_dim, layer_count) -> None:
        super().__init__()
        
        self.reshaper = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.02),
            NodeLevelBatchNorm(hidden_dim),
            nn.Dropout(0.2)
        )
        
        layers = [
            (
                egretblock(in_dim=hidden_dim, edge_dim=edge_dim, num_heads=4), 
                ('x, edge_index, edge_attr -> x')
            )
                for _ in range(layer_count)
        ]
        
        self.gnn_block = gnn.Sequential(
            'x, edge_index, edge_attr', layers
        )
        
        self.reducer = Mlp(
            in_features=hidden_dim, 
            hidden_features=hidden_dim, 
            out_features=out_dim,
            act_layer=nn.LeakyReLU(0.02),
            drop=0.2
        )
        
    def forward(self, x, edge_index, edge_attr): 
        
        z = self.reshaper(x)
        z = self.gnn_block(z, edge_index, edge_attr)
        z = self.reducer(z)

        return z
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        z = self.fc1(x)
        z = self.act(z)
        x = self.drop(z)
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

class MGraphDTA(nn.Module):
    def __init__(self, protein_feat_dim, drug_feat_dim, protein_edge_dim, drug_edge_dim, filter_num=32, out_dim=1):
        super().__init__()
    
        self.protein_encoder = GraphEncoder(protein_feat_dim, 512, filter_num, protein_edge_dim, 4)
        self.ligand_encoder = GraphEncoder(drug_feat_dim, 64, filter_num, drug_edge_dim, 2)
        
        # self.cross_attn1 = nn.MultiheadAttention(
        #     embed_dim=filter_num, 
        #     num_heads=4, 
        #     dropout=0.2
        # )
        
        # self.cross_attn2 = nn.MultiheadAttention(
        #     embed_dim=filter_num, 
        #     num_heads=4, 
        #     dropout=0.2
        # )
        
        self.cross_attn1 = MultiHeadAttention(
            d_model=filter_num, 
            num_heads=4, 
            dropout=0.2
        )
        
        self.cross_attn2 = MultiHeadAttention(
            d_model=filter_num, 
            num_heads=4, 
            dropout=0.2
        )

        self.classifier = Mlp(
            in_features=filter_num * 2,
            hidden_features=8,
            out_features=1,
            act_layer=nn.LeakyReLU(0.02),
            drop=0.2
        )
        
        self.prot_pe = PositionalEncoding(d_model = filter_num, dropout = 0.2, max_len = 2800)
        self.mol_pe = PositionalEncoding(d_model = filter_num, dropout = 0.2, max_len = 30)

    def forward(self, data):
        
        protein = data[1]
        drug = data[0]
                
        print("in protein")
        prot_x = self.protein_encoder(protein.x, protein.edge_index, protein.edge_attr)
        print("in drug")
        ligand_x = self.ligand_encoder(drug.x, drug.edge_index, drug.edge_attr)
        
        prot_x = apply_positional_encoding_to_batch(prot_x, protein.batch, self.prot_pe)
        mol_x = apply_positional_encoding_to_batch(ligand_x, drug.batch, self.mol_pe)
        
        attn_mask1 = create_attention_mask(drug.batch, protein.batch)
        attn_mask2 = create_attention_mask(protein.batch, drug.batch)
           
        attn_feat1, _ = self.cross_attn1(prot_x, mol_x, mask=attn_mask1)       
        attn_feat2, _ = self.cross_attn2(mol_x, prot_x, mask=attn_mask2)
        
        protein_x = prot_x + attn_feat1
        molecule_x = mol_x + attn_feat2
        
        feat1 = gnn.global_mean_pool(protein_x, protein.batch)
        feat2 = gnn.global_mean_pool(molecule_x, drug.batch)
        # feat3 = gnn.global_mean_pool(prot_x, protein.batch)
        # feat4 = gnn.global_mean_pool(mol_x, drug.batch)
        # feat5 = gnn.global_mean_pool(attn_feat1, protein.batch)
        # feat6 = gnn.global_mean_pool(attn_feat2, drug.batch)

        # x = torch.cat([feat1, feat2, feat3, feat4, feat5, feat6], dim=-1)
        x = torch.cat([feat1, feat2], dim=-1)
        x = self.classifier(x)

        return x


