import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.norm import BatchNorm
import torch_geometric.nn as gnn
from torch import Tensor
from collections import OrderedDict
from torch_geometric.utils import to_dense_adj, to_scipy_sparse_matrix
import numpy as np
from scipy.sparse.linalg import eigsh

# EGRET Layer Definition
class EGRETLayer(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, use_bias, config_dict=None):
        super(EGRETLayer, self).__init__()
        self.apply_attention = True
        self.transform_edge_for_att_calc = False
        self.apply_attention_on_edge = False
        self.aggregate_edge = False
        self.edge_dependent_attention = False
        self.self_loop = False  # or skip connection
        self.self_node_transform = False and self.self_loop
        self.activation = None
        if config_dict is not None:
            self.apply_attention = config_dict['apply_attention']
            self.transform_edge_for_att_calc = config_dict['transform_edge_for_att_calc']
            self.apply_attention_on_edge = config_dict['apply_attention_on_edge']
            self.aggregate_edge = config_dict['aggregate_edge']
            self.edge_dependent_attention = config_dict['edge_dependent_attention']
            self.self_loop = config_dict['self_loop']  # or skip connection
            self.self_node_transform = config_dict['self_node_transform'] and self.self_loop
            self.activation = config_dict['activation']
            self.feat_drop = nn.Dropout(config_dict['feat_drop'])
            self.attn_drop = nn.Dropout(config_dict['attn_drop'])
            self.edge_feat_drop = nn.Dropout(config_dict['edge_feat_drop'])
            self.use_batch_norm = config_dict['use_batch_norm']

        self.fc = nn.Linear(in_dim, out_dim, bias=use_bias)
        self.bn_fc = nn.BatchNorm1d(num_features=out_dim) if self.use_batch_norm else nn.Identity()
        if self.edge_dependent_attention:
            self.attn_fc = nn.Linear(2 * out_dim + edge_dim, 1, bias=use_bias)
        else:
            self.attn_fc = nn.Linear(2 * out_dim, 1, bias=use_bias)
        if self.aggregate_edge:
            self.fc_edge = nn.Linear(edge_dim, out_dim, bias=use_bias)
            self.bn_fc_edge = nn.BatchNorm1d(num_features=out_dim) if self.use_batch_norm else nn.Identity()
        if self.self_node_transform:
            self.fc_self = nn.Linear(in_dim, out_dim, bias=use_bias)
            self.bn_fc_self = nn.BatchNorm1d(num_features=out_dim) if self.use_batch_norm else nn.Identity()
        if self.transform_edge_for_att_calc:
            self.fc_edge_for_att_calc = nn.Linear(edge_dim, edge_dim, bias=use_bias)
            self.bn_fc_edge_for_att_calc = nn.BatchNorm1d(num_features=edge_dim) if self.use_batch_norm else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)
        if self.aggregate_edge:
            nn.init.xavier_normal_(self.fc_edge.weight, gain=gain)
        if self.self_node_transform:
            nn.init.xavier_normal_(self.fc_self.weight, gain=gain)
        if self.transform_edge_for_att_calc:
            nn.init.xavier_normal_(self.fc_edge_for_att_calc.weight, gain=gain)

    def edge_attention(self, edges):
        if self.edge_dependent_attention:
            if self.transform_edge_for_att_calc:
                z2 = torch.cat([edges.src['z'], edges.dst['z'], self.bn_fc_edge_for_att_calc(self.fc_edge_for_att_calc(edges.data['ex']))], dim=1)
            else:
                z2 = torch.cat([edges.src['z'], edges.dst['z'], edges.data['ex']], dim=1)
        else:
            z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)

        if self.aggregate_edge:
            ez = self.bn_fc_edge(self.fc_edge(edges.data['ex']))
            return {'e': F.leaky_relu(a, negative_slope=0.2), 'ez': ez}
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        if self.aggregate_edge:
            return {'z': edges.src['z'], 'e': edges.data['e'], 'ez': edges.data['ez']}
        else:
            return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        if not self.apply_attention:
            h = torch.sum(nodes.mailbox['z'], dim=1)
        else:
            alpha = self.attn_drop(F.softmax(nodes.mailbox['e'], dim=1))
            h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        if self.aggregate_edge:
            if self.apply_attention_on_edge:
                h = h + torch.sum(alpha * nodes.mailbox['ez'], dim=1)
            else:
                h = h + torch.sum(nodes.mailbox['ez'], dim=1)
        return {'h': h}

    def forward(self, data):
        edge_index = data.edge_index
        edge_attr = data.edge_attr if 'edge_attr' in data else torch.ones(edge_index.size(1), 1, device=edge_index.device)

        data.x = self.feat_drop(data.x)
        edge_attr = self.edge_feat_drop(edge_attr)
        data.z = self.bn_fc(self.fc(data.x))

        edge_index, _ = gnn.utils.add_self_loops(edge_index, num_nodes=data.x.size(0))
        data.edge_index = edge_index
        data.edge_attr = edge_attr

        row, col = edge_index
        edge_idx = torch.arange(edge_index.size(1), device=edge_index.device)
        data.z_col = data.z[col]
        data.z_row = data.z[row]
        data.z_edge = edge_attr
        out = self.edge_attention(data)
        edge_weight = out['e']
        data.edge_attr = out['e']
        if self.aggregate_edge:
            data.edge_attr = out['ez']

        data.edge_weight = edge_weight
        data.x = gnn.utils.scatter_(gnn.utils.segment_sum, edge_weight.view(-1) * data.z[row], col, dim=0, dim_size=data.z.size(0))
        if self.self_loop:
            data.x = data.x + data.z

        if self.activation is not None:
            data.x = self.activation(data.x)
        return data

# Laplacian Positional Encoding
def laplacian_positional_encoding(data, num_positional_features):
    adj = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
    L = np.diag(adj.sum(1).cpu()) - adj.cpu().numpy()
    _, vec = eigsh(L, k=num_positional_features, which='SM')
    data.pos = torch.from_numpy(vec).float().to(data.edge_index.device)
    return data
