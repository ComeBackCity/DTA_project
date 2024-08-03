import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

class EGRETLayer(MessagePassing):
    def __init__(self, in_dim, out_dim, edge_dim, use_bias, config_dict=None):
        super(EGRETLayer, self).__init__(aggr='add')  # "Add" aggregation (Step 5).
        
        # experimental hyperparams
        self.apply_attention = True
        self.transform_edge_for_att_calc = False
        self.apply_attention_on_edge = False
        self.aggregate_edge = False
        self.edge_dependent_attention = False
        self.self_loop = False
        self.self_node_transform = False and self.self_loop
        self.activation = None
        
        if config_dict is not None:
            self.apply_attention = config_dict['apply_attention']
            self.transform_edge_for_att_calc = config_dict['transform_edge_for_att_calc']
            self.apply_attention_on_edge = config_dict['apply_attention_on_edge']
            self.aggregate_edge = config_dict['aggregate_edge']
            self.edge_dependent_attention = config_dict['edge_dependent_attention']
            self.self_loop = config_dict['self_loop']
            self.self_node_transform = config_dict['self_node_transform'] and self.self_loop
            self.activation = config_dict['activation']
            self.feat_drop = nn.Dropout(config_dict['feat_drop'])
            self.attn_drop = nn.Dropout(config_dict['attn_drop'])
            self.edge_feat_drop = nn.Dropout(config_dict['edge_feat_drop'])
            self.use_batch_norm = config_dict['use_batch_norm']

        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=use_bias)
        self.bn_fc = nn.BatchNorm1d(num_features=out_dim) if self.use_batch_norm else nn.Identity()
        
        # equation (2)
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
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)
        if self.aggregate_edge:
            nn.init.xavier_normal_(self.fc_edge.weight, gain=gain)
        if self.self_node_transform:
            nn.init.xavier_normal_(self.fc_self.weight, gain=gain)
        if self.transform_edge_for_att_calc:
            nn.init.xavier_normal_(self.fc_edge_for_att_calc.weight, gain=gain)

    def forward(self, x, edge_index, edge_attr):
        x = self.feat_drop(x)
        edge_attr = self.edge_feat_drop(edge_attr)
        
        # equation (1)
        z = self.bn_fc(self.fc(x))
        
        # propagate_type: (x: Tensor, edge_attr: Tensor)
        return self.propagate(edge_index, x=z, edge_attr=edge_attr, size=None)

    def message(self, x_j, x_i, edge_attr):
        # equation (2)
        if self.edge_dependent_attention:
            if self.transform_edge_for_att_calc:
                edge_attr = self.bn_fc_edge_for_att_calc(self.fc_edge_for_att_calc(edge_attr))
            z2 = torch.cat([x_i, x_j, edge_attr], dim=1)
        else:
            z2 = torch.cat([x_i, x_j], dim=1)
            
        a = self.attn_fc(z2)
        
        if self.aggregate_edge:
            ez = self.bn_fc_edge(self.fc_edge(edge_attr))
            return {'e': F.leaky_relu(a, negative_slope=0.2), 'ez': ez, 'x_j': x_j}
        
        return {'e': F.leaky_relu(a), 'x_j': x_j}

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        alpha = None
        if self.apply_attention:
            alpha = self.attn_drop(softmax(inputs['e'], index, ptr, dim_size))
            out = torch.sum(alpha * inputs['x_j'], dim=1)
        else:
            out = torch.sum(inputs['x_j'], dim=1)
        
        if self.aggregate_edge:
            if self.apply_attention_on_edge:
                out += torch.sum(alpha * inputs['ez'], dim=1)
            else:
                out += torch.sum(inputs['ez'], dim=1)
        
        return out

    def update(self, aggr_out, x):
        if self.self_loop:
            if self.self_node_transform:
                aggr_out += self.bn_fc_self(self.fc_self(x))
            else:
                aggr_out += x
        
        if self.activation is not None:
            aggr_out = self.activation(aggr_out)
        
        return aggr_out

class MultiHeadEGRETLayer(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, num_heads, use_bias, merge='cat', config_dict=None):
        super(MultiHeadEGRETLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(EGRETLayer(in_dim, out_dim, edge_dim, use_bias, config_dict=config_dict))
        self.merge = merge

    def forward(self, x, edge_index, edge_attr):
        head_outs_all = [attn_head(x, edge_index, edge_attr) for attn_head in self.heads]
        head_outs = []
        head_attn_scores = []
        for x in head_outs_all:
            head_outs += [x[0]]
            head_attn_scores += [x[1].cpu().detach()]
        if self.merge == 'cat':
            return torch.cat(head_outs, dim=1), head_attn_scores
        else:
            return torch.mean(torch.stack(head_outs)), head_attn_scores

config_dict = {
    'use_batch_norm': False,
    'feat_drop': 0.0,
    'attn_drop': 0.0,
    'edge_feat_drop': 0.0,
    'hidden_dim': 32,
    'out_dim': 32,
    'apply_attention': True,
    'transform_edge_for_att_calc': True,
    'apply_attention_on_edge': True,
    'aggregate_edge': True,
    'edge_dependent_attention': True,
    'self_loop': False,
    'self_node_transform': True,
    'activation': None
}
