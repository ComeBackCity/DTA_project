import torch
import torch.nn as nn
import torch.functional
from torch_geometric.nn import GINEConv, GlobalAttention
from torch_geometric.utils import to_dense_adj, degree, subgraph
from torch_geometric.data import Data

# --- Custom BatchNorm over all nodes in the batch ---
class NodeLevelBatchNorm(torch.nn.modules.batchnorm._BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum,
                         affine, track_running_stats)

    def _check_input_dim(self, x):
        if x.dim() != 2:
            raise ValueError(f'expected 2D input (got {x.dim()}D)')

    def forward(self, x):
        self._check_input_dim(x)
        if self.momentum is None:
            avg_factor = 0.0
        else:
            avg_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    avg_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    avg_factor = self.momentum

        return torch.functional.F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training or not self.track_running_stats,
            avg_factor,
            self.eps
        )

# --- Helpers for Laplacian Positional Encoding ---
def extract_individual_graphs(edge_index, batch):
    graphs = []
    for gid in batch.unique():
        mask = (batch == gid)
        sub_ei, _ = subgraph(mask, edge_index, relabel_nodes=True)
        graphs.append(Data(edge_index=sub_ei,
                           num_nodes=int(mask.sum())))
    return graphs

class LaplacianPositionalEncoding(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, edge_index, batch, N):
        device = edge_index.device
        pos = torch.zeros(N, self.k, device=device)
        for gid, g in enumerate(extract_individual_graphs(edge_index, batch)):
            n = g.num_nodes
            if n == 0:
                continue
            A = to_dense_adj(g.edge_index, max_num_nodes=n)[0]
            deg = degree(g.edge_index[0], n, dtype=torch.float32) + 1e-6
            Dinv = deg.pow(-0.5)
            L = torch.eye(n, device=device) - (Dinv.view(-1,1) * A * Dinv.view(1,-1))
            try:
                eigvals, eigvecs_full = torch.linalg.eigh(L)
                eigvecs = eigvecs_full[:, :min(self.k, eigvecs_full.size(1))]
                if eigvecs.size(1) < self.k:
                    padding = torch.zeros(n, self.k - eigvecs.size(1), device=device)
                    eigvecs = torch.cat([eigvecs, padding], dim=1)
            except RuntimeError:
                eigvecs = torch.randn(n, self.k, device=device)
            idx = (batch == gid).nonzero(as_tuple=True)[0]
            if idx.numel() == n:
                pos[idx] = eigvecs
        return pos


# --- 1D‑CNN block to replace MLP in GINEConv ---
class Conv1dBlock(nn.Module):
    def __init__(self, in_features, hidden, kernel_size=3, padding=1):
        super().__init__()
        self.in_features = in_features
        self.net = nn.Sequential(
            nn.Conv1d(in_features, hidden, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(hidden, in_features, kernel_size, padding=padding),
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.net(x)
        return x.squeeze(-1)

class GraphEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim,
                 edge_dim, n_layers, pe_dim=6, num_heads=4):
        super().__init__()
        self.pe = LaplacianPositionalEncoding(pe_dim)
        self.init_lin = nn.Sequential(
            nn.Linear(in_dim + pe_dim, hid_dim),
            nn.LeakyReLU(0.02),
            NodeLevelBatchNorm(hid_dim),
            nn.Dropout(0.4),
        )
        
        self.local_attention = nn.MultiheadAttention(
            embed_dim=hid_dim,
            num_heads=num_heads,
            dropout=0.2,
            batch_first=True
        )
        
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            conv_block = Conv1dBlock(hid_dim, hid_dim)
            dropout_rate = min(0.4 + (i * 0.05), 0.5)
            self.layers += [
                GINEConv(nn=conv_block, train_eps=True, edge_dim=edge_dim),
                NodeLevelBatchNorm(hid_dim),
                nn.LeakyReLU(0.02),
                nn.Dropout(dropout_rate),
            ]
            
        self.pool = GlobalAttention(gate_nn=nn.Sequential(
            nn.Linear(hid_dim, 1), nn.Sigmoid()
        ))
        self.out_lin = nn.Linear(hid_dim, out_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        N = x.size(0)
        pos = self.pe(edge_index, batch, N)
        x = self.init_lin(torch.cat([x, pos], dim=1))
        
        unique_batches = batch.unique()
        for b in unique_batches:
            mask = (batch == b)
            if mask.sum() > 0:
                local_x = x[mask]
                
                node_map = torch.full((N,), -1, dtype=torch.long, device=x.device)
                node_map[mask] = torch.arange(mask.sum(), device=x.device)

                edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
                local_edge_index = edge_index[:, edge_mask]
                local_edge_attr = edge_attr[edge_mask] if edge_attr is not None else None
                local_edge_index = node_map[local_edge_index]

                involved_nodes = torch.unique(local_edge_index.flatten())
                key_padding_mask = torch.ones((1, local_x.size(0)), dtype=torch.bool, device=x.device)
                key_padding_mask[0, involved_nodes] = False

                attn_output, _ = self.local_attention(
                    local_x.unsqueeze(0),
                    local_x.unsqueeze(0),
                    local_x.unsqueeze(0),
                    key_padding_mask=key_padding_mask
                )
                x[mask] = attn_output.squeeze(0)
        
        for layer in self.layers:
            if isinstance(layer, GINEConv):
                x_res = x
                x = layer(x, edge_index, edge_attr)
                x = x + x_res
            else:
                x = layer(x)
        
        x = self.pool(x, batch)
        return self.out_lin(x)

import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0, is_causal=False, enable_gqa=False):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must divide num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.is_causal = is_causal
        self.enable_gqa = enable_gqa

        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, d_model * 2)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x_q, x_kv, attn_mask=None):
        B, Lq, _ = x_q.shape
        B, Lk, _ = x_kv.shape

        # Linear projections
        q = self.q_proj(x_q).view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, Lq, head_dim)
        kv = self.kv_proj(x_kv).view(B, Lk, 2, self.num_heads, self.head_dim)
        k, v = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)  # (B, heads, Lk, head_dim)

        # Efficient Scaled Dot-Product Attention
        out, attn_weight = self.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout,
            is_causal=self.is_causal,
            enable_gqa=self.enable_gqa
        )

        # Final linear projection
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        out = self.out_proj(out)
        return out, attn_weight

    def scaled_dot_product_attention(self, query, key, value,
                                      attn_mask=None, dropout_p=0.0,
                                      is_causal=False, scale=None,
                                      enable_gqa=False):
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)

        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
            attn_bias = attn_bias.masked_fill(temp_mask.logical_not(), float("-inf"))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias = attn_bias.masked_fill(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias = attn_bias + attn_mask

        if enable_gqa:
            key = key.repeat_interleave(query.size(-3) // key.size(-3), dim=-3)
            value = value.repeat_interleave(query.size(-3) // value.size(-3), dim=-3)

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight = attn_weight + attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

        out = attn_weight @ value
        return out, attn_weight

class Mlp(nn.Module):
    def __init__(self, in_f, hid_f, out_f, drop=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_f, hid_f),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hid_f),
            nn.Dropout(drop),
            
            nn.Linear(hid_f, hid_f * 2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hid_f * 2),
            nn.Dropout(drop + 0.1),
            
            nn.Linear(hid_f * 2, hid_f),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hid_f),
            nn.Dropout(drop),
            
            nn.Linear(hid_f, out_f),
        )
    def forward(self, x):
        return self.net(x)

class MGraphDTA(nn.Module):
    def __init__(self,
                 prot_feat_dim, drug_feat_dim,
                 prot_edge_dim, drug_edge_dim,
                 hid_dim=512,
                 out_f=1,
                 pe_dim=10, prot_layers=6,
                 drug_layers=3, heads=4):
        super().__init__()
        self.prot_enc = GraphEncoder(
            in_dim=prot_feat_dim, hid_dim=hid_dim,
            out_dim=hid_dim, edge_dim=prot_edge_dim,
            n_layers=prot_layers, pe_dim=pe_dim
        )
        self.drug_enc = GraphEncoder(
            in_dim=drug_feat_dim, hid_dim=hid_dim,
            out_dim=hid_dim, edge_dim=drug_edge_dim,
            n_layers=drug_layers, pe_dim=pe_dim
        )
        
        self.cross1 = MultiHeadAttention(hid_dim, heads, dropout=0.5)
        self.cross2 = MultiHeadAttention(hid_dim, heads, dropout=0.5)
        
        self.classifier = Mlp(hid_dim*2, 256, out_f, drop=0.4)

    def forward(self, data):
        drug, prot, _ = data
        
        x_p = self.prot_enc(prot.x, prot.edge_index,
                            prot.edge_attr, prot.batch)
        x_d = self.drug_enc(drug.x, drug.edge_index,
                            drug.edge_attr, drug.batch)

        x_p = x_p.unsqueeze(1)
        x_d = x_d.unsqueeze(1)

        a1, _ = self.cross1(x_p, x_d)
        a2, _ = self.cross2(x_d, x_p)

        x_p = (x_p + a1).squeeze(1)
        x_d = (x_d + a2).squeeze(1)

        return self.classifier(torch.cat([x_p, x_d], dim=1))
