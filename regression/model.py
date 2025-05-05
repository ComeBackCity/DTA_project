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
            if n == 0: continue
            A = to_dense_adj(g.edge_index, max_num_nodes=n)[0]
            deg = degree(g.edge_index[0], n, dtype=torch.float32)
            Dinv = torch.diag(deg.pow(-0.5)); Dinv[torch.isinf(Dinv)] = 0.0
            L = torch.eye(n, device=device) - Dinv @ A @ Dinv
            try:
                eigvecs = torch.linalg.eigh(L)[1][:, :self.k]
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
        # x: [E, in_features] → [E, in_features, 1]
        x = x.unsqueeze(-1)
        x = self.net(x)         # [E, in_features, 1]
        return x.squeeze(-1)    # [E, in_features]

# --- GIN‑variant encoder with 1D‑CNN in GINEConv + residuals ---
class GraphEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim,
                 edge_dim, n_layers, pe_dim=6):
        super().__init__()
        self.pe = LaplacianPositionalEncoding(pe_dim)
        self.init_lin = nn.Sequential(
            nn.Linear(in_dim + pe_dim, hid_dim),
            nn.LeakyReLU(0.02),
            NodeLevelBatchNorm(hid_dim),
            nn.Dropout(0.2),
        )
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            conv_block = Conv1dBlock(hid_dim, hid_dim)
            self.layers += [
                GINEConv(nn=conv_block, train_eps=True, edge_dim=edge_dim),
                NodeLevelBatchNorm(hid_dim),
                nn.LeakyReLU(0.02),
                nn.Dropout(0.2),
            ]
        self.pool = GlobalAttention(gate_nn=nn.Sequential(
            nn.Linear(hid_dim, 1), nn.Sigmoid()
        ))
        self.out_lin = nn.Linear(hid_dim, out_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        N = x.size(0)
        pos = self.pe(edge_index, batch, N)
        x = self.init_lin(torch.cat([x, pos], dim=1))
        for layer in self.layers:
            if isinstance(layer, GINEConv):
                x_res = x
                x = layer(x, edge_index, edge_attr)
                x = x + x_res
            else:
                x = layer(x)
        x = self.pool(x, batch)
        return self.out_lin(x)

# --- Custom multi‑head cross‑attention with residuals ---
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must divide num_heads"
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.q_lin = nn.Linear(d_model, d_model)
        self.kv_lin = nn.Linear(d_model, d_model * 2)
        self.out_lin = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x_q, x_kv, mask=None):
        B, Lq, D = x_q.shape
        _, Lk, _ = x_kv.shape
        q = self.q_lin(x_q).view(B, Lq, self.num_heads, self.d_k).transpose(1,2)
        kv = self.kv_lin(x_kv).view(B, Lk, 2, self.num_heads, self.d_k)
        k, v = kv[:,:,0].transpose(1,2), kv[:,:,1].transpose(1,2)
        scores = (q @ k.transpose(-2,-1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask==0, float('-inf'))
        attn = self.softmax(scores); attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1,2).contiguous().view(B, Lq, D)
        return self.proj_drop(self.out_lin(out)), attn

# --- Simple MLP classifier ---
class Mlp(nn.Module):
    def __init__(self, in_f, hid_f, out_f, drop=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_f, hid_f),
            nn.LeakyReLU(),
            NodeLevelBatchNorm(hid_f),
            nn.Dropout(drop),
            nn.Linear(hid_f, out_f),
        )
    def forward(self, x):
        return self.net(x)

# --- Full MGraphDTA model with residual cross‑attention ---
class MGraphDTA(nn.Module):
    def __init__(self,
                 prot_feat_dim, drug_feat_dim,
                 prot_edge_dim, drug_edge_dim,
                 filt=64, out_f=1,
                 pe_dim=10, prot_layers=6,
                 drug_layers=3, heads=4):
        super().__init__()
        self.prot_enc = GraphEncoder(
            in_dim=prot_feat_dim, hid_dim=filt,
            out_dim=filt, edge_dim=prot_edge_dim,
            n_layers=prot_layers, pe_dim=pe_dim
        )
        self.drug_enc = GraphEncoder(
            in_dim=drug_feat_dim, hid_dim=filt,
            out_dim=filt, edge_dim=drug_edge_dim,
            n_layers=drug_layers, pe_dim=pe_dim
        )
        self.cross1 = MultiHeadAttention(filt, heads, dropout=0.2)
        self.cross2 = MultiHeadAttention(filt, heads, dropout=0.2)
        self.classifier = Mlp(filt*2, 128, out_f, drop=0.2)

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
