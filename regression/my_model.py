import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GlobalAttention

class SimpleGATEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, edge_dim=None, num_layers=3, heads=4, dropout=0.15):
        super().__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.non_linears = nn.ModuleList()
        
        for i in range(num_layers):
            input_dim = in_dim if i == 0 else hidden_dim
            self.layers.append(
                GATv2Conv(input_dim, hidden_dim // heads, heads=heads, edge_dim=edge_dim, dropout=dropout, residual=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.non_linears.append(nn.LeakyReLU(0.2))
            
        self.final_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_attr=None):
        for layer, batch_norm, non_linearity in zip(self.layers, self.batch_norms, self.non_linears):
            x = layer(x, edge_index, edge_attr)
            x = batch_norm(x)
            x = non_linearity(x)
            x = self.dropout(x)
        return self.final_proj(x)

class SimpleCrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, q_feat, k_feat, q_batch, k_batch):
        Q = self.q_proj(q_feat)
        K = self.k_proj(k_feat)
        V = self.v_proj(k_feat)

        scores = torch.matmul(Q, K.T) / (Q.size(-1) ** 0.5)
        mask = q_batch[:, None] == k_batch[None, :]
        scores[~mask] = float('-inf')

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attended = self.out_proj(torch.matmul(attn, V))

        return attended, attn

class SimpleGATCrossModel(nn.Module):
    def __init__(self,
                 prot_feat_dim,
                 drug_feat_dim,
                 prot_edge_dim,
                 drug_edge_dim,
                 hidden_dim=128,
                 prot_layers=4,
                 drug_layers=2,
                 out_dim=1,
                 heads=4):
        super().__init__()
        self.prot_enc = SimpleGATEncoder(
            in_dim=prot_feat_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            edge_dim=prot_edge_dim,
            num_layers=prot_layers,
            heads=heads,
            dropout=0.15
        )

        self.drug_enc = SimpleGATEncoder(
            in_dim=drug_feat_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            edge_dim=drug_edge_dim,
            num_layers=drug_layers,
            heads=heads,
            dropout=0.15
        )

        self.cross_attn_p_to_d = SimpleCrossAttention(hidden_dim)
        self.cross_attn_d_to_p = SimpleCrossAttention(hidden_dim)

        # Weight vectors for each feature dimension
        self.fusion_weights_p = nn.Parameter(torch.ones(hidden_dim, 2) * 0.5)  # (hidden_dim, 2)
        self.fusion_weights_d = nn.Parameter(torch.ones(hidden_dim, 2) * 0.5)  # (hidden_dim, 2)

        self.pool_p = GlobalAttention(gate_nn=nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1)
        ))

        self.pool_d = GlobalAttention(gate_nn=nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Dropout(0.15),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1)
        ))

        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim // 4, out_dim)
        )

    def forward(self, data):
        drug, prot, _ = data
        x_d = self.drug_enc(drug.x, drug.edge_index, drug.edge_attr)
        x_p = self.prot_enc(prot.x, prot.edge_index, prot.edge_attr)

        a1, attn1 = self.cross_attn_d_to_p(x_p, x_d, prot.batch, drug.batch)
        a2, attn2 = self.cross_attn_p_to_d(x_d, x_p, drug.batch, prot.batch)

        # Apply softmax along the weight dimension for each feature
        weights_p = F.softmax(self.fusion_weights_p, dim=1)  # (hidden_dim, 2)
        weights_d = F.softmax(self.fusion_weights_d, dim=1)  # (hidden_dim, 2)

        # Element-wise multiplication and sum for each feature dimension
        x_p = x_p * weights_p[:, 0].unsqueeze(0) + a1 * weights_p[:, 1].unsqueeze(0)
        x_d = x_d * weights_d[:, 0].unsqueeze(0) + a2 * weights_d[:, 1].unsqueeze(0)

        x_p = self.pool_p(x_p, prot.batch)
        x_d = self.pool_d(x_d, drug.batch)

        combined = torch.cat([x_p, x_d], dim=1)
        out = self.out(combined)

        return out