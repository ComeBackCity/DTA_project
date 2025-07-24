import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GlobalAttention
from gvp import StructureEncoder  # Import your GVP block

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

class AttentionFusion(nn.Module):
    """
    Bidirectional cross-attention fusion between GAT and GVP features.
    """

    def __init__(self, hidden_dim, num_heads=4, dropout=0.15):
        super().__init__()
        self.cross_attn_gat_to_gvp = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_gvp_to_gat = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        # Input size = 4 * hidden_dim (gat, gvp, attn_gat_to_gvp, attn_gvp_to_gat)
        self.linear_stack = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, gat_feat, gvp_feat):
        """
        Args:
            gat_feat: (N, D)
            gvp_feat: (N, D)
        Returns:
            fused: (N, D)
        """

        # GAT → GVP cross-attention
        attn_gat_to_gvp, _ = self.cross_attn_gat_to_gvp(gat_feat.unsqueeze(1), gvp_feat.unsqueeze(1), gvp_feat.unsqueeze(1))
        attn_gat_to_gvp = attn_gat_to_gvp.squeeze(1)  # (N, D)

        # GVP → GAT cross-attention
        attn_gvp_to_gat, _ = self.cross_attn_gvp_to_gat(gvp_feat.unsqueeze(1), gat_feat.unsqueeze(1), gat_feat.unsqueeze(1))
        attn_gvp_to_gat = attn_gvp_to_gat.squeeze(1)  # (N, D)

        # Concatenate: original features + attentions
        fused_input = torch.cat([gat_feat, gvp_feat, attn_gat_to_gvp, attn_gvp_to_gat], dim=1)  # (N, 4D)

        return self.linear_stack(fused_input)


class SimpleGATGVPCrossModel(nn.Module):
    def __init__(self,
                 prot_feat_dim,
                 drug_feat_dim,
                 prot_edge_dim,
                 drug_edge_dim,
                 hidden_dim=128,
                 prot_layers=4,
                 prot_gvp_layer=3,
                 drug_layers=2,
                 dropout=0.15,  
                 out_dim=1,
                 heads=4):
        super().__init__()

        self.prot_gat = SimpleGATEncoder(
            in_dim=prot_feat_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            edge_dim=prot_edge_dim,
            num_layers=prot_layers,
            heads=heads,
            dropout=dropout
        )

        self.prot_gvp = StructureEncoder(
            node_in_dim=(6, 3),
            node_h_dim=(hidden_dim, 16),
            edge_in_dim=(32, 1),
            edge_h_dim=(32, 1),
            seq_in=False,
            num_layers=prot_gvp_layer,
            drop_rate=dropout
        )

        self.prot_fusion = AttentionFusion(hidden_dim=hidden_dim, num_heads=4, dropout=dropout)

        self.drug_enc = SimpleGATEncoder(
            in_dim=drug_feat_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            edge_dim=drug_edge_dim,
            num_layers=drug_layers,
            heads=heads,
            dropout=dropout
        )

        self.cross_attn_p_to_d = SimpleCrossAttention(hidden_dim)
        self.cross_attn_d_to_p = SimpleCrossAttention(hidden_dim)

        self.fusion_weights_p = nn.Parameter(torch.ones(hidden_dim, 2) * 0.5)
        self.fusion_weights_d = nn.Parameter(torch.ones(hidden_dim, 2) * 0.5)

        self.pool_p = GlobalAttention(gate_nn=nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1)
        ))

        self.pool_d = GlobalAttention(gate_nn=nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Dropout(dropout),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1)
        ))

        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, out_dim)
        )

    def forward(self, data):
        drug, prot_gat, prot_gvp, _ = data

        x_d = self.drug_enc(drug.x, drug.edge_index, drug.edge_attr)
        x_p_gat = self.prot_gat(prot_gat.x, prot_gat.edge_index, prot_gat.edge_attr)
        x_p_gvp = self.prot_gvp((prot_gvp.node_s, prot_gvp.node_v), (prot_gvp.edge_s, prot_gvp.edge_v), prot_gvp.edge_index)

        x_p = self.prot_fusion(x_p_gat, x_p_gvp)

        a1, attn1 = self.cross_attn_d_to_p(x_p, x_d, prot_gat.batch, drug.batch)
        a2, attn2 = self.cross_attn_p_to_d(x_d, x_p, drug.batch, prot_gat.batch)

        weights_p = F.softmax(self.fusion_weights_p, dim=1)
        weights_d = F.softmax(self.fusion_weights_d, dim=1)

        x_p = x_p * weights_p[:, 0].unsqueeze(0) + a1 * weights_p[:, 1].unsqueeze(0)
        x_d = x_d * weights_d[:, 0].unsqueeze(0) + a2 * weights_d[:, 1].unsqueeze(0)

        x_p = self.pool_p(x_p, prot_gat.batch)
        x_d = self.pool_d(x_d, drug.batch)

        combined = torch.cat([x_p, x_d], dim=1)
        return self.out(combined)
