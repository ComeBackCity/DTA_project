import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GlobalAttention
from egnn_pytorch import EGNN_Network, EGNN

# ------------------------
# Inception GNN block with GATv2 only
# ------------------------
class InceptionGATv2Block(nn.Module):
    def __init__(self, in_dim, out_dim, num_paths=3, heads=4):
        super().__init__()
        self.paths = nn.ModuleList([
            GATv2Conv(in_dim, out_dim // heads, heads=heads)
            for _ in range(num_paths)
        ])
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index):
        outs = [layer(x, edge_index) for layer in self.paths]
        out = torch.cat(outs, dim=-1)
        return self.norm(F.relu(out))


# ------------------------
# Inception GNN Stack
# ------------------------
class InceptionGNNStack(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, num_paths=3, heads=4):
        super().__init__()
        self.layers = nn.ModuleList([
            InceptionGATv2Block(
                in_dim if i == 0 else hidden_dim,
                hidden_dim,
                num_paths=num_paths,
                heads=heads
            ) for i in range(num_layers)
        ])

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
        return x


# ------------------------
# Dual Branch Encoder
# ------------------------
class DualBranchEncoder(nn.Module):
    def __init__(self,
                 coord_in_dim, scalar_in_dim, edge_in_dim,
                 hidden_dim,
                 num_egnn_layers,
                 num_gatv2_layers):
        super().__init__()
        self.egnn_stack = EGNN_Network(
            depth=num_egnn_layers,
            dim=coord_in_dim,
            edge_dim=edge_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_egnn_layers,
            dropout=0.1,
            norm='layer'
        )
        self.scalar_stack = InceptionGNNStack(scalar_in_dim, hidden_dim, num_gatv2_layers)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

    def forward(self, coord_x, scalar_x, pos, edge_index, edge_attr):
        coord_embed, _ = self.egnn_stack(coord_x, pos, edge_index, edge_attr)
        scalar_embed = self.scalar_stack(scalar_x, edge_index)
        return self.fusion(torch.cat([coord_embed, scalar_embed], dim=-1))


# ------------------------
# Cross-Attention Module
# ------------------------
class InteractionAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, prot_embed, mol_embed, prot_batch, mol_batch):
        Q = self.query_proj(prot_embed)
        K = self.key_proj(mol_embed)
        V = self.value_proj(mol_embed)

        scores = torch.matmul(Q, K.T) / (Q.size(-1) ** 0.5)
        mask = prot_batch[:, None] == mol_batch[None, :]
        scores[~mask] = float('-inf')

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        return torch.matmul(attn, V), attn


# ------------------------
# Final DualBranchAffinityModel
# ------------------------
class DualBranchAffinityModel(nn.Module):
    def __init__(self,
                 coord_dim, scalar_dim, edge_dim,
                 hidden_dim,
                 prot_egnn_layers=2, prot_gatv2_layers=2,
                 mol_egnn_layers=2, mol_gatv2_layers=2,
                 num_paths=3, heads=4,
                 out_dim=1):
        super().__init__()

        self.prot_encoder = DualBranchEncoder(
            coord_dim, scalar_dim, edge_dim, hidden_dim,
            prot_egnn_layers, prot_gatv2_layers
        )

        self.mol_encoder = DualBranchEncoder(
            coord_dim, scalar_dim, edge_dim, hidden_dim,
            mol_egnn_layers, mol_gatv2_layers
        )

        self.cross_attn = InteractionAttention(hidden_dim)

        self.pool = GlobalAttention(gate_nn=nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        ))

        self.out_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_dim)
        )

        self.attn_map = None

    def forward(self, prot, mol):
        prot_embed = self.prot_encoder(
            prot.coord_x, prot.scalar_x, prot.pos,
            prot.edge_index, prot.edge_attr
        )

        mol_embed = self.mol_encoder(
            mol.coord_x, mol.scalar_x, mol.pos,
            mol.edge_index, mol.edge_attr
        )

        interaction, attn = self.cross_attn(
            prot_embed, mol_embed, prot.batch, mol.batch
        )

        self.attn_map = attn.detach()

        fused = prot_embed + interaction
        pooled = self.pool(fused, prot.batch)

        return self.out_layer(pooled)

    def get_attention_map(self):
        return self.attn_map