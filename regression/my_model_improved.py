import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GlobalAttention

class SimpleGATEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, edge_dim=None, num_layers=3, heads=4, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.02)
        
        for i in range(num_layers):
            input_dim = in_dim if i == 0 else hidden_dim
            self.layers.append(GATv2Conv(input_dim, hidden_dim // heads, heads=heads, edge_dim=edge_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            
        self.final_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_attr=None):
        for layer, batch_norm in zip(self.layers, self.batch_norms):
            x = layer(x, edge_index, edge_attr)
            x = batch_norm(x)
            x = self.leaky_relu(x)
            x = self.dropout(x)
        return self.final_proj(x)

class SimpleCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads 
        self.head_dim = dim // self.num_heads
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        
        # Separate linear projections for each head
        self.q_projs = nn.ModuleList([
            nn.Linear(dim, self.head_dim) for _ in range(self.num_heads)
        ])
        self.k_projs = nn.ModuleList([
            nn.Linear(dim, self.head_dim) for _ in range(self.num_heads)
        ])
        self.v_projs = nn.ModuleList([
            nn.Linear(dim, self.head_dim) for _ in range(self.num_heads)
        ])
        
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, q_feat, k_feat, q_batch, k_batch):
        # q_feat, k_feat: (num_nodes, feat_dim)
        # q_batch, k_batch: (num_nodes,)
        
        # Project queries, keys, and values for each head
        Q_heads = [proj(q_feat) for proj in self.q_projs]  # List of (num_nodes_q, head_dim)
        K_heads = [proj(k_feat) for proj in self.k_projs]  # List of (num_nodes_k, head_dim)
        V_heads = [proj(k_feat) for proj in self.v_projs]  # List of (num_nodes_k, head_dim)
        
        # Stack heads
        Q = torch.stack(Q_heads, dim=0)  # (num_heads, num_nodes_q, head_dim)
        K = torch.stack(K_heads, dim=0)  # (num_heads, num_nodes_k, head_dim)
        V = torch.stack(V_heads, dim=0)  # (num_heads, num_nodes_k, head_dim)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (num_heads, num_nodes_q, num_nodes_k)
        
        # Create attention mask for each head
        mask = q_batch[:, None] == k_batch[None, :]  # (num_nodes_q, num_nodes_k)
        mask = mask.unsqueeze(0).expand(self.num_heads, -1, -1)  # (num_heads, num_nodes_q, num_nodes_k)
        scores[~mask] = float('-inf')
        
        # Apply attention
        attn = F.softmax(scores, dim=-1)  # (num_heads, num_nodes_q, num_nodes_k)
        attn = self.attn_dropout(attn)
        attn = torch.nan_to_num(attn, nan=0.0)
        
        # Apply attention to values
        out = torch.matmul(attn, V)  # (num_heads, num_nodes_q, head_dim)
        
        # Reshape and combine heads
        out = out.transpose(0, 1).contiguous().view(-1, self.num_heads * self.head_dim)  # (num_nodes_q, dim)
        out = self.dropout(out)
        out = self.out_proj(out)  # (num_nodes_q, dim)
        
        # Average attention weights across heads for visualization
        attn = attn.mean(dim=0)  # (num_nodes_q, num_nodes_k)
        
        return out, attn

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
            heads=heads
        )

        self.drug_enc = SimpleGATEncoder(
            in_dim=drug_feat_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            edge_dim=drug_edge_dim,
            num_layers=drug_layers,
            heads=heads
        )

        self.cross_attn_p_to_d = SimpleCrossAttention(hidden_dim, num_heads=heads)
        self.cross_attn_d_to_p = SimpleCrossAttention(hidden_dim, num_heads=heads)

        self.fusion_weights = nn.Parameter(torch.tensor([0.5, 0.5]))

        self.pool = GlobalAttention(gate_nn=nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.02),
            nn.Linear(hidden_dim // 2, 1)
        ))

        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.02),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.02),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.LeakyReLU(0.02),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, out_dim)
        )

    def forward(self, data):
        drug, prot, _ = data
        x_d = self.drug_enc(drug.x, drug.edge_index, drug.edge_attr)
        x_p = self.prot_enc(prot.x, prot.edge_index, prot.edge_attr)

        a1, attn1 = self.cross_attn_d_to_p(x_p, x_d, prot.batch, drug.batch)
        a2, attn2 = self.cross_attn_p_to_d(x_d, x_p, drug.batch, prot.batch)

        weights = F.softmax(self.fusion_weights, dim=0)
        x_p = weights[0] * x_p + weights[1] * a1
        x_d = weights[0] * x_d + weights[1] * a2

        x_p = self.pool(x_p, prot.batch)
        x_d = self.pool(x_d, drug.batch)

        combined = torch.cat([x_p, x_d], dim=1)
        out = self.out(combined)

        return out 