import torch
import torch.nn as nn
import torch.functional
from torch_geometric.nn import GINEConv, GlobalAttention, TransformerConv
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

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must divide num_heads"
        self.d_k = d_model // num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Separate linear layers for query, key, and value
        self.q_lin = nn.Linear(d_model, d_model)
        self.k_lin = nn.Linear(d_model, d_model)
        self.v_lin = nn.Linear(d_model, d_model)
        self.out_lin = nn.Linear(d_model, d_model)
        
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # Initialize attention layers with Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.q_lin.weight, gain=1.0)
        nn.init.xavier_uniform_(self.k_lin.weight, gain=1.0)
        nn.init.xavier_uniform_(self.v_lin.weight, gain=1.0)
        nn.init.xavier_uniform_(self.out_lin.weight, gain=1.0)
        
        # Initialize biases to zero
        nn.init.zeros_(self.q_lin.bias)
        nn.init.zeros_(self.k_lin.bias)
        nn.init.zeros_(self.v_lin.bias)
        nn.init.zeros_(self.out_lin.bias)

    def forward(self, x_q, x_kv, batch_q, batch_kv, mask=None):
        # Add sequence dimension and move it to front
        x_q = x_q.unsqueeze(1).transpose(0, 1)  # [1, N_q, D]
        x_kv = x_kv.unsqueeze(1).transpose(0, 1)  # [1, N_kv, D]
        
        # Get batch sizes
        B_q = batch_q.max().item() + 1
        B_kv = batch_kv.max().item() + 1
        assert B_q == B_kv, "Query and key-value batches must match"
        
        # Project queries, keys, and values
        q = self.q_lin(x_q)  # [1, N_q, D]
        k = self.k_lin(x_kv)  # [1, N_kv, D]
        v = self.v_lin(x_kv)  # [1, N_kv, D]
        
        # Reshape for multi-head attention
        q = q.view(1, -1, self.num_heads, self.d_k).transpose(1, 2)  # [1, num_heads, N_q, d_k]
        k = k.view(1, -1, self.num_heads, self.d_k).transpose(1, 2)  # [1, num_heads, N_kv, d_k]
        v = v.view(1, -1, self.num_heads, self.d_k).transpose(1, 2)  # [1, num_heads, N_kv, d_k]
        
        # Create attention mask for valid protein-drug pairs
        attn_mask = (batch_q.unsqueeze(1) == batch_kv.unsqueeze(0))  # [N_q, N_kv]
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, N_q, N_kv]
        attn_mask = attn_mask.expand(1, self.num_heads, -1, -1)  # [1, num_heads, N_q, N_kv]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)  # [1, num_heads, N_q, N_kv]
        scores = scores.masked_fill(~attn_mask, float('-inf'))
        
        # Apply attention
        attn = self.softmax(scores)  # [1, num_heads, N_q, N_kv]
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # [1, num_heads, N_q, d_k]
        out = out.transpose(1, 2).contiguous()  # [1, N_q, num_heads, d_k]
        out = out.view(1, -1, self.d_model)  # [1, N_q, D]
        out = out.squeeze(0)  # [N_q, D]
        
        return self.proj_drop(self.out_lin(out)), attn

class GraphEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim,
                 edge_dim, n_layers, pe_dim=6, num_heads=4):
        super().__init__()
        self.pe = LaplacianPositionalEncoding(pe_dim)
        
        # Initialize input projection
        self.init_lin = nn.Sequential(
            nn.Linear(in_dim + pe_dim, hid_dim),
            nn.LeakyReLU(0.02),
            NodeLevelBatchNorm(hid_dim),
            nn.Dropout(0.3),
        )
        
        # TransformerConv for better long-range interactions
        self.transformer = TransformerConv(
            in_channels=hid_dim,
            out_channels=hid_dim,
            heads=num_heads,
            concat=False,
            beta=True,  # Enable skip connection
            edge_dim=edge_dim,
            dropout=0.3
        )
        
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            conv_block = Conv1dBlock(hid_dim, hid_dim)
            dropout_rate = min(0.3 + (i * 0.05), 0.4)
            self.layers += [
                GINEConv(nn=conv_block, train_eps=True, edge_dim=edge_dim),
                NodeLevelBatchNorm(hid_dim),
                nn.LeakyReLU(0.02),
                nn.Dropout(dropout_rate),
            ]
            
        self.pool = GlobalAttention(gate_nn=nn.Sequential(
            nn.Linear(hid_dim, hid_dim // 2),
            nn.LeakyReLU(0.02),
            nn.Linear(hid_dim // 2, 1),
            nn.Sigmoid()
        ))
        
        self.out_lin = nn.Linear(hid_dim, out_dim)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # Initialize input layer weights with Kaiming initialization
        nn.init.kaiming_normal_(self.init_lin[0].weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.zeros_(self.init_lin[0].bias)
        
        
        # Initialize GINEConv weights
        for i in range(0, len(self.layers), 4):
            nn.init.xavier_uniform_(self.layers[i].nn.net[0].weight, gain=1.0)
            nn.init.xavier_uniform_(self.layers[i].nn.net[2].weight, gain=1.0)
            nn.init.zeros_(self.layers[i].nn.net[0].bias)
            nn.init.zeros_(self.layers[i].nn.net[2].bias)
        
        # Initialize pooling weights
        nn.init.kaiming_normal_(self.pool.gate_nn[0].weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.pool.gate_nn[2].weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.zeros_(self.pool.gate_nn[0].bias)
        nn.init.zeros_(self.pool.gate_nn[2].bias)
        
        # Initialize output layer
        nn.init.xavier_uniform_(self.out_lin.weight, gain=1.0)
        nn.init.zeros_(self.out_lin.bias)

    def forward(self, x, edge_index, edge_attr, batch, pool=True):
        N = x.size(0)
        pos = self.pe(edge_index, batch, N)
        x = self.init_lin(torch.cat([x, pos], dim=1))
        
        # Apply transformer for long-range interactions
        x = self.transformer(x, edge_index, edge_attr)
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, GINEConv):
                x_res = x
                x = layer(x, edge_index, edge_attr)
                x = x + x_res
            else:
                x = layer(x)
        
        if pool:
            x = self.pool(x, batch)
            x = self.out_lin(x)
        
        return x

class Mlp(nn.Module):
    def __init__(self, in_f, hid_f, out_f, drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_f, hid_f),
            nn.LeakyReLU(0.02),
            nn.BatchNorm1d(hid_f),
            nn.Dropout(drop),
            
            nn.Linear(hid_f, hid_f * 2),
            nn.LeakyReLU(0.02),
            nn.BatchNorm1d(hid_f * 2),
            nn.Dropout(drop + 0.1),
            
            nn.Linear(hid_f * 2, hid_f),
            nn.LeakyReLU(0.02),
            nn.BatchNorm1d(hid_f),
            nn.Dropout(drop),
            
            nn.Linear(hid_f, out_f),
        )
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # Initialize MLP weights with Kaiming initialization
        for i in [0, 4, 8, 12]:  # Linear layer indices
            nn.init.kaiming_normal_(self.net[i].weight, mode='fan_in', nonlinearity='leaky_relu')
            nn.init.zeros_(self.net[i].bias)

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
        
        self.cross1 = MultiHeadAttention(hid_dim, heads, dropout=0.3)
        self.cross2 = MultiHeadAttention(hid_dim, heads, dropout=0.3)
        
        # Separate attention mechanisms for protein and drug
        self.prot_pool = GlobalAttention(gate_nn=nn.Sequential(
            nn.Linear(hid_dim, hid_dim // 2),
            nn.LeakyReLU(0.02),
            nn.Linear(hid_dim // 2, 1),
            nn.Sigmoid()
        ))
        
        self.drug_pool = GlobalAttention(gate_nn=nn.Sequential(
            nn.Linear(hid_dim, hid_dim // 2),
            nn.LeakyReLU(0.02),
            nn.Linear(hid_dim // 2, 1),
            nn.Sigmoid()
        ))
        
        # Final attention for combined features
        self.final_pool = GlobalAttention(gate_nn=nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim),
            nn.LeakyReLU(0.02),
            nn.Linear(hid_dim, 1),
            nn.Sigmoid()
        ))
        
        self.classifier = Mlp(hid_dim*2, 256, out_f, drop=0.3)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # Initialize protein pooling weights
        nn.init.kaiming_normal_(self.prot_pool.gate_nn[0].weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.prot_pool.gate_nn[2].weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.zeros_(self.prot_pool.gate_nn[0].bias)
        nn.init.zeros_(self.prot_pool.gate_nn[2].bias)
        
        # Initialize drug pooling weights
        nn.init.kaiming_normal_(self.drug_pool.gate_nn[0].weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.drug_pool.gate_nn[2].weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.zeros_(self.drug_pool.gate_nn[0].bias)
        nn.init.zeros_(self.drug_pool.gate_nn[2].bias)
        
        # Initialize final pooling weights
        nn.init.kaiming_normal_(self.final_pool.gate_nn[0].weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.final_pool.gate_nn[2].weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.zeros_(self.final_pool.gate_nn[0].bias)
        nn.init.zeros_(self.final_pool.gate_nn[2].bias)

    def forward(self, data):
        drug, prot, _ = data
        
        # Get node-level representations without pooling
        x_p = self.prot_enc(prot.x, prot.edge_index,
                            prot.edge_attr, prot.batch, pool=False)
        x_d = self.drug_enc(drug.x, drug.edge_index,
                            drug.edge_attr, drug.batch, pool=False)

        # Cross attention between protein and drug at node level
        a1, _ = self.cross1(x_p, x_d, prot.batch, drug.batch)
        a2, _ = self.cross2(x_d, x_p, drug.batch, prot.batch)

        # Combine with residual connections
        x_p = x_p + a1  # [N_p, D]
        x_d = x_d + a2  # [N_d, D]

        # Pool protein and drug separately with their own attention mechanisms
        x_p = self.prot_pool(x_p, prot.batch)  # [B, D]
        x_d = self.drug_pool(x_d, drug.batch)  # [B, D]

        # Combine features and apply final attention
        combined = torch.cat([x_p, x_d], dim=1)  # [B, 2D]
        
        # Create batch indices for final pooling
        batch_size = combined.size(0)
        batch_indices = torch.arange(batch_size, device=combined.device)
        final = self.final_pool(combined, batch_indices)  # [B, 2D]

        return self.classifier(final)
