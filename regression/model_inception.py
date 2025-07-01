import torch
import torch.nn as nn
import torch.functional
from torch_geometric.nn import GINEConv, GlobalAttention, TransformerConv
from torch_geometric.utils import to_dense_adj, degree, subgraph
from torch_geometric.data import Data

# Keep all existing classes from model.py
# ... existing code ...

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
        # Expand batch indices to create a mask matrix
        batch_q_expanded = batch_q.unsqueeze(1)  # [N_q, 1]
        batch_kv_expanded = batch_kv.unsqueeze(0)  # [1, N_kv]
        
        # Create mask where batch indices match (same protein-drug pair)
        attn_mask = (batch_q_expanded == batch_kv_expanded)  # [N_q, N_kv]
        
        # Expand mask for multi-head attention
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, N_q, N_kv]
        attn_mask = attn_mask.expand(1, self.num_heads, -1, -1)  # [1, num_heads, N_q, N_kv]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)  # [1, num_heads, N_q, N_kv]
        
        # Apply mask by setting masked positions to large negative value
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
            beta=True,
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
        
    def forward(self, x, edge_index, edge_attr, batch, pool=True):
        N = x.size(0)
        pos = self.pe(edge_index, batch, N)
        x = self.init_lin(torch.cat([x, pos], dim=1))
        
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
        
    def forward(self, x):
        return self.net(x)

class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels//4, kernel_size=1),
            nn.BatchNorm1d(out_channels//4),
            nn.LeakyReLU(0.02)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels//4, kernel_size=1),
            nn.BatchNorm1d(out_channels//4),
            nn.LeakyReLU(0.02),
            nn.Conv1d(out_channels//4, out_channels//4, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels//4),
            nn.LeakyReLU(0.02)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels//4, kernel_size=1),
            nn.BatchNorm1d(out_channels//4),
            nn.LeakyReLU(0.02),
            nn.Conv1d(out_channels//4, out_channels//4, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels//4),
            nn.LeakyReLU(0.02)
        )
        
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels//4, kernel_size=1),
            nn.BatchNorm1d(out_channels//4),
            nn.LeakyReLU(0.02)
        )
        
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)

class FeatureInception(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, dropout=0.3):
        super().__init__()
        
        # Inception blocks for processing features
        self.inception1 = InceptionBlock1D(in_dim, hidden_dim)
        self.inception2 = InceptionBlock1D(hidden_dim, hidden_dim)
        self.inception3 = InceptionBlock1D(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.final_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        # Process with Inception blocks
        # x is expected to be [batch_size, seq_len, features]
        # check if x is CLS embedding
        if len(x.shape) == 2:
            x = x.unsqueeze(2)
        else:
            x = x.transpose(1, 2)  # [batch_size, features, seq_len]
        x = self.inception1(x)
        x = self.dropout(x)
        x = self.inception2(x)
        x = self.dropout(x)
        x = self.inception3(x)
        x = self.pool(x).squeeze(-1)  # [batch_size, hidden_dim]
        x = self.final_proj(x)
        
        return x

class MGraphDTAInception(nn.Module):
    def __init__(self,
                 prot_feat_dim, drug_feat_dim,
                 prot_edge_dim, drug_edge_dim,
                 prot_lm_feat_dim, drug_lm_feat_dim,
                 hid_dim=512,
                 out_f=1,
                 pe_dim=10,
                 prot_layers=6,
                 drug_layers=3,
                 heads=4):
        super().__init__()
        
        # Initialize weights with proper methods
        def init_weights(m):
            if isinstance(m, nn.Linear):
                # Use Kaiming initialization for weights
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                # Initialize bias to small values
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, (nn.BatchNorm1d, NodeLevelBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Parameter):
                # Initialize learnable parameters
                if m.dim() == 1:  # For scalar parameters like weights
                    nn.init.constant_(m, 0.5)
                else:
                    nn.init.kaiming_normal_(m, mode='fan_in', nonlinearity='leaky_relu')

        # GNN Encoders
        self.prot_gnn_enc = GraphEncoder(
            in_dim=prot_feat_dim, hid_dim=hid_dim,
            out_dim=hid_dim, edge_dim=prot_edge_dim,
            n_layers=prot_layers, pe_dim=pe_dim
        )
        self.drug_gnn_enc = GraphEncoder(
            in_dim=drug_feat_dim, hid_dim=hid_dim,
            out_dim=hid_dim, edge_dim=drug_edge_dim,
            n_layers=drug_layers, pe_dim=pe_dim
        )
        
        # Feature Inception Modules
        self.prot_inception = FeatureInception(prot_lm_feat_dim, hid_dim)
        self.drug_inception = FeatureInception(drug_lm_feat_dim, hid_dim)
        
        # Learnable weights for combining features
        self.prot_weight = nn.Parameter(torch.ones(1) * 0.5)
        self.drug_weight = nn.Parameter(torch.ones(1) * 0.5)
        
        # Fusion layers
        self.prot_fusion = nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim),
            nn.LeakyReLU(0.02),
            nn.Dropout(0.3)
        )
        
        self.drug_fusion = nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim),
            nn.LeakyReLU(0.02),
            nn.Dropout(0.3)
        )

        # Cross Attention
        self.cross1 = MultiHeadAttention(hid_dim, heads, dropout=0.3)
        self.cross2 = MultiHeadAttention(hid_dim, heads, dropout=0.3)
        
        # Pooling layers
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
        
        # Final Pooling
        self.final_pool = GlobalAttention(gate_nn=nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim),
            nn.LeakyReLU(0.02),
            nn.Linear(hid_dim, 1),
            nn.Sigmoid()
        ))
        
        # Classifier
        self.classifier = Mlp(hid_dim*2, 256, out_f, drop=0.3)
        
        # Apply weight initialization
        self.apply(init_weights)
        
        # Initialize learnable weights specifically
        nn.init.constant_(self.prot_weight, 0.5)
        nn.init.constant_(self.drug_weight, 0.5)
        
        # Add debug mode flag
        self.debug = False  # Set to True to enable debugging

    def _init_attention_weights(self):
        """Initialize attention weights specifically"""
        for m in [self.cross1, self.cross2]:
            if hasattr(m, 'q_lin'):
                nn.init.xavier_uniform_(m.q_lin.weight)
                nn.init.xavier_uniform_(m.k_lin.weight)
                nn.init.xavier_uniform_(m.v_lin.weight)
                nn.init.xavier_uniform_(m.out_lin.weight)
                if m.q_lin.bias is not None:
                    nn.init.constant_(m.q_lin.bias, 0.0)
                    nn.init.constant_(m.k_lin.bias, 0.0)
                    nn.init.constant_(m.v_lin.bias, 0.0)
                    nn.init.constant_(m.out_lin.bias, 0.0)

    def _init_pooling_weights(self):
        """Initialize pooling layer weights specifically"""
        for pool in [self.prot_pool, self.drug_pool, self.final_pool]:
            for m in pool.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)

    def _init_gnn_weights(self):
        """Initialize GNN weights specifically"""
        for gnn in [self.prot_gnn_enc, self.drug_gnn_enc]:
            for m in gnn.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.01)
                elif isinstance(m, (nn.BatchNorm1d, NodeLevelBatchNorm)):
                    if m.weight is not None:
                        nn.init.constant_(m.weight, 1.0)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)

    def _init_inception_weights(self):
        """Initialize inception module weights specifically"""
        for inception in [self.prot_inception, self.drug_inception]:
            for m in inception.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.01)
                elif isinstance(m, nn.BatchNorm1d):
                    if m.weight is not None:
                        nn.init.constant_(m.weight, 1.0)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)

    def _init_fusion_weights(self):
        """Initialize fusion layer weights specifically"""
        for fusion in [self.prot_fusion, self.drug_fusion]:
            for m in fusion.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.01)

    def _init_classifier_weights(self):
        """Initialize classifier weights specifically"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def initialize_weights(self):
        """Initialize all weights in the model"""
        self._init_attention_weights()
        self._init_pooling_weights()
        self._init_gnn_weights()
        self._init_inception_weights()
        self._init_fusion_weights()
        self._init_classifier_weights()
        
        # Initialize learnable weights
        nn.init.constant_(self.prot_weight, 0.5)
        nn.init.constant_(self.drug_weight, 0.5)

    def forward(self, data):
        # data should be a batch of Data objects (protein_graph, drug_graph) paired together
        drug_data, prot_data, _ = data

        if self.debug:
            print("\n=== Input Data Shapes ===")
            print(f"Protein x: {prot_data.x.shape}, dtype: {prot_data.x.dtype}, device: {prot_data.x.device}")
            print(f"Drug x: {drug_data.x.shape}, dtype: {drug_data.x.dtype}, device: {drug_data.x.device}")
            print(f"Protein edge_index: {prot_data.edge_index.shape}, dtype: {prot_data.edge_index.dtype}")
            print(f"Drug edge_index: {prot_data.edge_index.shape}, dtype: {drug_data.edge_index.dtype}")
            print(f"Protein batch: {prot_data.batch.shape}, max batch: {prot_data.batch.max().item()}")
            print(f"Drug batch: {drug_data.batch.shape}, max batch: {drug_data.batch.max().item()}")
            print(f"Protein edge_attr: {prot_data.edge_attr.shape if hasattr(prot_data, 'edge_attr') else 'None'}")
            print(f"Drug edge_attr: {drug_data.edge_attr.shape if hasattr(drug_data, 'edge_attr') else 'None'}")
            print(f"Protein cls_embedding: {prot_data.cls_embedding.shape if hasattr(prot_data, 'cls_embedding') else 'None'}")
            print(f"Drug cls_embedding: {drug_data.cls_embedding.shape if hasattr(drug_data, 'cls_embedding') else 'None'}")

        # Process graphs with GNNs at node level (pool=False)
        x_p_gnn = self.prot_gnn_enc(
            prot_data.x,
            prot_data.edge_index,
            prot_data.edge_attr,
            prot_data.batch, 
            pool=False
        )
        
        x_d_gnn = self.drug_gnn_enc(
            drug_data.x,
            drug_data.edge_index,
            drug_data.edge_attr,
            drug_data.batch,
            pool=False
        )

        if self.debug:
            print("\n=== After GNN Encoding ===")
            print(f"Protein GNN output: {x_p_gnn.shape}, mean: {x_p_gnn.mean().item():.4f}, std: {x_p_gnn.std().item():.4f}")
            print(f"Drug GNN output: {x_d_gnn.shape}, mean: {x_d_gnn.mean().item():.4f}, std: {x_d_gnn.std().item():.4f}")
            print(f"Protein GNN min: {x_p_gnn.min().item():.4f}, max: {x_p_gnn.max().item():.4f}")
            print(f"Drug GNN min: {x_d_gnn.min().item():.4f}, max: {x_d_gnn.max().item():.4f}")
        
        # Apply Cross Attention at node level
        x_p_cross, attn_p = self.cross2(x_p_gnn, x_d_gnn, prot_data.batch, drug_data.batch)
        x_d_cross, attn_d = self.cross1(x_d_gnn, x_p_gnn, drug_data.batch, prot_data.batch)

        if self.debug:
            print("\n=== After Cross Attention ===")
            print(f"Protein cross output: {x_p_cross.shape}, mean: {x_p_cross.mean().item():.4f}, std: {x_p_cross.std().item():.4f}")
            print(f"Drug cross output: {x_d_cross.shape}, mean: {x_d_cross.mean().item():.4f}, std: {x_d_cross.std().item():.4f}")
            print(f"Attention weights - Protein: {attn_p.mean().item():.4f}, Drug: {attn_d.mean().item():.4f}")
            print(f"Attention shapes - Protein: {attn_p.shape}, Drug: {attn_d.shape}")
        
        # Combine features with learnable weights
        prot_weight = torch.sigmoid(self.prot_weight)
        drug_weight = torch.sigmoid(self.drug_weight)
        
        if self.debug:
            print("\n=== Learnable Weights ===")
            print(f"Protein weight: {prot_weight.item():.4f}")
            print(f"Drug weight: {drug_weight.item():.4f}")
        
        x_p_combined = prot_weight * x_p_gnn + (1 - prot_weight) * x_p_cross
        x_d_combined = drug_weight * x_d_gnn + (1 - drug_weight) * x_d_cross
        
        if self.debug:
            print("\n=== After Weight Combination ===")
            print(f"Protein combined: {x_p_combined.shape}, mean: {x_p_combined.mean().item():.4f}, std: {x_p_combined.std().item():.4f}")
            print(f"Drug combined: {x_d_combined.shape}, mean: {x_d_combined.mean().item():.4f}, std: {x_d_combined.std().item():.4f}")
        
        # Apply individual pooling
        prot_gnn_pooled = self.prot_pool(x_p_combined, prot_data.batch)
        drug_gnn_pooled = self.drug_pool(x_d_combined, drug_data.batch)

        if self.debug:
            print("\n=== After Pooling ===")
            print(f"Protein pooled: {prot_gnn_pooled.shape}, mean: {prot_gnn_pooled.mean().item():.4f}, std: {prot_gnn_pooled.std().item():.4f}")
            print(f"Drug pooled: {drug_gnn_pooled.shape}, mean: {drug_gnn_pooled.mean().item():.4f}, std: {drug_gnn_pooled.std().item():.4f}")

        # Process language model features
        if not hasattr(prot_data, 'cls_embedding') or prot_data.cls_embedding is None:
            raise ValueError("Protein data object missing 'cls_embedding' attribute.")
        if not hasattr(drug_data, 'cls_embedding') or drug_data.cls_embedding is None:
            raise ValueError("Drug data object missing 'cls_embedding' attribute.")

        inception_prot_out = self.prot_inception(prot_data.cls_embedding)
        inception_drug_out = self.drug_inception(drug_data.cls_embedding)

        if self.debug:
            print("\n=== After Inception ===")
            print(f"Protein inception: {inception_prot_out.shape}, mean: {inception_prot_out.mean().item():.4f}, std: {inception_prot_out.std().item():.4f}")
            print(f"Drug inception: {inception_drug_out.shape}, mean: {inception_drug_out.mean().item():.4f}, std: {inception_drug_out.std().item():.4f}")

        # Combine features
        prot_fused = self.prot_fusion(torch.cat([prot_gnn_pooled, inception_prot_out], dim=1))
        drug_fused = self.drug_fusion(torch.cat([drug_gnn_pooled, inception_drug_out], dim=1))

        if self.debug:
            print("\n=== After Fusion ===")
            print(f"Protein fused: {prot_fused.shape}, mean: {prot_fused.mean().item():.4f}, std: {prot_fused.std().item():.4f}")
            print(f"Drug fused: {drug_fused.shape}, mean: {drug_fused.mean().item():.4f}, std: {drug_fused.std().item():.4f}")

        combined = torch.cat([prot_fused, drug_fused], dim=1)
        batch_size = combined.size(0)
        batch_indices = torch.arange(batch_size, device=combined.device)
        final = self.final_pool(combined, batch_indices)

        if self.debug:
            print("\n=== Before Classifier ===")
            print(f"Final pooled: {final.shape}, mean: {final.mean().item():.4f}, std: {final.std().item():.4f}")

        out = self.classifier(final)

        if self.debug:
            print("\n=== Final Output ===")
            print(f"Output shape: {out.shape}, mean: {out.mean().item():.4f}, std: {out.std().item():.4f}")
            print(f"Output min: {out.min().item():.4f}, max: {out.max().item():.4f}\n")

        return out 