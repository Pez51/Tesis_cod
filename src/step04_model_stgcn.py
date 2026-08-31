import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, gru_seq):
        scores = self.attn(gru_seq)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * gru_seq, dim=1)
        return context

class SpatioTemporalGNN(nn.Module):
    def __init__(self, num_nodes=1891, in_features=13, embed_dim=16, hidden_dim=64, num_heads=2, dropout=0.15):
        super(SpatioTemporalGNN, self).__init__()
        
        self.num_nodes = num_nodes
        self.node_embeddings = nn.Embedding(num_nodes, embed_dim)
        self.in_proj = nn.Linear(in_features + embed_dim, hidden_dim)
        
        # Convolución Espacial GAT
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Recurrencia Temporal GRU + Atención
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.temp_attn = TemporalAttention(hidden_dim)
        
        # Cabezal 1: Regresión Continua
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
        # Cabezal 2: Clasificación de Brotes Dual-Stream
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + 1, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_seq, edge_index):
        B, P, N, F = x_seq.shape
        
        node_ids = torch.arange(N, device=x_seq.device)
        node_embeds = self.node_embeddings(node_ids)
        node_embeds_exp = node_embeds.unsqueeze(0).unsqueeze(0).expand(B, P, N, -1)
        
        x_combined = torch.cat([x_seq, node_embeds_exp], dim=-1)

        if B > 1:
            offsets = torch.arange(B, device=x_seq.device).repeat_interleave(edge_index.size(1)) * N
            batch_edge_src = edge_index[0].repeat(B) + offsets
            batch_edge_dst = edge_index[1].repeat(B) + offsets
            batch_edge_index = torch.stack([batch_edge_src, batch_edge_dst], dim=0)
        else:
            batch_edge_index = edge_index

        spatial_outputs = []
        for t in range(P):
            x_t = x_combined[:, t, :, :].reshape(B * N, -1)
            
            h0 = self.in_proj(x_t)
            h_gat1 = self.gat1(h0, batch_edge_index)
            h1 = self.norm1(h0 + self.gelu(h_gat1))
            h1 = self.dropout(h1)
            
            h_gat2 = self.gat2(h1, batch_edge_index)
            h2 = self.norm2(h1 + self.gelu(h_gat2))
            
            spatial_outputs.append(h2.reshape(B, N, -1))

        spatial_seq = torch.stack(spatial_outputs, dim=1).permute(0, 2, 1, 3).reshape(B * N, P, -1)
        gru_out, _ = self.gru(spatial_seq)
        h_context = self.temp_attn(gru_out)

        pred_reg_flat = self.regressor(h_context)
        pred_reg = pred_reg_flat.reshape(B, N)

        h_cls_input = torch.cat([h_context, pred_reg_flat.detach()], dim=-1)
        pred_cls = self.classifier(h_cls_input).reshape(B, N)

        return pred_reg, pred_cls