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
        
        # Convolución Espacial con Skip Connections (Kapoor et al., 2020)
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # Recibe h1 concatenado con h0 original para evitar over-smoothing
        self.gat2 = GATConv(hidden_dim * 2, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Fusión Macro-Epidémica Global (STAN - Gao et al., 2021)
        # Recibe la señal local (hidden_dim) + señal global max-pooled (hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.temp_attn = TemporalAttention(hidden_dim)
        
        # Cabezal de Regresión Continua
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
        # Cabezal de Clasificación de Brotes Informado
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + 1 + embed_dim, 48),
            nn.LayerNorm(48),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(48, 1)
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
            
            # Skip connection de Kapoor et al.: concatena h1 con h0
            h1_cat = torch.cat([h1, h0], dim=-1)
            h_gat2 = self.gat2(h1_cat, batch_edge_index)
            h2 = self.norm2(h1 + self.gelu(h_gat2))
            
            # Macro-Pooling de STAN: Extrae el pulso epidémico nacional en la semana t
            h2_reshaped = h2.reshape(B, N, -1)
            global_wave, _ = torch.max(h2_reshaped, dim=1, keepdim=True) # (B, 1, hidden_dim)
            global_wave_exp = global_wave.expand(-1, N, -1)               # (B, N, hidden_dim)
            
            # Concatenación de contexto local + macro nacional
            h2_fused = torch.cat([h2_reshaped, global_wave_exp], dim=-1) # (B, N, hidden_dim * 2)
            spatial_outputs.append(h2_fused)

        spatial_seq = torch.stack(spatial_outputs, dim=1).permute(0, 2, 1, 3).reshape(B * N, P, -1)
        gru_out, _ = self.gru(spatial_seq)
        h_context = self.temp_attn(gru_out)

        pred_reg_flat = self.regressor(h_context)
        pred_reg = pred_reg_flat.reshape(B, N)

        node_embeds_flat = node_embeds.unsqueeze(0).expand(B, N, -1).reshape(B * N, -1)
        h_cls_input = torch.cat([h_context, pred_reg_flat, node_embeds_flat], dim=-1)
        pred_cls = self.classifier(h_cls_input).reshape(B, N)

        return pred_reg, pred_cls