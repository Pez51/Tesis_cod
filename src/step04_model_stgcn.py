import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class SpatioTemporalGNN(nn.Module):
    def __init__(self, in_features, hidden_dim=64, num_heads=2, dropout=0.2):
        super(SpatioTemporalGNN, self).__init__()
        
        # Proyección de entrada
        self.in_proj = nn.Linear(in_features, hidden_dim)
        
        # Bloque Espacial 1: GAT con conexión residual y LayerNorm
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # Bloque Espacial 2: GAT con conexión residual y LayerNorm
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Bloque Temporal: GRU de 2 capas
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        # Cabezal de Regresión (Predicción de casos en escala logarítmica)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
        # Cabezal de Clasificación (Probabilidad de brote)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_seq, edge_index):
        B, P, N, F = x_seq.shape
        
        # Repetir índices de aristas para todo el batch de forma vectorizada
        batch_edge_index = edge_index
        if B > 1:
            offsets = torch.arange(B, device=x_seq.device).repeat_interleave(edge_index.size(1)) * N
            batch_edge_src = edge_index[0].repeat(B) + offsets
            batch_edge_dst = edge_index[1].repeat(B) + offsets
            batch_edge_index = torch.stack([batch_edge_src, batch_edge_dst], dim=0)

        spatial_outputs = []
        for t in range(P):
            x_t = x_seq[:, t, :, :].reshape(B * N, F)
            
            # Proyección y primera capa GAT con Skip-Connection
            h0 = self.in_proj(x_t)
            h_gat1 = self.gat1(h0, batch_edge_index)
            h1 = self.norm1(h0 + self.relu(h_gat1))
            h1 = self.dropout(h1)
            
            # Segunda capa GAT con Skip-Connection
            h_gat2 = self.gat2(h1, batch_edge_index)
            h2 = self.norm2(h1 + self.relu(h_gat2))
            
            spatial_outputs.append(h2.reshape(B, N, -1))

        # Secuencia temporal para GRU: (B * N, P, hidden_dim)
        spatial_seq = torch.stack(spatial_outputs, dim=1).permute(0, 2, 1, 3).reshape(B * N, P, -1)
        gru_out, _ = self.gru(spatial_seq)
        h_last = gru_out[:, -1, :] # Último estado temporal (B * N, hidden_dim)

        # Inferencia multitarea
        pred_reg = self.regressor(h_last).reshape(B, N)
        pred_cls = self.classifier(h_last).reshape(B, N)

        return pred_reg, pred_cls