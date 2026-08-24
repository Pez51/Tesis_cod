import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class SpatioTemporalGNN(nn.Module):
    def __init__(self, in_features, hidden_dim=64, num_heads=2, dropout=0.2):
        super(SpatioTemporalGNN, self).__init__()
        
        # Bloque Espacial: Graph Attention Network (GAT)
        self.gat1 = GATConv(in_features, hidden_dim, heads=num_heads, concat=True, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        
        # Bloque Temporal: Gated Recurrent Unit (GRU)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Cabezal 1: Regresión de casos futuros
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
        # Cabezal 2: Clasificación binaria de alerta/brote (0 o 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_seq, edge_index):
        """
        x_seq: Tensor de entrada (Batch_size, Seq_len, Num_nodes, Num_features)
        edge_index: Tensor de aristas del grafo (2, Num_edges)
        """
        B, P, N, F = x_seq.shape
        
        # 1. Procesamiento Espacial paso a paso temporal
        spatial_outputs = []
        for t in range(P):
            x_t = x_seq[:, t, :, :]               # (B, N, F)
            x_t_flat = x_t.reshape(B * N, F)      # Agrupación para convolución PyG
            
            # Repetir índices de aristas para todo el batch
            batch_edge_index = edge_index
            if B > 1:
                offsets = torch.arange(B, device=x_seq.device).repeat_interleave(edge_index.size(1)) * N
                batch_edge_src = edge_index[0].repeat(B) + offsets
                batch_edge_dst = edge_index[1].repeat(B) + offsets
                batch_edge_index = torch.stack([batch_edge_src, batch_edge_dst], dim=0)

            # Capas de Atención sobre Grafos
            h = self.gat1(x_t_flat, batch_edge_index)
            h = self.relu(h)
            h = self.dropout(h)
            h = self.gat2(h, batch_edge_index)
            h = self.relu(h)                      # (B * N, hidden_dim)
            
            h_spatial = h.reshape(B, N, -1)       # (B, N, hidden_dim)
            spatial_outputs.append(h_spatial)

        # 2. Procesamiento Temporal con GRU
        # Formato para GRU: (B * N, Seq_len, hidden_dim)
        spatial_seq = torch.stack(spatial_outputs, dim=1) # (B, P, N, hidden_dim)
        spatial_seq = spatial_seq.permute(0, 2, 1, 3).reshape(B * N, P, -1)

        gru_out, _ = self.gru(spatial_seq)
        h_temporal = gru_out[:, -1, :]                   # Último estado oculto: (B * N, hidden_dim)

        # 3. Predicciones Multitarea
        pred_reg = self.regressor(h_temporal).reshape(B, N)
        pred_cls = self.classifier(h_temporal).reshape(B, N)

        return pred_reg, pred_cls