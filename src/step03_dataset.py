import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.utils import load_config

class SpatioTemporalEDADataset(Dataset):
    def __init__(self, data_tensor, outbreak_matrix, seq_len=8, pred_horizon=1, target_idx=0):
        self.X, self.Y_reg, self.Y_cls = [], [], []
        T, N, F = data_tensor.shape

        for t in range(T - seq_len - pred_horizon + 1):
            x_window = data_tensor[t : t + seq_len]                                      # (P, N, F)
            y_reg = data_tensor[t + seq_len + pred_horizon - 1, :, target_idx]           # (N,)
            y_cls = outbreak_matrix[t + seq_len + pred_horizon - 1, :]                   # (N,)

            self.X.append(x_window)
            self.Y_reg.append(y_reg)
            self.Y_cls.append(y_cls)

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.Y_reg = torch.tensor(np.array(self.Y_reg), dtype=torch.float32)
        self.Y_cls = torch.tensor(np.array(self.Y_cls), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y_reg[idx], self.Y_cls[idx]

def get_dataloaders():
    config = load_config()
    processed_dir = config["paths"]["processed_dir"]

    data_tensor = np.load(os.path.join(processed_dir, "tensor_eda_TNF.npy"))
    outbreak_matrix = np.load(os.path.join(processed_dir, "targets_outbreak.npy"))

    T, N, F = data_tensor.shape
    train_ratio = config["model_params"]["train_split"]
    val_ratio = config["model_params"]["val_split"]

    train_end = int(T * train_ratio)
    val_end = int(T * (train_ratio + val_ratio))

    # 1. Transformación Logarítmica: log(1 + x) para estabilizar la varianza
    data_log = np.log1p(data_tensor)

    train_raw = data_log[:train_end]
    val_raw = data_log[train_end:val_end]
    test_raw = data_log[val_end:]

    # 2. Estandarización Z-Score basada únicamente en el conjunto Train
    mean = np.mean(train_raw, axis=(0, 1), keepdims=True)
    std = np.std(train_raw, axis=(0, 1), keepdims=True) + 1e-5

    train_norm = (train_raw - mean) / std
    val_norm = (val_raw - mean) / std
    test_norm = (test_raw - mean) / std

    # 3. Cálculo de pos_weight para balancear la pérdida BCE de brotes
    train_outbreaks = outbreak_matrix[:train_end]
    num_pos = np.sum(train_outbreaks == 1)
    num_neg = np.sum(train_outbreaks == 0)
    pos_weight = float(num_neg / (num_pos + 1e-5))

    seq_len = config["model_params"]["seq_len"]
    pred_horizon = config["model_params"]["pred_horizon"]
    target_idx = config["model_params"]["target_idx"]
    batch_size = config["model_params"]["batch_size"]

    train_ds = SpatioTemporalEDADataset(train_norm, outbreak_matrix[:train_end], seq_len, pred_horizon, target_idx)
    val_ds = SpatioTemporalEDADataset(val_norm, outbreak_matrix[train_end:val_end], seq_len, pred_horizon, target_idx)
    test_ds = SpatioTemporalEDADataset(test_norm, outbreak_matrix[val_end:], seq_len, pred_horizon, target_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    scalers = {
        "mean": float(mean[:, :, target_idx].squeeze()),
        "std": float(std[:, :, target_idx].squeeze()),
        "pos_weight": pos_weight
    }

    print(f"[DATASET] Muestras -> Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"[DATASET] Factor de balance de brotes (pos_weight): {pos_weight:.2f}")
    return train_loader, val_loader, test_loader, scalers