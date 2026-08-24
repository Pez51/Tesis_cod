import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.utils import load_config

class SpatioTemporalEDADataset(Dataset):
    def __init__(self, data_tensor, outbreak_matrix, seq_len=8, pred_horizon=1, target_idx=0):
        """
        data_tensor: Tensor normalizado de forma (T, N, F)
        outbreak_matrix: Matriz binaria de brotes (T, N)
        seq_len (P): Semanas históricas de entrada
        pred_horizon (H): Semanas a futuro para la predicción
        target_idx: Índice de la variable objetivo (0 = episodios_men5)
        """
        self.X, self.Y_reg, self.Y_cls = [], [], []
        T, N, F = data_tensor.shape

        for t in range(T - seq_len - pred_horizon + 1):
            # Ventana histórica de entrada: (P, N, F)
            x_window = data_tensor[t : t + seq_len]
            # Objetivo de regresión en t + P + H - 1: (N,)
            y_reg = data_tensor[t + seq_len + pred_horizon - 1, :, target_idx]
            # Objetivo de clasificación de brote en t + P + H - 1: (N,)
            y_cls = outbreak_matrix[t + seq_len + pred_horizon - 1, :]

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

    # 1. Cargar tensores procesados
    data_tensor = np.load(os.path.join(processed_dir, "tensor_eda_TNF.npy"))
    outbreak_matrix = np.load(os.path.join(processed_dir, "targets_outbreak.npy"))

    T, N, F = data_tensor.shape
    train_ratio = config["model_params"]["train_split"]
    val_ratio = config["model_params"]["val_split"]

    train_end = int(T * train_ratio)
    val_end = int(T * (train_ratio + val_ratio))

    # 2. División temporal estricta
    train_raw = data_tensor[:train_end]
    val_raw = data_tensor[train_end:val_end]
    test_raw = data_tensor[val_end:]

    # 3. Normalización Min-Max calculada únicamente sobre Train (evita Data Leakage)
    scaler_min = train_raw.min(axis=(0, 1), keepdims=True)
    scaler_max = train_raw.max(axis=(0, 1), keepdims=True)
    scaler_max[scaler_max == scaler_min] += 1e-5  # Evitar división entre cero

    train_norm = (train_raw - scaler_min) / (scaler_max - scaler_min)
    val_norm = (val_raw - scaler_min) / (scaler_max - scaler_min)
    test_norm = (test_raw - scaler_min) / (scaler_max - scaler_min)

    seq_len = config["model_params"]["seq_len"]
    pred_horizon = config["model_params"]["pred_horizon"]
    target_idx = config["model_params"]["target_idx"]
    batch_size = config["model_params"]["batch_size"]

    # 4. Creación de Datasets
    train_ds = SpatioTemporalEDADataset(train_norm, outbreak_matrix[:train_end], seq_len, pred_horizon, target_idx)
    val_ds = SpatioTemporalEDADataset(val_norm, outbreak_matrix[train_end:val_end], seq_len, pred_horizon, target_idx)
    test_ds = SpatioTemporalEDADataset(test_norm, outbreak_matrix[val_end:], seq_len, pred_horizon, target_idx)

    # 5. DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    scalers = {
        "min": scaler_min[:, :, target_idx].squeeze(),
        "max": scaler_max[:, :, target_idx].squeeze()
    }

    print(f"[DATASET] Muestras generadas -> Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader, scalers