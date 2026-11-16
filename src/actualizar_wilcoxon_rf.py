import os
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestRegressor

from src.utils import load_config, load_pickle
from src.step03_dataset import get_dataloaders
from src.step04_model_stgcn import SpatioTemporalGNN


def main():
    np.random.seed(42)
    config = load_config()
    processed_dir = config["paths"]["processed_dir"]
    models_dir = config["paths"]["models_dir"]
    reports_dir = "reports"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Cargar datos y obtener errores absolutos del ST-GNN en Test
    _, _, test_loader, scalers = get_dataloaders(batch_size=32)
    graph_data = torch.load(os.path.join(processed_dir, "graph_topology.pt"), weights_only=True)
    edge_index = graph_data["edge_index"].to(device)
    num_nodes = graph_data["num_nodes"]

    mean_target = scalers["mean_target"]
    std_target = scalers["std_target"]

    test_inputs, all_targets_reg = [], []
    for bx, by_reg, _ in test_loader:
        test_inputs.append(bx)
        by_reg_abs = np.clip(np.expm1(by_reg.numpy() * std_target + mean_target), 0, None)
        all_targets_reg.append(by_reg_abs)

    y_test_reg_true = np.concatenate(all_targets_reg, axis=0)  # (184, 196)
    num_features = test_inputs[0].shape[-1]

    stgnn_model = SpatioTemporalGNN(
        num_nodes=num_nodes, in_features=num_features, embed_dim=8, hidden_dim=64, dropout=0.15
    ).to(device)
    stgnn_model.load_state_dict(
        torch.load(os.path.join(models_dir, "best_stgnn_model.pt"), map_location=device, weights_only=True)
    )
    stgnn_model.eval()

    preds_stgnn_reg = []
    with torch.no_grad():
        for bx in test_inputs:
            p_reg, _ = stgnn_model(bx.to(device), edge_index)
            reg_abs = np.clip(np.expm1(p_reg.cpu().numpy() * std_target + mean_target), 0, None)
            preds_stgnn_reg.append(reg_abs)

    y_pred_stgnn_reg = np.concatenate(preds_stgnn_reg, axis=0)
    err_stgnn = np.abs(y_test_reg_true.flatten() - y_pred_stgnn_reg.flatten())

    # 2. Entrenar Random Forest Espaciotemporal y obtener errores en el mismo bloque Test
    tensor_data = np.load(os.path.join(processed_dir, "tensor_eda_TNF.npy"))
    T = tensor_data.shape[0]
    train_end = int(T * config["model_params"]["train_split"])
    val_end = int(T * (config["model_params"]["train_split"] + config["model_params"]["val_split"]))

    X_list = [tensor_data[t - 1, :, :] for t in range(36, train_end)]
    y_list = [tensor_data[t, :, 0] for t in range(36, train_end)]
    X_train = np.concatenate(X_list, axis=0)
    y_train_reg = np.concatenate(y_list, axis=0)

    rf = RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, np.log1p(y_train_reg))

    # Alinear exactamente a las mismas últimas 184 semanas del Test del ST-GNN
    n_test_weeks = y_test_reg_true.shape[0]
    start_test_t = T - n_test_weeks
    rf_preds = [np.clip(np.expm1(rf.predict(tensor_data[t - 1, :, :])), 0, None) for t in range(start_test_t, T)]
    rf_trues = [tensor_data[t, :, 0] for t in range(start_test_t, T)]

    err_rf = np.abs(np.array(rf_trues).flatten() - np.array(rf_preds).flatten())

    # 3. Prueba de Wilcoxon (mismo protocolo de 10,000 pares que step06_baselines_comparison.py)
    idx_sub = np.random.choice(len(err_stgnn), min(10000, len(err_stgnn)), replace=False)
    stat, p_val = wilcoxon(err_stgnn[idx_sub], err_rf[idx_sub], alternative="less")
    decision = "Rechaza H0 (p < 0.05)" if p_val < 0.05 else "No Rechaza H0"

    # 4. Actualizar reports/tabla_wilcoxon_oe4.csv conservando los 3 baselines previos
    csv_path = os.path.join(reports_dir, "tabla_wilcoxon_oe4.csv")
    df_w = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame(
        columns=["Baseline", "Wilcoxon_W", "p_value", "Decision"]
    )
    df_w = df_w[df_w["Baseline"] != "Random Forest Espaciotemporal"]
    nueva_fila = pd.DataFrame([{
        "Baseline": "Random Forest Espaciotemporal",
        "Wilcoxon_W": stat,
        "p_value": p_val,
        "Decision": decision,
    }])
    df_w = pd.concat([df_w, nueva_fila], ignore_index=True)
    df_w.to_csv(csv_path, index=False)

    print("\n=== tabla_wilcoxon_oe4.csv actualizada ===")
    print(df_w.to_string(index=False))


if __name__ == "__main__":
    main()