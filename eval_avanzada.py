import os
import torch
import numpy as np
from scipy.ndimage import maximum_filter1d
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    f1_score, confusion_matrix, roc_curve, auc
)
from src.utils import load_config, load_pickle
from src.step03_dataset import get_dataloaders
from src.step04_model_stgcn import SpatioTemporalGNN

# 1. Función de tolerancia temporal (+/- 1 semana)
def compute_tolerant_metrics(y_true_mat, y_pred_mat, tolerance=1):
    y_true_exp = maximum_filter1d(y_true_mat, size=2 * tolerance + 1, axis=0, mode='nearest')
    y_pred_exp = maximum_filter1d(y_pred_mat, size=2 * tolerance + 1, axis=0, mode='nearest')

    tp = np.sum((y_pred_mat == 1) & (y_true_exp == 1))
    fp = np.sum((y_pred_mat == 1) & (y_true_exp == 0))
    fn = np.sum((y_true_mat == 1) & (y_pred_exp == 0))
    tn = np.sum((y_pred_mat == 0) & (y_true_mat == 0))

    prec = tp / (tp + fp + 1e-6)
    rec = tp / (tp + fn + 1e-6)
    f1_brote = 2 * (prec * rec) / (prec + rec + 1e-6)
    f1_norm = 2 * tn / (2 * tn + fp + fn + 1e-6)
    f1_macro = (f1_brote + f1_norm) / 2.0

    return f1_macro, f1_brote, prec, rec

def main():
    config = load_config()
    processed_dir = config["paths"]["processed_dir"]
    best_model_path = os.path.join(config["paths"]["models_dir"], "best_stgnn_model.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[EVALUACIÓN CLÍNICO-EPIDEMIOLÓGICA] Dispositivo: {device}")

    _, val_loader, test_loader, scalers = get_dataloaders()
    graph_data = torch.load(os.path.join(processed_dir, "graph_topology.pt"), weights_only=False)
    edge_index = graph_data["edge_index"].to(device)
    num_nodes = graph_data["num_nodes"]

    metadata = load_pickle(os.path.join(processed_dir, "metadata.pkl"))
    time_index = metadata["time_index"]
    ubigeos = metadata["ubigeos"]
    endemic_q3 = np.load(os.path.join(processed_dir, "endemic_channel_q3.npy"))
    tensor_data = np.load(os.path.join(processed_dir, "tensor_eda_TNF.npy"))
    stds_vec = np.std(tensor_data[:, :, 0], axis=0)

    # Macrorregiones INEI
    macro_def = {
        "COSTA": ["07", "11", "13", "14", "15", "18", "20", "23", "24"],
        "SELVA": ["01", "16", "17", "22", "25"]
    }
    node_regions = np.array([
        "COSTA" if u[:2] in macro_def["COSTA"] else ("SELVA" if u[:2] in macro_def["SELVA"] else "SIERRA")
        for u in ubigeos
    ])

    seq_len = config["model_params"]["seq_len"]
    hidden_dim = config["model_params"].get("hidden_dim", 64)
    num_features = tensor_data.shape[2]

    num_val = len(val_loader.dataset)
    num_test = len(test_loader.dataset)
    T_total = len(tensor_data)
    train_T = int(T_total * config["model_params"]["train_split"])
    val_T = int(T_total * config["model_params"]["val_split"])

    val_weeks = [time_index[train_T + seq_len + i][1] for i in range(num_val)]
    test_weeks = [time_index[train_T + val_T + seq_len + i][1] for i in range(num_test)]

    val_q3_flat = np.stack([endemic_q3[w - 1, :] for w in val_weeks], axis=0).flatten()
    test_q3_flat = np.stack([endemic_q3[w - 1, :] for w in test_weeks], axis=0).flatten()

    val_stds_flat = np.tile(stds_vec, num_val)
    test_stds_flat = np.tile(stds_vec, num_test)

    val_reg_flat = np.tile(node_regions, num_val)
    test_reg_flat = np.tile(node_regions, num_test)

    model = SpatioTemporalGNN(
        num_nodes=num_nodes, in_features=num_features, embed_dim=8, hidden_dim=hidden_dim, dropout=0.15
    ).to(device)
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()

    # 1. Inferencia sobre Validación
    val_probs, val_regs, val_trues = [], [], []
    mean_val, std_val = scalers["mean_target"], scalers["std_target"]

    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            for bx, _, by_cls in val_loader:
                bx = bx.to(device)
                pr_reg, pr_cls = model(bx, edge_index)
                pred_log = pr_reg.cpu().numpy() * std_val + mean_val
                val_regs.append(np.clip(np.expm1(pred_log), a_min=0, a_max=None))
                val_probs.append(torch.sigmoid(pr_cls).cpu().numpy())
                val_trues.append(by_cls.numpy().astype(int))

    y_v_true = np.concatenate(val_trues, axis=0).flatten()
    y_v_prob = np.concatenate(val_probs, axis=0).flatten()
    y_v_reg = np.concatenate(val_regs, axis=0).flatten()

    denom_v = np.where(val_stds_flat < 1e-3, 1.0, val_stds_flat)
    z_v = np.clip(-1.5 * ((y_v_reg - val_q3_flat) / denom_v), -30.0, 30.0)
    p_phy_v = 1.0 / (1.0 + np.exp(z_v))

    # Calibración optimizando F1-Brote (Clase 1)
    optimal_params = {}
    print("\n[CALIBRACIÓN DE PARÁMETROS REGIONALES]")
    for reg in ["COSTA", "SIERRA", "SELVA"]:
        m = (val_reg_flat == reg)
        yt_sub, yp_sub, yphy_sub = y_v_true[m], y_v_prob[m], p_phy_v[m]

        best_f1, best_w, best_c = 0.0, 0.40, 0.40
        for w in [0.15, 0.30, 0.40, 0.50, 0.60]:
            comb = w * yp_sub + (1.0 - w) * yphy_sub
            for c in np.linspace(0.15, 0.60, 46):
                f1_b = f1_score(yt_sub, (comb >= c).astype(int), average="binary", zero_division=0)
                if f1_b > best_f1:
                    best_f1, best_w, best_c = f1_b, w, c
        optimal_params[reg] = (best_w, best_c)
        print(f" -> {reg:<6}: Peso w = {best_w:.2f} | Corte c = {best_c:.2f} | F1-Brote (Val): {best_f1:.4f}")

    # 2. Inferencia sobre Test
    test_regs, test_trues_reg, test_probs, test_trues_cls = [], [], [], []
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            for bx, by_reg, by_cls in test_loader:
                bx = bx.to(device)
                pr_reg, pr_cls = model(bx, edge_index)
                pred_log = pr_reg.cpu().numpy() * std_val + mean_val
                target_log = by_reg.numpy() * std_val + mean_val

                test_regs.append(np.clip(np.expm1(pred_log), a_min=0, a_max=None))
                test_trues_reg.append(np.clip(np.expm1(target_log), a_min=0, a_max=None))
                test_probs.append(torch.sigmoid(pr_cls).cpu().numpy())
                test_trues_cls.append(by_cls.numpy().astype(int))

    y_t_reg = np.concatenate(test_trues_reg, axis=0).flatten()
    y_p_reg = np.concatenate(test_regs, axis=0).flatten()
    y_t_cls = np.concatenate(test_trues_cls, axis=0).flatten()
    y_p_cls = np.concatenate(test_probs, axis=0).flatten()

    denom_t = np.where(test_stds_flat < 1e-3, 1.0, test_stds_flat)
    z_t = np.clip(-1.5 * ((y_p_reg - test_q3_flat) / denom_t), -30.0, 30.0)
    p_phy_t = 1.0 / (1.0 + np.exp(z_t))

    final_score = np.zeros_like(y_p_cls)
    y_pred_cls = np.zeros_like(y_t_cls)

    for reg, (w_r, c_r) in optimal_params.items():
        m = (test_reg_flat == reg)
        final_score[m] = w_r * y_p_cls[m] + (1.0 - w_r) * p_phy_t[m]
        y_pred_cls[m] = (final_score[m] >= c_r).astype(int)

    # Métricas Continuas
    r2 = r2_score(y_t_reg, y_p_reg)
    mae = mean_absolute_error(y_t_reg, y_p_reg)
    rmse = np.sqrt(mean_squared_error(y_t_reg, y_p_reg))
    fpr, tpr, _ = roc_curve(y_t_cls, final_score)
    roc_auc = auc(fpr, tpr)

    # Métricas Discretas Puntuales (Semana Exacta)
    f1_macro_exact = f1_score(y_t_cls, y_pred_cls, average="macro")
    f1_brote_exact = f1_score(y_t_cls, y_pred_cls, average="binary")

    # Métricas con Tolerancia Temporal (+/- 1 Semana)
    y_true_mat = y_t_cls.reshape(num_test, num_nodes)
    y_pred_mat = y_pred_cls.reshape(num_test, num_nodes)
    f1_m_tol, f1_b_tol, prec_tol, rec_tol = compute_tolerant_metrics(y_true_mat, y_pred_mat, tolerance=1)

    print("\n" + "=" * 75)
    print(" REPORTE EPIDEMIOLÓGICO CONSOLIDADO: TEST SET (2021 - 2024)")
    print("=" * 75)
    print(f" -> Modelo Espaciotemporal : ST-GNN Provincial (196 Nodos, 17 Variables)")
    print(f" -> Regresión Continua    : R² = {r2:.4f} | MAE = {mae:.2f} casos | RMSE = {rmse:.2f}")
    print(f" -> Capacidad Global (AUC) : {roc_auc:.4f}")
    print("-" * 75)
    print(f" [EVALUACIÓN PUNTUAL EXACTA]  (Semana s = Semana Real)")
    print(f"    * F1-Macro             : {f1_macro_exact:.4f}")
    print(f"    * F1-Brote (Clase 1)   : {f1_brote_exact:.4f}")
    print("-" * 75)
    print(f" [VIGILANCIA EPIDEMIOLÓGICA] (Ventana Operativa +/- 1 Semana)")
    print(f"    * F1-Macro Tolerante   : {f1_m_tol:.4f}")
    print(f"    * F1-Brote Tolerante   : {f1_b_tol:.4f}")
    print(f"    * Precisión Operativa  : {prec_tol * 100:.2f}%")
    print(f"    * Sensibilidad (Recall): {rec_tol * 100:.2f}%")
    print("=" * 75)

    print("\n Desglose Regional con Tolerancia Temporal (+/- 1 sem):")
    for reg in ["COSTA", "SIERRA", "SELVA"]:
        p_mask = (node_regions == reg)
        m_macro, m_brote, m_prec, m_rec = compute_tolerant_metrics(
            y_true_mat[:, p_mask], y_pred_mat[:, p_mask], tolerance=1
        )
        print(f"    * {reg:<6}: F1-Macro = {m_macro:.4f} | F1-Brote = {m_brote:.4f} | Recall = {m_rec * 100:.1f}%")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    main()