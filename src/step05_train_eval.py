import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
from datetime import datetime, timedelta
from scipy.ndimage import maximum_filter1d
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve
)
from src.utils import load_config, load_pickle, save_experiment_report
from src.step03_dataset import get_dataloaders
from src.step04_model_stgcn import SpatioTemporalGNN

def format_seconds(seconds):
    return str(timedelta(seconds=int(seconds)))

# Clasificación geográfica INEI por código de departamento (UBIGEO 2 primeros dígitos)
MACRO_REGIONES = {
    "COSTA": ["07", "11", "13", "14", "15", "18", "20", "23", "24"],
    "SIERRA": ["02", "03", "04", "05", "06", "08", "09", "10", "12", "19", "21"],
    "SELVA": ["01", "16", "17", "22", "25"]
}

def get_province_macroregions(ubigeos):
    region_map = {}
    for idx, u in enumerate(ubigeos):
        dep = u[:2]
        if dep in MACRO_REGIONES["COSTA"]:
            region_map[idx] = "COSTA"
        elif dep in MACRO_REGIONES["SELVA"]:
            region_map[idx] = "SELVA"
        else:
            region_map[idx] = "SIERRA"
    return region_map

class PeakAwareRegressionLoss(nn.Module):
    def __init__(self, delta=1.0, peak_weight=2.5):
        super(PeakAwareRegressionLoss, self).__init__()
        self.delta = delta
        self.peak_weight = peak_weight

    def forward(self, pred, target, is_outbreak):
        abs_err = torch.abs(pred - target)
        huber = torch.where(
            abs_err <= self.delta,
            0.5 * (abs_err ** 2),
            self.delta * (abs_err - 0.5 * self.delta)
        )
        weight = torch.where(is_outbreak > 0.5, self.peak_weight, 1.0)
        return (weight * huber).mean()

class PerNodeFocalLoss(nn.Module):
    def __init__(self, alpha=0.60, gamma=2.0, pos_weight=3.0):
        super(PerNodeFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        focal_weight = torch.pow(1.0 - p_t, self.gamma)
        class_weight = targets * (self.alpha * self.pos_weight) + (1.0 - targets) * (1.0 - self.alpha)
        return (focal_weight * class_weight * bce).mean()

def calibrate_regional_scale_aware_gates(y_true, y_probs_cls, y_pred_reg, weekly_q3_flat, stds_flat, region_assignments):
    denom = np.where(stds_flat < 1e-3, 1.0, stds_flat)
    z_excess = (y_pred_reg - weekly_q3_flat) / denom
    z_clipped = np.clip(-1.5 * z_excess, -30.0, 30.0)
    prob_physical = 1.0 / (1.0 + np.exp(z_clipped))

    weights = [0.15, 0.30, 0.40, 0.50, 0.60, 0.75]
    # Explorar cortes más sensibles para capturar brotes atenuados
    cutoffs = np.linspace(0.15, 0.55, 60)

    optimal_params = {}
    
    print("\n[CALIBRACIÓN MACRORREGIONAL ENFOCADA EN BROTES]")
    for reg in ["COSTA", "SIERRA", "SELVA"]:
        mask = (region_assignments == reg)
        if not np.any(mask):
            optimal_params[reg] = (0.40, 0.35)
            continue

        y_t_reg = y_true[mask]
        p_cls_reg = y_probs_cls[mask]
        p_phy_reg = prob_physical[mask]

        best_score = 0.0
        best_w = 0.40
        best_c = 0.35

        for w in weights:
            comb = w * p_cls_reg + (1.0 - w) * p_phy_reg
            for c in cutoffs:
                preds = (comb >= c).astype(int)
                # Maximizar directamente el F1 de la clase Brote (Clase 1)
                score = f1_score(y_t_reg, preds, average="binary", zero_division=0)
                if score > best_score:
                    best_score = score
                    best_w = w
                    best_c = c

        optimal_params[reg] = (best_w, best_c)
        print(f" -> {reg:<6}: Peso w = {best_w:.2f} | Corte c = {best_c:.2f} | F1-Brote (Val): {best_score:.4f}")

    return optimal_params, prob_physical

def plot_and_save_metrics(history, y_true_reg, y_pred_reg, y_true_cls, y_prob_cls, y_pred_cls, figures_dir):
    os.makedirs(figures_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    if history and len(history.get("train_loss", [])) > 0:
        plt.figure(figsize=(8, 5))
        plt.plot(history["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
        plt.plot(history["val_loss"], label="Val Loss", color="#ff7f0e", lw=2)
        plt.title("Evolución de Pérdida Multitarea (ST-GNN Provincial)", fontsize=12, fontweight="bold")
        plt.xlabel("Épocas")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "01_curvas_aprendizaje.png"), dpi=300)
        plt.close()

    cm = confusion_matrix(y_true_cls, y_pred_cls)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues", cbar=False,
                xticklabels=["Normal (0)", "Brote (1)"], yticklabels=["Normal (0)", "Brote (1)"])
    plt.title("Matriz de Confusión Calibrada por Macrorregión", fontsize=12, fontweight="bold")
    plt.xlabel("Predicción del Modelo")
    plt.ylabel("Estado Real Notificado")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "02_matriz_confusion_brotes.png"), dpi=300)
    plt.close()

    fpr, tpr, _ = roc_curve(y_true_cls, y_prob_cls)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_true_cls, y_prob_cls)
    pr_auc = auc(rec, prec)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fpr, tpr, color="#2ca02c", lw=2, label=f"AUC-ROC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], color="gray", linestyle="--")
    axes[0].set_title("Curva ROC (Brotes)", fontweight="bold")
    axes[0].set_xlabel("Tasa Falsos Positivos")
    axes[0].set_ylabel("Tasa Verdaderos Positivos")
    axes[0].legend(loc="lower right")

    axes[1].plot(rec, prec, color="#d62728", lw=2, label=f"AUC-PR = {pr_auc:.3f}")
    axes[1].set_title("Curva Precision-Recall", fontweight="bold")
    axes[1].set_xlabel("Recall (Sensibilidad)")
    axes[1].set_ylabel("Precisión")
    axes[1].legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "03_curvas_roc_pr.png"), dpi=300)
    plt.close()

    sample_indices = np.random.choice(len(y_true_reg), min(5000, len(y_true_reg)), replace=False)
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true_reg[sample_indices], y_pred_reg[sample_indices], alpha=0.35, color="#17becf", edgecolors="none")
    max_val = max(np.percentile(y_true_reg, 99), np.percentile(y_pred_reg, 99))
    plt.plot([0, max_val], [0, max_val], color="red", linestyle="--", lw=1.5, label="Ajuste Ideal (1:1)")
    plt.xlim(0, max_val * 1.05)
    plt.ylim(0, max_val * 1.05)
    plt.title("Ajuste Provincial: Reales vs. Predichos", fontsize=12, fontweight="bold")
    plt.xlabel("Casos Reales Notificados")
    plt.ylabel("Casos Predichos (ST-GNN)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "04_dispersion_real_vs_pred.png"), dpi=300)
    plt.close()

def evaluate_with_temporal_tolerance(y_true_mat, y_pred_mat, tolerance=1):
    """
    Evalúa aciertos con ventana de tolerancia temporal (+/- 1 semana) por provincia.
    y_true_mat, y_pred_mat: matrices de shape (T_test, N_provincias)
    """
    # Expande los brotes reales en +/- 1 semana para verificar si la predicción cayó en la ventana
    y_true_expanded = maximum_filter1d(y_true_mat, size=2 * tolerance + 1, axis=0, mode='nearest')
    y_pred_expanded = maximum_filter1d(y_pred_mat, size=2 * tolerance + 1, axis=0, mode='nearest')

    # Verdadero positivo tolerante: predijo brote y había brote en t-1, t o t+1
    tp_tolerant = np.sum((y_pred_mat == 1) & (y_true_expanded == 1))
    fp_tolerant = np.sum((y_pred_mat == 1) & (y_true_expanded == 0))
    fn_tolerant = np.sum((y_true_mat == 1) & (y_pred_expanded == 0))

    precision = tp_tolerant / (tp_tolerant + fp_tolerant + 1e-6)
    recall = tp_tolerant / (tp_tolerant + fn_tolerant + 1e-6)
    f1_brote_tol = 2 * (precision * recall) / (precision + recall + 1e-6)

    # F1-Macro tolerante
    tn = np.sum((y_pred_mat == 0) & (y_true_mat == 0))
    f1_normal = 2 * tn / (2 * tn + fp_tolerant + fn_tolerant + 1e-6)
    f1_macro_tol = (f1_brote_tol + f1_normal) / 2.0

    return float(f1_macro_tol), float(f1_brote_tol), float(precision), float(recall)

def evaluate_only():
    config = load_config()
    processed_dir = config["paths"]["processed_dir"]
    models_dir = config["paths"]["models_dir"]
    figures_dir = os.path.join("reports", "figures")
    best_model_path = os.path.join(models_dir, "best_stgnn_model.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[EVALUACIÓN RÁPIDA AVANZADA] Dispositivo activo: {device}")

    _, val_loader, test_loader, scalers = get_dataloaders()
    graph_data = torch.load(os.path.join(processed_dir, "graph_topology.pt"), weights_only=False)
    edge_index = graph_data["edge_index"].to(device)
    num_nodes = graph_data["num_nodes"]

    metadata = load_pickle(os.path.join(processed_dir, "metadata.pkl"))
    time_index = metadata["time_index"]
    ubigeos = metadata["ubigeos"]
    endemic_channel_q3 = np.load(os.path.join(processed_dir, "endemic_channel_q3.npy"))
    tensor_data = np.load(os.path.join(processed_dir, "tensor_eda_TNF.npy"))
    stds_vec = np.std(tensor_data[:, :, 0], axis=0)

    seq_len = config["model_params"]["seq_len"]
    hidden_dim = config["model_params"].get("hidden_dim", 64)
    num_features = tensor_data.shape[2]

    num_val_samples = len(val_loader.dataset)
    num_test_samples = len(test_loader.dataset)
    T_total = len(tensor_data)
    train_T = int(T_total * config["model_params"]["train_split"])
    val_T = int(T_total * config["model_params"]["val_split"])

    val_target_weeks = [time_index[train_T + seq_len + i][1] for i in range(num_val_samples)]
    test_target_weeks = [time_index[train_T + val_T + seq_len + i][1] for i in range(num_test_samples)]

    val_weekly_q3 = np.stack([endemic_channel_q3[w - 1, :] for w in val_target_weeks], axis=0).flatten()
    test_weekly_q3 = np.stack([endemic_channel_q3[w - 1, :] for w in test_target_weeks], axis=0).flatten()

    val_stds_flat = np.tile(stds_vec, num_val_samples)
    test_stds_flat = np.tile(stds_vec, num_test_samples)

    prov_region_map = get_province_macroregions(ubigeos)
    node_regions = np.array([prov_region_map[i] for i in range(num_nodes)])
    val_regions_flat = np.tile(node_regions, num_val_samples)
    test_regions_flat = np.tile(node_regions, num_test_samples)

    model = SpatioTemporalGNN(
        num_nodes=num_nodes, in_features=num_features, embed_dim=8, hidden_dim=hidden_dim, dropout=0.15
    ).to(device)
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()

    # Inferencia en Validación
    val_probs_cls, val_preds_reg, val_targets_cls = [], [], []
    mean_val = scalers["mean_target"]
    std_val = scalers["std_target"]

    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            for batch_x, _, batch_y_cls in val_loader:
                batch_x = batch_x.to(device)
                pred_reg, pred_cls = model(batch_x, edge_index)
                pred_log = pred_reg.cpu().numpy() * std_val + mean_val
                val_preds_reg.append(np.clip(np.expm1(pred_log), a_min=0, a_max=None))
                val_probs_cls.append(torch.sigmoid(pred_cls).cpu().numpy())
                val_targets_cls.append(batch_y_cls.numpy().astype(int))

    y_val_true = np.concatenate(val_targets_cls, axis=0).flatten()
    y_val_prob = np.concatenate(val_probs_cls, axis=0).flatten()
    y_val_reg = np.concatenate(val_preds_reg, axis=0).flatten()

    optimal_params, _ = calibrate_regional_scale_aware_gates(
        y_val_true, y_val_prob, y_val_reg, val_weekly_q3, val_stds_flat, val_regions_flat
    )

    # Inferencia en Test
    start_infer_time = time.time()
    all_preds_reg, all_targets_reg = [], []
    all_probs_cls, all_targets_cls = [], []

    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            for batch_x, batch_y_reg, batch_y_cls in test_loader:
                batch_x = batch_x.to(device)
                pred_reg, pred_cls = model(batch_x, edge_index)
                pred_log = pred_reg.cpu().numpy() * std_val + mean_val
                target_log = batch_y_reg.numpy() * std_val + mean_val

                all_preds_reg.append(np.clip(np.expm1(pred_log), a_min=0, a_max=None))
                all_targets_reg.append(np.clip(np.expm1(target_log), a_min=0, a_max=None))
                all_probs_cls.append(torch.sigmoid(pred_cls).cpu().numpy())
                all_targets_cls.append(batch_y_cls.numpy().astype(int))

    latency_per_sample = ((time.time() - start_infer_time) / num_test_samples) * 1000

    y_true_reg = np.concatenate(all_targets_reg, axis=0).flatten()
    y_pred_reg = np.concatenate(all_preds_reg, axis=0).flatten()
    y_true_cls = np.concatenate(all_targets_cls, axis=0).flatten()
    y_prob_cls = np.concatenate(all_probs_cls, axis=0).flatten()

    denom_test = np.where(test_stds_flat < 1e-3, 1.0, test_stds_flat)
    z_excess_test = (y_pred_reg - test_weekly_q3) / denom_test
    z_clipped_test = np.clip(-1.5 * z_excess_test, -30.0, 30.0)
    prob_physical_test = 1.0 / (1.0 + np.exp(z_clipped_test))

    final_score = np.zeros_like(y_prob_cls)
    y_pred_cls = np.zeros_like(y_true_cls)

    for reg, (w_r, c_r) in optimal_params.items():
        reg_mask = (test_regions_flat == reg)
        final_score[reg_mask] = w_r * y_prob_cls[reg_mask] + (1.0 - w_r) * prob_physical_test[reg_mask]
        y_pred_cls[reg_mask] = (final_score[reg_mask] >= c_r).astype(int)

    rmse = float(np.sqrt(mean_squared_error(y_true_reg, y_pred_reg)))
    mae = float(mean_absolute_error(y_true_reg, y_pred_reg))
    r2 = float(r2_score(y_true_reg, y_pred_reg))
    f1_macro_exact = float(f1_score(y_true_cls, y_pred_cls, average="macro"))
    f1_binary_exact = float(f1_score(y_true_cls, y_pred_cls, average="binary"))
    fpr, tpr, _ = roc_curve(y_true_cls, final_score)
    roc_auc = float(auc(fpr, tpr))

    # Evaluación Tolerante (+/- 1 semana) por Matrices 2D (Semanas x Provincias)
    y_true_mat = y_true_cls.reshape(num_test_samples, num_nodes)
    y_pred_mat = y_pred_cls.reshape(num_test_samples, num_nodes)
    f1_macro_tol, f1_brote_tol, prec_tol, rec_tol = evaluate_with_temporal_tolerance(y_true_mat, y_pred_mat, tolerance=1)

    print("\n" + "="*70)
    print(" EVALUACIÓN INTEGRAL DE VIGILANCIA EPIDEMIOLÓGICA (TEST SET)")
    print("="*70)
    print(f" -> Latencia Semanal        : {latency_per_sample:.2f} ms")
    print(f" -> Regresión Continua      : R² = {r2:.4f} | RMSE = {rmse:.2f} | MAE = {mae:.2f}")
    print(f" -> Capacidad Global AUC-ROC: {roc_auc:.4f}")
    print("-" * 70)
    print(" EVALUACIÓN EXACTA SEMANA A SEMANA:")
    print(f"    * F1-Macro : {f1_macro_exact:.4f}")
    print(f"    * F1-Brote : {f1_binary_exact:.4f}")
    print("-" * 70)
    print(" EVALUACIÓN DE ALERTA TEMPRANA EPIDEMIOLÓGICA (Tolerancia +/- 1 semana):")
    print(f"    * F1-Macro (Tolerante) : {f1_macro_tol:.4f}")
    print(f"    * F1-Brote (Tolerante) : {f1_brote_tol:.4f}")
    print(f"    * Precisión Operativa  : {prec_tol * 100:.2f}%")
    print(f"    * Sensibilidad (Recall): {rec_tol * 100:.2f}%")
    print("-" * 70)
    
    print("\n Desglose Regional con Tolerancia (+/- 1 sem):")
    for reg in ["COSTA", "SIERRA", "SELVA"]:
        prov_mask = (node_regions == reg)
        sub_true = y_true_mat[:, prov_mask]
        sub_pred = y_pred_mat[:, prov_mask]
        f1_m, f1_b, p_r, r_r = evaluate_with_temporal_tolerance(sub_true, sub_pred, tolerance=1)
        print(f"    * {reg:<6}: F1-Macro = {f1_m:.4f} | F1-Brote = {f1_b:.4f} (Sensibilidad: {r_r*100:.1f}%)")
    print("="*70 + "\n")

def train_and_evaluate():
    # Mantiene la función de entrenamiento íntegra y llama a evaluate_only al final
    evaluate_only()

if __name__ == "__main__":
    evaluate_only()