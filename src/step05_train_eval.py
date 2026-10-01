import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import random
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import maximum_filter1d
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve
)
from src.utils import load_config, load_pickle, save_experiment_report
from src.step03_dataset import get_dataloaders
from src.step04_model_stgcn import SpatioTemporalGNN

# 0. Semilla Global de Reproducibilidad Científica
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

def format_seconds(seconds):
    return str(timedelta(seconds=int(seconds)))

# 1. Clasificación geográfica INEI por código departamental (UBIGEO 2 primeros dígitos)
MACRO_REGIONES = {
    "COSTA": ["07", "11", "13", "14", "15", "18", "20", "23", "24"],
    "SELVA": ["01", "16", "17", "22", "25"]
}

def get_province_macroregions(ubigeos):
    region_map = {}
    for idx, u in enumerate(ubigeos):
        dep = str(u)[:2]
        if dep in MACRO_REGIONES["COSTA"]:
            region_map[idx] = "COSTA"
        elif dep in MACRO_REGIONES["SELVA"]:
            region_map[idx] = "SELVA"
        else:
            region_map[idx] = "SIERRA"
    return region_map

# 2. Funciones de Pérdida Multitarea
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

# 3.A. Función Modular Unificada: Compuerta Física con Supresión Clínica (< 5 casos)
def compute_physical_probability(y_pred_reg, weekly_q3_flat, stds_flat):
    denom = np.where(stds_flat < 1e-3, 1.0, stds_flat)
    z_excess = (y_pred_reg - weekly_q3_flat) / denom
    # Condición técnica MINSA: suprime alertas espurias si la predicción es menor a 5 casos
    z_excess = np.where(y_pred_reg < 5.0, -10.0, z_excess)
    z_clipped = np.clip(-1.5 * z_excess, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(z_clipped))

# 3.B. Calibración Macrorregional de Compuertas
def calibrate_regional_scale_aware_gates(y_true, y_probs_cls, y_pred_reg, weekly_q3_flat, stds_flat, region_assignments):
    prob_physical = compute_physical_probability(y_pred_reg, weekly_q3_flat, stds_flat)

    weights = [0.15, 0.30, 0.40, 0.50, 0.60, 0.75]
    cutoffs = np.linspace(0.15, 0.70, 56)

    optimal_params = {}
    print("\n[CALIBRACIÓN MACRORREGIONAL ENFOCADA EN BROTES]")
    for reg in ["COSTA", "SIERRA", "SELVA"]:
        mask = (region_assignments == reg)
        if not np.any(mask):
            optimal_params[reg] = (0.40, 0.44)
            continue

        y_t_reg = y_true[mask]
        p_cls_reg = y_probs_cls[mask]
        p_phy_reg = prob_physical[mask]

        best_score = 0.0
        best_w = 0.40
        best_c = 0.44

        for w in weights:
            comb = w * p_cls_reg + (1.0 - w) * p_phy_reg
            for c in cutoffs:
                preds = (comb >= c).astype(int)
                score = f1_score(y_t_reg, preds, average="binary", zero_division=0)
                if score > best_score:
                    best_score = score
                    best_w = w
                    best_c = c

        optimal_params[reg] = (best_w, best_c)
        print(f" -> {reg:<6}: Peso w = {best_w:.2f} | Corte c = {best_c:.2f} | F1-Brote (Val): {best_score:.4f}")

    return optimal_params, prob_physical

# 4. Evaluación Epidemiológica con Tolerancia Temporal (+/- 1 semana)
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

    return f1_macro, f1_brote, prec, rec, tp, fp, fn, tn

# 5. Generación de Gráficos Analíticos (Figuras 01 a 06)
def plot_and_save_all_figures(
    history, y_true_reg, y_pred_reg, y_true_cls, final_score, y_pred_exact,
    y_true_mat, y_pred_mat, node_regions, metadata, test_target_weeks,
    endemic_channel_q3, figures_dir
):
    os.makedirs(figures_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # FIGURA 01: Curvas de Aprendizaje
    if history and len(history.get("train_loss", [])) > 0:
        plt.figure(figsize=(8, 5))
        plt.plot(history["train_loss"], label="Perdida Entrenamiento (Train Loss)", color="#1f77b4", lw=2)
        plt.plot(history["val_loss"], label="Perdida Validacion (Val Loss)", color="#ff7f0e", lw=2)
        plt.title("Evolucion de la Perdida Multitarea (ST-GNN Provincial)", fontsize=12, fontweight="bold")
        plt.xlabel("Epocas", fontsize=11)
        plt.ylabel("Loss Multitarea", fontsize=11)
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "01_curvas_aprendizaje.png"), dpi=300)
        plt.close()

    # FIGURA 02: Matriz de Confusión Comparativa
    cm_exact = confusion_matrix(y_true_cls, y_pred_exact)
    cm_exact_norm = cm_exact.astype('float') / (cm_exact.sum(axis=1)[:, np.newaxis] + 1e-6)

    _, _, _, _, tp_tol, fp_tol, fn_tol, tn_tol = compute_tolerant_metrics(y_true_mat, y_pred_mat, tolerance=1)
    cm_tol = np.array([[tn_tol, fp_tol], [fn_tol, tp_tol]])
    cm_tol_norm = cm_tol.astype('float') / (cm_tol.sum(axis=1)[:, np.newaxis] + 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.heatmap(cm_exact_norm, annot=True, fmt=".2%", cmap="Blues", cbar=False, ax=axes[0],
                xticklabels=["Normal (0)", "Brote (1)"], yticklabels=["Normal (0)", "Brote (1)"],
                annot_kws={"size": 13, "weight": "bold"})
    axes[0].set_title("A) Evaluacion Puntual Rigida\n(Semana Exacta)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Prediccion del Modelo", fontsize=11)
    axes[0].set_ylabel("Estado Real Notificado", fontsize=11)

    sns.heatmap(cm_tol_norm, annot=True, fmt=".2%", cmap="Greens", cbar=False, ax=axes[1],
                xticklabels=["Normal (0)", "Brote (1)"], yticklabels=["Normal (0)", "Brote (1)"],
                annot_kws={"size": 13, "weight": "bold"})
    axes[1].set_title("B) Vigilancia Epidemiologica Operativa\n(Ventana con Tolerancia +/- 1 semana)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Prediccion del Modelo", fontsize=11)
    axes[1].set_ylabel("Estado Real Notificado", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "02_matriz_confusion_comparativa.png"), dpi=300)
    plt.close()

    # FIGURA 03: Curvas ROC y Precision-Recall
    fpr, tpr, _ = roc_curve(y_true_cls, final_score)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_true_cls, final_score)
    pr_auc = auc(rec, prec)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fpr, tpr, color="#2ca02c", lw=2.5, label=f"AUC-ROC = {roc_auc:.4f}")
    axes[0].plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5)
    axes[0].set_title("Curva ROC (Deteccion de Brotes)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Tasa de Falsos Positivos (1 - Especificidad)", fontsize=11)
    axes[0].set_ylabel("Tasa de Verdaderos Positivos (Sensibilidad)", fontsize=11)
    axes[0].legend(loc="lower right", frameon=True)

    axes[1].plot(rec, prec, color="#d62728", lw=2.5, label=f"AUC-PR = {pr_auc:.4f}")
    axes[1].set_title("Curva Precision-Recall", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Sensibilidad (Recall)", fontsize=11)
    axes[1].set_ylabel("Precision", fontsize=11)
    axes[1].legend(loc="lower left", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "03_curvas_roc_pr.png"), dpi=300)
    plt.close()

    # FIGURA 04: Dispersión Casos Reales vs. Predichos
    sample_idx = np.random.choice(len(y_true_reg), min(6000, len(y_true_reg)), replace=False)
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true_reg[sample_idx], y_pred_reg[sample_idx], alpha=0.35, color="#17becf", edgecolors="none")
    max_val = max(np.percentile(y_true_reg, 99.5), np.percentile(y_pred_reg, 99.5))
    plt.plot([0, max_val], [0, max_val], color="red", linestyle="--", lw=2, label="Ajuste Teorico Ideal (1:1)")
    plt.xlim(0, max_val * 1.05)
    plt.ylim(0, max_val * 1.05)
    r2_val = r2_score(y_true_reg, y_pred_reg)
    mae_val = mean_absolute_error(y_true_reg, y_pred_reg)
    rmse_val = np.sqrt(mean_squared_error(y_true_reg, y_pred_reg))
    plt.title(f"Ajuste Continuo: Reales vs. Predichos\n(R2: {r2_val:.4f} | MAE: {mae_val:.2f} | RMSE: {rmse_val:.2f})",
              fontsize=12, fontweight="bold")
    plt.xlabel("Casos Reales Notificados (MINSA)", fontsize=11)
    plt.ylabel("Casos Predichos (ST-GNN)", fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "04_dispersion_real_vs_pred.png"), dpi=300)
    plt.close()

    # FIGURA 05: Desempeño Comparativo Macrorregional
    regions = ["COSTA", "SIERRA", "SELVA"]
    f1_exact_macro, f1_tol_macro = [], []
    f1_exact_brote, f1_tol_brote = [], []

    for reg in regions:
        p_mask = (node_regions == reg)
        sub_true = y_true_mat[:, p_mask]
        sub_pred = y_pred_mat[:, p_mask]
        
        f1_em = f1_score(sub_true.flatten(), sub_pred.flatten(), average="macro", zero_division=0)
        f1_eb = f1_score(sub_true.flatten(), sub_pred.flatten(), average="binary", zero_division=0)
        f1_tm, f1_tb, _, _, _, _, _, _ = compute_tolerant_metrics(sub_true, sub_pred, tolerance=1)
        
        f1_exact_macro.append(f1_em)
        f1_tol_macro.append(f1_tm)
        f1_exact_brote.append(f1_eb)
        f1_tol_brote.append(f1_tb)

    x = np.arange(len(regions))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(x - width/2, f1_exact_macro, width, label="Semana Exacta", color="#4a7bb7")
    axes[0].bar(x + width/2, f1_tol_macro, width, label="Tolerancia +/- 1 sem", color="#2ca02c")
    axes[0].set_title("A) F1-Score Macro por Macrorregion", fontsize=12, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(regions, fontsize=11, fontweight="bold")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(frameon=True)

    axes[1].bar(x - width/2, f1_exact_brote, width, label="Semana Exacta", color="#e76f51")
    axes[1].bar(x + width/2, f1_tol_brote, width, label="Tolerancia +/- 1 sem", color="#2a9d8f")
    axes[1].set_title("B) F1-Score Brote (Clase 1) por Macrorregion", fontsize=12, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(regions, fontsize=11, fontweight="bold")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "05_desempeno_regional_comparativo.png"), dpi=300)
    plt.close()

    # FIGURA 06: Serie Temporal Longitudinal de Seguimiento
    prov_samples = {}
    catalog = metadata.get("provinces_catalog", [])
    for reg in regions:
        p_indices = np.where(node_regions == reg)[0]
        casos_prom = np.mean(y_true_mat[:, p_indices], axis=0)
        best_p_idx = p_indices[np.argmax(casos_prom)]
        
        name = f"Provincia {best_p_idx}"
        if catalog and best_p_idx < len(catalog):
            p_dict = catalog[best_p_idx]
            p_prov = str(p_dict.get('provincia', name)).encode('ascii', 'ignore').decode('ascii')
            p_dep = str(p_dict.get('departamento', '')).encode('ascii', 'ignore').decode('ascii')
            name = f"{p_prov} ({p_dep})"
        prov_samples[reg] = (best_p_idx, name)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    semanas_x = np.arange(1, len(test_target_weeks) + 1)

    for i, reg in enumerate(regions):
        p_idx, p_name = prov_samples[reg]
        real_cases = y_true_mat[:, p_idx]
        pred_cases = y_pred_reg.reshape(len(test_target_weeks), -1)[:, p_idx]
        q3_prov = [endemic_channel_q3[w - 1, p_idx] for w in test_target_weeks]
        brotes_pred = np.where(y_pred_mat[:, p_idx] == 1)[0]

        axes[i].plot(semanas_x, real_cases, label="Casos Reales", color="#1f77b4", lw=2)
        axes[i].plot(semanas_x, pred_cases, label="Prediccion ST-GNN", color="#ff7f0e", lw=1.8, linestyle="--")
        axes[i].plot(semanas_x, q3_prov, label="Umbral Endemico (Q3)", color="#d62728", lw=1.2, linestyle=":")
        
        if len(brotes_pred) > 0:
            axes[i].scatter(semanas_x[brotes_pred], real_cases[brotes_pred], color="red", marker="^", s=60,
                            label="Alerta de Brote Emitida", zorder=5)

        axes[i].set_title(f"Macrorregion {reg}: {p_name}", fontsize=11, fontweight="bold")
        axes[i].set_ylabel("Casos Semanales", fontsize=10)
        axes[i].legend(loc="upper right", frameon=True, fontsize=9)

    axes[2].set_xlabel("Semanas Continuas del Periodo de Prueba (Test Set 2021 - 2024)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "06_serie_temporal_seguimiento_brotes.png"), dpi=300)
    plt.close()

# 6. Modo Inferencia / Evaluación Rápida
def evaluate_only(history=None):
    config = load_config()
    processed_dir = config["paths"]["processed_dir"]
    models_dir = config["paths"]["models_dir"]
    figures_dir = os.path.join("reports", "figures")
    best_model_path = os.path.join(models_dir, "best_stgnn_model.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[EVALUACIÓN RÁPIDA CON VISUALIZACIÓN INTEGRAL] Dispositivo: {device}")

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

    # Inferencia en Prueba (Test Set)
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

    # Corrección clave: uso directo de la función unificada en Test
    prob_physical_test = compute_physical_probability(y_pred_reg, test_weekly_q3, test_stds_flat)

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

    y_true_mat = y_true_cls.reshape(num_test_samples, num_nodes)
    y_pred_mat = y_pred_cls.reshape(num_test_samples, num_nodes)
    f1_macro_tol, f1_brote_tol, prec_tol, rec_tol, _, _, _, _ = compute_tolerant_metrics(y_true_mat, y_pred_mat, tolerance=1)

    print("\n[GENERANDO SUITE COMPLETA DE FIGURAS ANALÍTICAS (01 a 06)...]")
    plot_and_save_all_figures(
        history, y_true_reg, y_pred_reg, y_true_cls, final_score, y_pred_cls,
        y_true_mat, y_pred_mat, node_regions, metadata, test_target_weeks,
        endemic_channel_q3, figures_dir
    )

    print("\n" + "="*70)
    print(" REPORTE CONSOLIDADO: TEST SET (2021 - 2024)")
    print("="*70)
    print(f" -> Latencia Semanal        : {latency_per_sample:.2f} ms")
    print(f" -> Regresion Continua      : R2 = {r2:.4f} | RMSE = {rmse:.2f} | MAE = {mae:.2f}")
    print(f" -> Capacidad Global (AUC) : {roc_auc:.4f}")
    print("-" * 70)
    print(f" [EVALUACION PUNTUAL EXACTA]  : F1-Macro = {f1_macro_exact:.4f} | F1-Brote = {f1_binary_exact:.4f}")
    print(f" [VIGILANCIA OPERATIVA (+/-1s)] : F1-Macro = {f1_macro_tol:.4f} | F1-Brote = {f1_brote_tol:.4f}")
    print(f"                               Precision = {prec_tol*100:.2f}% | Sensibilidad = {rec_tol*100:.2f}%")
    print("-" * 70)
    print(" Desglose Regional con Tolerancia (+/- 1 sem):")
    for reg in ["COSTA", "SIERRA", "SELVA"]:
        p_mask = (node_regions == reg)
        m_macro, m_brote, _, m_rec, _, _, _, _ = compute_tolerant_metrics(y_true_mat[:, p_mask], y_pred_mat[:, p_mask], tolerance=1)
        print(f"    * {reg:<6}: F1-Macro = {m_macro:.4f} | F1-Brote = {m_brote:.4f} | Recall = {m_rec*100:.1f}%")
    print("="*70 + "\n")

# 7. Ciclo de Entrenamiento Completo
def train_and_evaluate():
    config = load_config()
    processed_dir = config["paths"]["processed_dir"]
    models_dir = config["paths"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[ENTRENAMIENTO PROVINCIAL] Dispositivo: {device}")

    train_loader, val_loader, _, scalers = get_dataloaders()
    graph_data = torch.load(os.path.join(processed_dir, "graph_topology.pt"), weights_only=False)
    edge_index = graph_data["edge_index"].to(device)
    num_nodes = graph_data["num_nodes"]

    hidden_dim = config["model_params"].get("hidden_dim", 64)
    accum_steps = 2
    embed_dim = 8
    tensor_data = np.load(os.path.join(processed_dir, "tensor_eda_TNF.npy"))
    num_features = tensor_data.shape[2]

    model = SpatioTemporalGNN(
        num_nodes=num_nodes, in_features=num_features, embed_dim=embed_dim, hidden_dim=hidden_dim, dropout=0.15
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["model_params"]["learning_rate"], weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=1, eta_min=1e-5)
    scaler_amp = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    criterion_reg = PeakAwareRegressionLoss(delta=1.0, peak_weight=2.5)
    criterion_cls = PerNodeFocalLoss(alpha=0.60, gamma=2.0, pos_weight=float(scalers["pos_weight"]))

    epochs = config["model_params"]["epochs"]
    best_val_loss = float("inf")
    patience = 20
    patience_counter = 0
    best_model_path = os.path.join(models_dir, "best_stgnn_model.pt")

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, (batch_x, batch_y_reg, batch_y_cls) in enumerate(train_loader):
            batch_x, batch_y_reg, batch_y_cls = batch_x.to(device), batch_y_reg.to(device), batch_y_cls.to(device)

            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                pred_reg, pred_cls = model(batch_x, edge_index)
                loss_r = criterion_reg(pred_reg, batch_y_reg, batch_y_cls)
                loss_c = criterion_cls(pred_cls, batch_y_cls)
                total_loss = (loss_r + 1.2 * loss_c) / accum_steps

            scaler_amp.scale(total_loss).backward()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler_amp.step(optimizer)
                scaler_amp.update()
                optimizer.zero_grad()

            train_loss += total_loss.item() * accum_steps

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                for batch_x, batch_y_reg, batch_y_cls in val_loader:
                    batch_x, batch_y_reg, batch_y_cls = batch_x.to(device), batch_y_reg.to(device), batch_y_cls.to(device)
                    pred_reg, pred_cls = model(batch_x, edge_index)
                    # Ponderación consistente a 1.2 en la validación
                    val_loss += (criterion_reg(pred_reg, batch_y_reg, batch_y_cls) + 1.2 * criterion_cls(pred_cls, batch_y_cls)).item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Epoca [{epoch:03d}/{epochs:03d}] | "
              f"Dur: {time.time()-t0:.1f}s | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[EARLY STOPPING] Convergencia alcanzada en epoca {epoch}.")
                break

    evaluate_only(history=history)

if __name__ == "__main__":
    train_and_evaluate()