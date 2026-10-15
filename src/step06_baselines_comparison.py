import os
import time
import random
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import maximum_filter1d
from scipy.stats import wilcoxon
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, roc_auc_score

from src.utils import load_config, load_pickle
from src.step03_dataset import get_dataloaders
from src.step04_model_stgcn import SpatioTemporalGNN

# 0. Semilla Global de Reproducibilidad
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# 1. Definición del Modelo Baseline Neuronal Temporal Puro (Sin Grafo / Sin GAT)
class PureTemporalGRU(nn.Module):
    """
    Red Neuronal Recurrente Pura que procesa cada provincia de forma independiente.
    Modela dependencias temporales sin intercambio de información espacial entre nodos.
    """
    def __init__(self, in_features=17, hidden_dim=64, num_layers=2, dropout=0.15):
        super(PureTemporalGRU, self).__init__()
        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Num_Nodes, Features)
        B, L, N, F = x.shape
        # Permutar y aplanar para tratar cada nodo como una serie independiente: (B * N, L, F)
        x_reshaped = x.permute(0, 2, 1, 3).reshape(B * N, L, F)
        gru_out, _ = self.gru(x_reshaped)
        h_last = gru_out[:, -1, :]  # Último estado oculto: (B * N, hidden_dim)

        out_reg = self.reg_head(h_last).reshape(B, N)
        out_cls = self.cls_head(h_last).reshape(B, N)
        return out_reg, out_cls


# 2. Métricas de Evaluación Epidemiológica Operativa (+/- 1 semana)
def compute_tolerant_metrics(y_true_mat, y_pred_mat, tolerance=1):
    y_true_exp = maximum_filter1d(y_true_mat, size=2 * tolerance + 1, axis=0, mode='nearest')
    y_pred_exp = maximum_filter1d(y_pred_mat, size=2 * tolerance + 1, axis=0, mode='nearest')

    tp = np.sum((y_pred_mat == 1) & (y_true_exp == 1))
    fp = np.sum((y_pred_mat == 1) & (y_true_exp == 0))
    fn = np.sum((y_true_mat == 1) & (y_pred_exp == 0))
    tn = np.sum((y_pred_mat == 0) & (y_true_exp == 0))

    prec = tp / (tp + fp + 1e-6)
    rec = tp / (tp + fn + 1e-6)
    f1_brote = 2 * (prec * rec) / (prec + rec + 1e-6)
    f1_norm = 2 * tn / (2 * tn + fp + fn + 1e-6)
    f1_macro = (f1_brote + f1_norm) / 2.0

    return f1_macro, f1_brote, prec, rec


# 3. Función de Entrenamiento para el Baseline Neuronal Temporal
def train_temporal_baseline(train_loader, val_loader, num_nodes, num_features, device, epochs=35):
    print("\n" + "="*70)
    print(" [ENTRENAMIENTO BASELINE] Red Temporal Pura GRU (Sin Grafo Spatial)")
    print("="*70)
    model = PureTemporalGRU(in_features=num_features, hidden_dim=64, num_layers=2, dropout=0.15).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion_reg = nn.HuberLoss(delta=1.0)
    criterion_cls = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y_reg, batch_y_cls in train_loader:
            batch_x = batch_x.to(device)
            batch_y_reg = batch_y_reg.to(device)
            batch_y_cls = batch_y_cls.to(device)

            optimizer.zero_grad()
            pred_reg, pred_cls = model(batch_x)
            loss_r = criterion_reg(pred_reg, batch_y_reg)
            loss_c = criterion_cls(pred_cls, batch_y_cls)
            total_loss = loss_r + 1.2 * loss_c
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()

        avg_train = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y_reg, batch_y_cls in val_loader:
                batch_x = batch_x.to(device)
                pred_reg, pred_cls = model(batch_x)
                loss_r = criterion_reg(pred_reg, batch_y_reg.to(device))
                loss_c = criterion_cls(pred_cls, batch_y_cls.to(device))
                val_loss += (loss_r + 1.2 * loss_c).item()

        avg_val = val_loss / len(val_loader)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_weights = model.state_dict().copy()

        if epoch % 5 == 0 or epoch == epochs:
            print(f" Epoca [{epoch:02d}/{epochs:02d}] | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

    model.load_state_dict(best_weights)
    return model


# 4. Pipeline Principal de Comparación y Pruebas Estadísticas
def run_baselines_comparison():
    config = load_config()
    processed_dir = config["paths"]["processed_dir"]
    models_dir = config["paths"]["models_dir"]
    reports_dir = "reports"
    figures_dir = os.path.join(reports_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[OE4 COMPARATIVA] Dispositivo de Inferencia: {device}")

    # Cargar DataLoaders y datos procesados
    train_loader, val_loader, test_loader, scalers = get_dataloaders()
    graph_data = torch.load(os.path.join(processed_dir, "graph_topology.pt"), weights_only=False)
    edge_index = graph_data["edge_index"].to(device)
    num_nodes = graph_data["num_nodes"]

    metadata = load_pickle(os.path.join(processed_dir, "metadata.pkl"))
    time_index = metadata["time_index"]
    endemic_channel_q3 = np.load(os.path.join(processed_dir, "endemic_channel_q3.npy"))
    tensor_data = np.load(os.path.join(processed_dir, "tensor_eda_TNF.npy"))
    num_features = tensor_data.shape[2]

    num_test_samples = len(test_loader.dataset)
    T_total = len(tensor_data)
    train_T = int(T_total * config["model_params"]["train_split"])
    val_T = int(T_total * config["model_params"]["val_split"])
    seq_len = config["model_params"]["seq_len"]

    test_target_weeks = [time_index[train_T + val_T + seq_len + i][1] for i in range(num_test_samples)]
    test_q3_mat = np.stack([endemic_channel_q3[w - 1, :] for w in test_target_weeks], axis=0)  # (184, 196)
    test_q3_flat = test_q3_mat.flatten()

    mean_target = scalers["mean_target"]
    std_target = scalers["std_target"]

    # Extraer valores reales del Test Set
    all_targets_reg, all_targets_cls = [], []
    test_inputs = []
    for bx, by_reg, by_cls in test_loader:
        test_inputs.append(bx)
        by_reg_abs = np.clip(np.expm1(by_reg.numpy() * std_target + mean_target), 0, None)
        all_targets_reg.append(by_reg_abs)
        all_targets_cls.append(by_cls.numpy().astype(int))

    y_test_reg_true = np.concatenate(all_targets_reg, axis=0)  # (184, 196)
    y_test_cls_true = np.concatenate(all_targets_cls, axis=0)  # (184, 196)
    y_true_reg_flat = y_test_reg_true.flatten()
    y_true_cls_flat = y_test_cls_true.flatten()

    # ---------------------------------------------------------
    # EVALUACIÓN 1: MODELO PROPUESTO ST-GNN (Pre-entrenado)
    # ---------------------------------------------------------
    print("\n[EVALUANDO 1/4] Red Espaciotemporal Multitarea ST-GNN...")
    stgnn_model = SpatioTemporalGNN(
        num_nodes=num_nodes, in_features=num_features, embed_dim=8, hidden_dim=64, dropout=0.15
    ).to(device)
    stgnn_model_path = os.path.join(models_dir, "best_stgnn_model.pt")
    stgnn_model.load_state_dict(torch.load(stgnn_model_path, weights_only=True))
    stgnn_model.eval()

    preds_stgnn_reg, probs_stgnn_cls = [], []
    with torch.no_grad():
        for bx in test_inputs:
            bx = bx.to(device)
            p_reg, p_cls = stgnn_model(bx, edge_index)
            reg_abs = np.clip(np.expm1(p_reg.cpu().numpy() * std_target + mean_target), 0, None)
            preds_stgnn_reg.append(reg_abs)
            probs_stgnn_cls.append(torch.sigmoid(p_cls).cpu().numpy())

    y_pred_stgnn_reg = np.concatenate(preds_stgnn_reg, axis=0)
    y_prob_stgnn_cls = np.concatenate(probs_stgnn_cls, axis=0)
    y_pred_stgnn_cls = (y_prob_stgnn_cls >= 0.44).astype(int)

    # ---------------------------------------------------------
    # EVALUACIÓN 2: BASELINE TEMPORAL PURO (GRU Sin Grafo)
    # ---------------------------------------------------------
    print("\n[EVALUANDO 2/4] Red Temporal Pura GRU...")
    temporal_model = train_temporal_baseline(train_loader, val_loader, num_nodes, num_features, device)
    temporal_model.eval()

    preds_gru_reg, probs_gru_cls = [], []
    with torch.no_grad():
        for bx in test_inputs:
            bx = bx.to(device)
            p_reg, p_cls = temporal_model(bx)
            reg_abs = np.clip(np.expm1(p_reg.cpu().numpy() * std_target + mean_target), 0, None)
            preds_gru_reg.append(reg_abs)
            probs_gru_cls.append(torch.sigmoid(p_cls).cpu().numpy())

    y_pred_gru_reg = np.concatenate(preds_gru_reg, axis=0)
    y_prob_gru_cls = np.concatenate(probs_gru_cls, axis=0)
    y_pred_gru_cls = (y_prob_gru_cls >= 0.50).astype(int)

    # ---------------------------------------------------------
    # EVALUACIÓN 3: BASELINE ESTADÍSTICO AUTORREGRESIVO (ARIMA/SARIMA PROXY)
    # Modelo univariado autorregresivo persistente ponderado por rezagos
    # ---------------------------------------------------------
    print("\n[EVALUANDO 3/4] Modelo Estadístico Autorregresivo Provincial...")
    # Extraer casos de rezago t-1 (columna 0 en el tensor)
    # Para cada semana k en test, la estimación AR(1) utiliza el lag 1 y la tendencia estacional
    y_pred_ar_reg = np.zeros_like(y_test_reg_true)
    for i, bx in enumerate(test_inputs):
        # bx shape: (Batch, Seq_Len, Nodes, Feats). La última semana t0 es el índice -1, feature 0 (ep_men5)
        # Invertir transformación logarítmica de la feature de casos
        last_observed = np.expm1(bx[:, -1, :, 0].numpy() * std_target + mean_target)
        # Modelo autorregresivo lineal con decaimiento: y_pred = 0.85 * y_{t-1} + 0.15 * Q3_local
        q3_step = test_q3_mat[i:i+len(bx), :]
        y_pred_ar_reg[i:i+len(bx), :] = np.clip(0.85 * last_observed + 0.15 * q3_step, 0, None)

    y_pred_ar_cls = (y_pred_ar_reg > test_q3_mat).astype(int)

    # ---------------------------------------------------------
    # EVALUACIÓN 4: BASELINE HEURÍSTICO MINSA (Canal Endémico Q3)
    # La heurística vigente clasifica brote si y_t > Q3
    # ---------------------------------------------------------
    print("\n[EVALUANDO 4/4] Heurística Actual de Canal Endémico MINSA (Q3)...")
    # Para regresión, el canal endémico actúa asumiendo la mediana/Q3 como expectativa
    y_pred_minsa_reg = test_q3_mat.copy()
    y_pred_minsa_cls = (y_test_reg_true > test_q3_mat).astype(int)

    # ---------------------------------------------------------
    # CÁLCULO DE MÉTRICAS Y TABLA CONSOLIDADA
    # ---------------------------------------------------------
    modelos = {
        "ST-GNN Propuesto (OE1)": (y_pred_stgnn_reg, y_pred_stgnn_cls),
        "GRU Temporal Puro (Sin Grafo)": (y_pred_gru_reg, y_pred_gru_cls),
        "Modelo Autorregresivo Provincial": (y_pred_ar_reg, y_pred_ar_cls),
        "Canal Endémico MINSA (Q3)": (y_pred_minsa_reg, y_pred_minsa_cls)
    }

    resultados_lista = []
    errores_absolutos = {}

    for nombre, (pred_r, pred_c) in modelos.items():
        # Métricas de Regresión
        r2 = r2_score(y_true_reg_flat, pred_r.flatten())
        mae = mean_absolute_error(y_true_reg_flat, pred_r.flatten())
        rmse = np.sqrt(mean_squared_error(y_true_reg_flat, pred_r.flatten()))

        # Errores absolutos por muestra para prueba de hipótesis
        errores_absolutos[nombre] = np.abs(y_true_reg_flat - pred_r.flatten())

        # Métricas de Clasificación Puntual
        f1_mac_ex = f1_score(y_true_cls_flat, pred_c.flatten(), average="macro", zero_division=0)
        f1_bro_ex = f1_score(y_true_cls_flat, pred_c.flatten(), average="binary", zero_division=0)

        # Métricas con Tolerancia (+/- 1 sem)
        f1_mac_tol, f1_bro_tol, prec_tol, rec_tol = compute_tolerant_metrics(y_test_cls_true, pred_c, tolerance=1)

        resultados_lista.append({
            "Modelo": nombre,
            "R2": round(r2, 4),
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "F1-Macro (Exacto)": round(f1_mac_ex, 4),
            "F1-Brote (Exacto)": round(f1_bro_ex, 4),
            "F1-Macro (±1s)": round(f1_mac_tol, 4),
            "F1-Brote (±1s)": round(f1_bro_tol, 4),
            "Precisión (±1s)": f"{prec_tol*100:.1f}%",
            "Sensibilidad (±1s)": f"{rec_tol*100:.1f}%"
        })

    df_resultados = pd.DataFrame(resultados_lista)

    # ---------------------------------------------------------
    # PRUEBA DE HIPÓTESIS NO PARAMÉTRICA DE WILCOXON (OE4)
    # Contraste de errores absolutos de ST-GNN frente a baselines
    # ---------------------------------------------------------
    print("\n" + "="*70)
    print(" CONTRASTE DE HIPÓTESIS: PRUEBA DE WILCOXON (ST-GNN vs. BASELINES)")
    print("="*70)
    err_stgnn = errores_absolutos["ST-GNN Propuesto (OE1)"]

    wilcoxon_res = []
    for nombre in ["GRU Temporal Puro (Sin Grafo)", "Modelo Autorregresivo Provincial", "Canal Endémico MINSA (Q3)"]:
        err_baseline = errores_absolutos[nombre]
        # Muestreo estratificado de residuos para estabilidad en scipy (10,000 pares)
        idx_sub = np.random.choice(len(err_stgnn), min(10000, len(err_stgnn)), replace=False)
        stat, p_val = wilcoxon(err_stgnn[idx_sub], err_baseline[idx_sub], alternative="less")
        
        significativo = "Rechaza H0 (p < 0.05)" if p_val < 0.05 else "No Rechaza H0"
        print(f" -> ST-GNN vs {nombre:<30}: Stat = {stat:.2e} | p-valor = {p_val:.4e} | {significativo}")
        wilcoxon_res.append({"Baseline": nombre, "Wilcoxon_W": stat, "p_value": p_val, "Decision": significativo})

    df_wilcoxon = pd.DataFrame(wilcoxon_res)

    # Guardar Tablas en disco
    df_resultados.to_csv(os.path.join(reports_dir, "tabla_comparativa_oe4_baselines.csv"), index=False)
    df_wilcoxon.to_csv(os.path.join(reports_dir, "tabla_wilcoxon_oe4.csv"), index=False)
    print(f"\n[OK] Tablas consolidadas guardadas en carpeta: {reports_dir}/")

    # ---------------------------------------------------------
    # GENERACIÓN DE FIGURA COMPARATIVA COMPUESTA (FIGURA 08)
    # ---------------------------------------------------------
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    nombres_cortos = ["ST-GNN (Propuesta)", "GRU Puro", "Autorregresivo", "Canal Q3"]
    colores = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]

    # Subplot 1: R² de Regresión
    r2_vals = df_resultados["R2"].tolist()
    bars1 = axes[0].bar(nombres_cortos, r2_vals, color=colores, width=0.55)
    axes[0].set_title("A) Coeficiente de Determinacion (R2)", fontsize=11, fontweight="bold")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("R2 (Bondad de Ajuste)", fontsize=10)
    for b in bars1:
        yval = b.get_height()
        axes[0].text(b.get_x() + b.get_width()/2.0, yval + 0.02, f"{yval:.4f}", ha='center', va='bottom', fontsize=9, fontweight="bold")

    # Subplot 2: MAE (Error Absoluto Medio)
    mae_vals = df_resultados["MAE"].tolist()
    bars2 = axes[1].bar(nombres_cortos, mae_vals, color=colores, width=0.55)
    axes[1].set_title("B) Error Absoluto Medio (MAE en Casos)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Casos Semanales Desviados", fontsize=10)
    for b in bars2:
        yval = b.get_height()
        axes[1].text(b.get_x() + b.get_width()/2.0, yval + 0.3, f"{yval:.2f}", ha='center', va='bottom', fontsize=9, fontweight="bold")

    # Subplot 3: F1-Macro Operativo (±1 sem)
    f1_vals = df_resultados["F1-Macro (±1s)"].tolist()
    bars3 = axes[2].bar(nombres_cortos, f1_vals, color=colores, width=0.55)
    axes[2].set_title("C) F1-Macro Vigilancia Operativa (+/-1 sem)", fontsize=11, fontweight="bold")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("F1-Score Macro", fontsize=10)
    for b in bars3:
        yval = b.get_height()
        axes[2].text(b.get_x() + b.get_width()/2.0, yval + 0.02, f"{yval:.4f}", ha='center', va='bottom', fontsize=9, fontweight="bold")

    for ax in axes:
        ax.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    fig_path = os.path.join(figures_dir, "08_comparativa_baselines_oe4.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[OK] Grafico comparativo generado en: {fig_path}")

    # Mostrar Reporte Final en Consola
    print("\n" + "="*85)
    print(" REPORTE TABULAR CONSOLIDADO: EVALUACIÓN DE OBJETIVO ESPECÍFICO 4 (OE4)")
    print("="*85)
    print(df_resultados.to_string(index=False))
    print("="*85 + "\n")

if __name__ == "__main__":
    run_baselines_comparison()