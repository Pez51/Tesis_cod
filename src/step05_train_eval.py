import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve
)
from src.utils import load_config, save_experiment_report
from src.step03_dataset import get_dataloaders
from src.step04_model_stgcn import SpatioTemporalGNN

def format_seconds(seconds):
    return str(timedelta(seconds=int(seconds)))

# Pérdida de Regresión Ponderada para Forzar el Aprendizaje de Picos Epidémicos
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
        # Pondera más alto el error cuando el distrito está en brote
        weight = torch.where(is_outbreak > 0.5, self.peak_weight, 1.0)
        return (weight * huber).mean()

class PerNodeFocalLoss(nn.Module):
    def __init__(self, alpha=0.60, gamma=2.0, pos_weight=3.5):
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

def calibrate_scale_aware_gate(y_true, y_probs_cls, y_pred_reg, thresholds_vec, stds_vec):
    """Calibración protegida contra desbordamientos numéricos (exp overflow)."""
    denom = np.where(stds_vec < 1e-3, 1.0, stds_vec)
    z_excess = (y_pred_reg - thresholds_vec) / denom
    
    # Clip numérico para eliminar RuntimeWarning: overflow in exp
    z_clipped = np.clip(-1.5 * z_excess, -30.0, 30.0)
    prob_physical = 1.0 / (1.0 + np.exp(z_clipped))
    
    best_f1 = 0.0
    best_w = 0.50
    best_cutoff = 0.50

    weights = [0.10, 0.25, 0.40, 0.50, 0.60, 0.70, 0.85]
    cutoffs = np.linspace(0.20, 0.80, 61)

    for w in weights:
        combined_score = w * y_probs_cls + (1.0 - w) * prob_physical
        for c in cutoffs:
            preds = (combined_score >= c).astype(int)
            score = f1_score(y_true, preds, average="macro", zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_w = w
                best_cutoff = c

    return best_w, best_cutoff, best_f1

def plot_and_save_metrics(history, y_true_reg, y_pred_reg, y_true_cls, y_prob_cls, y_pred_cls, figures_dir):
    os.makedirs(figures_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Curvas de Pérdida
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    plt.plot(history["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2)
    plt.title("Evolución de Pérdida Multitarea (ST-GNN Optimizado)", fontsize=12, fontweight="bold")
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "01_curvas_aprendizaje.png"), dpi=300)
    plt.close()

    # 2. Matriz de Confusión
    cm = confusion_matrix(y_true_cls, y_pred_cls)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues", cbar=False,
                xticklabels=["Normal (0)", "Brote (1)"], yticklabels=["Normal (0)", "Brote (1)"])
    plt.title("Matriz de Confusión Calibrada (Alertas de Brote)", fontsize=12, fontweight="bold")
    plt.xlabel("Predicción del Modelo")
    plt.ylabel("Estado Real Notificado")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "02_matriz_confusion_brotes.png"), dpi=300)
    plt.close()

    # 3. Curvas ROC y Precision-Recall
    fpr, tpr, _ = roc_curve(y_true_cls, y_prob_cls)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_true_cls, y_prob_cls)
    pr_auc = auc(rec, prec)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fpr, tpr, color="#2ca02c", lw=2, label=f"AUC-ROC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], color="gray", linestyle="--")
    axes[0].set_title("Curva ROC (Detección de Brotes)", fontweight="bold")
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

    # 4. Dispersión Real vs Predicho
    sample_indices = np.random.choice(len(y_true_reg), min(5000, len(y_true_reg)), replace=False)
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true_reg[sample_indices], y_pred_reg[sample_indices], alpha=0.35, color="#17becf", edgecolors="none")
    max_val = max(np.percentile(y_true_reg, 99), np.percentile(y_pred_reg, 99))
    plt.plot([0, max_val], [0, max_val], color="red", linestyle="--", lw=1.5, label="Ajuste Ideal (1:1)")
    plt.xlim(0, max_val * 1.05)
    plt.ylim(0, max_val * 1.05)
    plt.title("Ajuste del Modelo: Casos Reales vs. Predichos", fontsize=12, fontweight="bold")
    plt.xlabel("Casos Reales Notificados")
    plt.ylabel("Casos Predichos (ST-GNN)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "04_dispersion_real_vs_pred.png"), dpi=300)
    plt.close()

def train_and_evaluate():
    config = load_config()
    processed_dir = config["paths"]["processed_dir"]
    models_dir = config["paths"]["models_dir"]
    figures_dir = os.path.join("reports", "figures")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[ENTRENAMIENTO] Dispositivo activo: {device}")
    if torch.cuda.is_available():
        print(f" -> GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()

    train_loader, val_loader, test_loader, scalers = get_dataloaders()
    graph_data = torch.load(os.path.join(processed_dir, "graph_topology.pt"), weights_only=False)
    edge_index = graph_data["edge_index"].to(device)
    num_nodes = graph_data["num_nodes"]

    tensor_data = np.load(os.path.join(processed_dir, "tensor_eda_TNF.npy"))
    stds_vec = np.std(tensor_data[:, :, 0], axis=0)
    thresholds_path = os.path.join(processed_dir, "district_thresholds.npy")
    district_thresholds = np.load(thresholds_path) if os.path.exists(thresholds_path) else np.full(num_nodes, 2.0)

    hidden_dim = config["model_params"].get("hidden_dim", 64)
    accum_steps = config["model_params"].get("gradient_accumulation_steps", 4)
    num_features = len(config["data_processing"]["features"])

    model = SpatioTemporalGNN(
        num_nodes=num_nodes, in_features=num_features, embed_dim=16, hidden_dim=hidden_dim, dropout=0.15
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["model_params"]["learning_rate"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=1, eta_min=1e-5)
    scaler_amp = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    criterion_reg = PeakAwareRegressionLoss(delta=1.0, peak_weight=2.5)
    criterion_cls = PerNodeFocalLoss(alpha=0.60, gamma=2.0, pos_weight=float(scalers["pos_weight"]))

    epochs = config["model_params"]["epochs"]
    best_val_loss = float("inf")
    patience = 20
    patience_counter = 0
    best_model_path = os.path.join(models_dir, "best_stgnn_model.pt")

    if os.path.exists(best_model_path):
        os.remove(best_model_path)

    history = {"train_loss": [], "val_loss": []}
    epoch_times = []

    time_start_global = datetime.now()
    start_total_secs = time.time()

    print("\n" + "="*85)
    print(f" INICIO ENTRENAMIENTO ST-GNN POTENCIADO (STAN + KAPOOR) | Hora: {time_start_global.strftime('%H:%M:%S')}")
    print("="*85)

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, (batch_x, batch_y_reg, batch_y_cls) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_y_reg = batch_y_reg.to(device)
            batch_y_cls = batch_y_cls.to(device)

            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                pred_reg, pred_cls = model(batch_x, edge_index)
                
                loss_r = criterion_reg(pred_reg, batch_y_reg, batch_y_cls)
                loss_c = criterion_cls(pred_cls, batch_y_cls)
                total_loss = (loss_r + 1.8 * loss_c) / accum_steps

            scaler_amp.scale(total_loss).backward()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler_amp.step(optimizer)
                scaler_amp.update()
                optimizer.zero_grad()

            train_loss += total_loss.item() * accum_steps

        avg_train_loss = train_loss / len(train_loader)

        torch.cuda.empty_cache()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                for batch_x, batch_y_reg, batch_y_cls in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y_reg = batch_y_reg.to(device)
                    batch_y_cls = batch_y_cls.to(device)

                    pred_reg, pred_cls = model(batch_x, edge_index)
                    l_r = criterion_reg(pred_reg, batch_y_reg, batch_y_cls)
                    l_c = criterion_cls(pred_cls, batch_y_cls)
                    val_loss += (l_r + 1.8 * l_c).item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()

        epoch_duration = time.time() - epoch_start_time
        epoch_times.append(epoch_duration)
        epoch_end_dt = datetime.now()
        elapsed_total = time.time() - start_total_secs

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        dur_str = f"{epoch_duration:.1f}s" if epoch_duration < 60 else f"{int(epoch_duration//60)}m {int(epoch_duration%60):02d}s"

        print(f"[{epoch_end_dt.strftime('%H:%M:%S')}] Época [{epoch:03d}/{epochs:03d}] | "
              f"Duración: {dur_str} | Acumulado: {format_seconds(elapsed_total)} | "
              f"Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | LR: {current_lr:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[EARLY STOPPING] Detenido en la época {epoch} por convergencia.")
                break

    time_end_global = datetime.now()
    total_train_secs = time.time() - start_total_secs
    avg_epoch_secs = np.mean(epoch_times)

    print("\n" + "-"*65)
    print(" CRONOGRAMA DE ENTRENAMIENTO")
    print("-"*65)
    print(f" Hora de Inicio        : {time_start_global.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Hora de Finalización  : {time_end_global.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Duración Total        : {format_seconds(total_train_secs)} ({total_train_secs/60:.2f} min)")
    print(f" Tiempo Promedio/Época : {avg_epoch_secs:.2f} s")
    print(f" Épocas Ejecutadas     : {len(history['train_loss'])}")
    print("-"*65 + "\n")

    # 1. Calibración Numéricamente Estable sobre Validación
    torch.cuda.empty_cache()
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()

    val_probs_cls, val_preds_reg, val_targets_cls = [], [], []
    mean_val = scalers["mean_target"]
    std_val = scalers["std_target"]

    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            for batch_x, _, batch_y_cls in val_loader:
                batch_x = batch_x.to(device)
                pred_reg, pred_cls = model(batch_x, edge_index)

                pred_log = pred_reg.cpu().numpy() * std_val + mean_val
                real_pred_reg = np.clip(np.expm1(pred_log), a_min=0, a_max=None)

                val_preds_reg.append(real_pred_reg)
                val_probs_cls.append(torch.sigmoid(pred_cls).cpu().numpy())
                val_targets_cls.append(batch_y_cls.numpy().astype(int))

    y_val_true = np.concatenate(val_targets_cls, axis=0).flatten()
    y_val_prob = np.concatenate(val_probs_cls, axis=0).flatten()
    y_val_reg = np.concatenate(val_preds_reg, axis=0).flatten()

    val_count = len(val_loader.dataset)
    val_thresholds_exp = np.tile(district_thresholds, val_count)
    val_stds_exp = np.tile(stds_vec, val_count)

    best_w, best_cutoff, val_f1 = calibrate_scale_aware_gate(
        y_val_true, y_val_prob, y_val_reg, val_thresholds_exp, val_stds_exp
    )
    print(f"[CALIBRACIÓN] Parámetros Óptimos -> Peso Neuronal w: {best_w:.2f} | Corte: {best_cutoff:.2f} (F1-Val: {val_f1:.4f})")

    # 2. Evaluación en Test y Latencia
    print("[EVALUACIÓN] Evaluando sobre conjunto de Prueba (Test)...")
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

                real_pred_reg = np.clip(np.expm1(pred_log), a_min=0, a_max=None)
                real_target_reg = np.clip(np.expm1(target_log), a_min=0, a_max=None)

                all_preds_reg.append(real_pred_reg)
                all_targets_reg.append(real_target_reg)

                prob_cls = torch.sigmoid(pred_cls).cpu().numpy()
                all_probs_cls.append(prob_cls)
                all_targets_cls.append(batch_y_cls.numpy().astype(int))

    total_infer_time = time.time() - start_infer_time
    total_test_samples = len(test_loader.dataset)
    latency_per_sample = (total_infer_time / total_test_samples) * 1000

    y_true_reg = np.concatenate(all_targets_reg, axis=0).flatten()
    y_pred_reg = np.concatenate(all_preds_reg, axis=0).flatten()
    y_true_cls = np.concatenate(all_targets_cls, axis=0).flatten()
    y_prob_cls = np.concatenate(all_probs_cls, axis=0).flatten()

    test_thresholds_exp = np.tile(district_thresholds, total_test_samples)
    test_stds_exp = np.tile(stds_vec, total_test_samples)

    denom_test = np.where(test_stds_exp < 1e-3, 1.0, test_stds_exp)
    z_excess_test = (y_pred_reg - test_thresholds_exp) / denom_test
    z_clipped_test = np.clip(-1.5 * z_excess_test, -30.0, 30.0)
    prob_physical_test = 1.0 / (1.0 + np.exp(z_clipped_test))
    
    final_score = best_w * y_prob_cls + (1.0 - best_w) * prob_physical_test
    y_pred_cls = (final_score >= best_cutoff).astype(int)

    rmse = float(np.sqrt(mean_squared_error(y_true_reg, y_pred_reg)))
    mae = float(mean_absolute_error(y_true_reg, y_pred_reg))
    r2 = float(r2_score(y_true_reg, y_pred_reg))
    f1_macro = float(f1_score(y_true_cls, y_pred_cls, average="macro"))
    f1_binary = float(f1_score(y_true_cls, y_pred_cls, average="binary"))
    fpr, tpr, _ = roc_curve(y_true_cls, final_score)
    roc_auc = float(auc(fpr, tpr))

    plot_and_save_metrics(history, y_true_reg, y_pred_reg, y_true_cls, final_score, y_pred_cls, figures_dir)

    report_data = {
        "CRONOMETRAJE COMPUTACIONAL": {
            "Dispositivo": str(device),
            "Hora de Inicio": time_start_global.strftime('%Y-%m-%d %H:%M:%S'),
            "Hora de Finalización": time_end_global.strftime('%Y-%m-%d %H:%M:%S'),
            "Tiempo Total de Entrenamiento": f"{format_seconds(total_train_secs)} ({total_train_secs/60:.2f} min)",
            "Tiempo Promedio por Época": f"{avg_epoch_secs:.2f} s",
            "Latencia de Inferencia (por semana)": f"{latency_per_sample:.2f} ms (1,891 distritos en paralelo)",
            "Épocas Ejecutadas": len(history["train_loss"])
        },
        "MÉTRICAS DE REGRESIÓN (CASOS CONTINUOS)": {
            "RMSE": f"{rmse:.4f} casos",
            "MAE": f"{mae:.4f} casos",
            "R² (Coef. Determinación)": f"{r2:.4f}"
        },
        "MÉTRICAS DE CLASIFICACIÓN (COMPUERTA ESCALAR CALIBRADA)": {
            "F1-Score (Macro)": f"{f1_macro:.4f}",
            "F1-Score (Clase Brote)": f"{f1_binary:.4f}",
            "AUC-ROC": f"{roc_auc:.4f}",
            "Ponderación Neuronal w": f"{best_w:.2f}",
            "Umbral de Corte": f"{best_cutoff:.2f}"
        }
    }
    save_experiment_report(report_data)

    print("\n" + "="*65)
    print(" EVALUACIÓN FINALIZADA Y REPORTE ACTUALIZADO")
    print(f" -> Tiempo Total : {format_seconds(total_train_secs)} ({total_train_secs/60:.2f} min)")
    print(f" -> Latencia     : {latency_per_sample:.2f} ms por semana")
    print(f" -> RMSE         : {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")
    print(f" -> F1-Macro     : {f1_macro:.4f} | AUC-ROC: {roc_auc:.4f} (w: {best_w:.2f}, Corte: {best_cutoff:.2f})")
    print("="*65 + "\n")

if __name__ == "__main__":
    train_and_evaluate()