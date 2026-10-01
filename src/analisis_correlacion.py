import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import load_config, load_pickle

def generar_matriz_correlacion():
    config = load_config()
    processed_dir = config["paths"]["processed_dir"]
    figures_dir = os.path.join("reports", "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Cargar tensor, metadatos y etiquetas
    tensor_path = os.path.join(processed_dir, "tensor_eda_TNF.npy")
    metadata_path = os.path.join(processed_dir, "metadata.pkl")
    targets_cls_path = os.path.join(processed_dir, "targets_outbreak.npy")

    tensor = np.load(tensor_path)  # Forma: (T, N, F)
    metadata = load_pickle(metadata_path)
    T, N, F = tensor.shape

    # Cargar o extraer los nombres de las 17 variables
    feature_names = metadata.get("features") or metadata.get("feature_names", None)
    if "features" in metadata:
        feature_names = metadata["features"]
    elif "feature_names" in metadata:
        feature_names = metadata["feature_names"]
    else:
        feature_names = [
            "ep_men5", "hosp_men5", "def_men5", "ep_may5", "hosp_may5", "def_may5",
            "delta_ep_men5", "accel_ep_men5", "ratio_hosp_men5", "ratio_def_men5",
            "roll_mean_4", "roll_std_4", "surge_ratio", "sem_sin", "sem_cos",
            "lag_departamental_ep", "lag_2_ep"
        ]

    # Cargar objetivos: regresión continua en t+1 y etiqueta de brote
    # Si targets_outbreak existe en disco, se utiliza; caso contrario se alinea del tensor
    if os.path.exists(targets_cls_path):
        targets_cls = np.load(targets_cls_path)  # (T, N)
    else:
        q3_matrix = np.load(os.path.join(processed_dir, "endemic_channel_q3.npy"))
        time_index = metadata["time_index"]
        target_weeks = [w for _, w in time_index]
        q3_flat = np.stack([q3_matrix[w - 1, :] for w in target_weeks], axis=0)
        targets_cls = (tensor[:, :, 0] > q3_flat).astype(int)

    # Casos futuros a predecir (t+1)
    target_reg = np.roll(tensor[:, :, 0], shift=-1, axis=0)
    target_cls = np.roll(targets_cls, shift=-1, axis=0)

    # Omitir la última semana por el desplazamiento temporal
    tensor_valid = tensor[:-1, :, :]  # (T-1, N, F)
    target_reg_valid = target_reg[:-1, :]
    target_cls_valid = target_cls[:-1, :]

    # 2. Aplanar dimensiones (T-1)*N para analizar la distribución completa
    total_samples = (T - 1) * N
    X_flat = tensor_valid.reshape(total_samples, F)
    y_reg_flat = target_reg_valid.reshape(total_samples, 1)
    y_cls_flat = target_cls_valid.reshape(total_samples, 1)

    # Construir DataFrame consolidado
    columnas = feature_names + ["Target_Casos_t1", "Target_Brote_t1"]
    datos_matriz = np.hstack([X_flat, y_reg_flat, y_cls_flat])
    df = pd.DataFrame(datos_matriz, columns=columnas)

    print(f"\n[MATRIZ DE CORRELACIÓN] Muestras procesadas: {total_samples:,} observaciones provincia-semana.")
    print(f"[MATRIZ DE CORRELACIÓN] Variables evaluadas: {F} predictoras + 2 targets.")

    # 3. Calcular Matrices de Correlación
    corr_pearson = df.corr(method="pearson")
    corr_spearman = df.corr(method="spearman")

    # 4. Generar Heatmap Visual en Alta Resolución
    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))

    # Máscara triangular superior para no duplicar visualmente
    mask = np.triu(np.ones_like(corr_pearson, dtype=bool))

    # Gráfico A: Pearson (Relaciones Lineales)
    sns.heatmap(
        corr_pearson, mask=mask, cmap="vlag", vmin=-1.0, vmax=1.0, center=0,
        annot=True, fmt=".2f", annot_kws={"size": 8}, square=True,
        linewidths=0.5, cbar_kws={"shrink": 0.7}, ax=axes[0]
    )
    axes[0].set_title("A) Correlacion Lineal de Pearson (Features vs Targets)", fontsize=13, fontweight="bold")
    axes[0].tick_params(axis='x', rotation=45)

    # Gráfico B: Spearman (Relaciones Monótonas / No Lineales)
    sns.heatmap(
        corr_spearman, mask=mask, cmap="icefire", vmin=-1.0, vmax=1.0, center=0,
        annot=True, fmt=".2f", annot_kws={"size": 8}, square=True,
        linewidths=0.5, cbar_kws={"shrink": 0.7}, ax=axes[1]
    )
    axes[1].set_title("B) Correlacion de Rangos de Spearman (Features vs Targets)", fontsize=13, fontweight="bold")
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    out_img = os.path.join(figures_dir, "07_matriz_correlacion_features.png")
    plt.savefig(out_img, dpi=300)
    plt.close()
    print(f"[OK] Grafico exportado exitosamente en: {out_img}")

    # 5. Diagnóstico de Multicolinealidad (Variables redundantes con r > 0.85)
    print("\n" + "="*75)
    print(" 1. DIAGNÓSTICO DE REDUNDANCIA Y MULTICOLINEALIDAD (|Pearson| >= 0.85)")
    print("="*75)
    redundancias = []
    for i in range(F):
        for j in range(i + 1, F):
            val = corr_pearson.iloc[i, j]
            if abs(val) >= 0.85:
                redundancias.append((feature_names[i], feature_names[j], val))

    if redundancias:
        for v1, v2, val in sorted(redundancias, key=lambda x: abs(x[2]), reverse=True):
            print(f" -> {v1:<18} <--> {v2:<18} | r = {val:+.3f}")
    else:
        print(" -> No se encontraron pares con multicolinealidad severa (|r| >= 0.85).")

    # 6. Diagnóstico de Poder Predictivo hacia el Brote (Target_Brote_t1)
    print("\n" + "="*75)
    print(" 2. RANKING DE ASOCIACIÓN HACIA LA DETECCIÓN DE BROTE (Target_Brote_t1)")
    print("="*75)
    target_corr = corr_spearman["Target_Brote_t1"].drop(["Target_Casos_t1", "Target_Brote_t1"])
    target_corr_sorted = target_corr.abs().sort_values(ascending=False)

    for var in target_corr_sorted.index:
        p_val = corr_pearson.loc[var, "Target_Brote_t1"]
        s_val = corr_spearman.loc[var, "Target_Brote_t1"]
        print(f" -> {var:<18} | Spearman = {s_val:+.3f} | Pearson = {p_val:+.3f}")

    # Guardar tablas CSV para el anexo de la tesis
    csv_out = os.path.join("reports", "matriz_correlacion_consolidada.csv")
    corr_pearson.to_csv(csv_out)
    print(f"\n[OK] Matriz tabular guardada para documentacion en: {csv_out}")

if __name__ == "__main__":
    generar_matriz_correlacion()