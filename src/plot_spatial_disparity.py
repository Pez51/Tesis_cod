import os
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

def generate_disparity_plots():
    os.makedirs("reports/figures", exist_ok=True)
    sns.set_theme(style="whitegrid")

    df = pl.read_parquet("data/processed/df_master_weekly.parquet")

    # Agregación por distrito
    distritos = df.group_by(["ubigeo", "departamento", "distrito"]).agg([
        (pl.col("ep_men5") + pl.col("ep_may5")).sum().alias("total_casos")
    ]).sort("total_casos", descending=True)

    total_nacional = distritos["total_casos"].sum()
    distritos = distritos.with_columns([
        ((pl.col("total_casos") / total_nacional) * 100).alias("pct_del_total"),
        (((pl.col("total_casos") / total_nacional) * 100).cum_sum()).alias("pct_acumulado")
    ])

    pdf = distritos.to_pandas()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Gráfico 1: Top 15 Distritos con mayor carga
    top15 = pdf.head(15).copy()
    top15["etiqueta"] = top15["distrito"] + " (" + top15["departamento"] + ")"
    sns.barplot(data=top15, x="total_casos", y="etiqueta", palette="Blues_r", ax=axes[0])
    axes[0].set_title("Top 15 Distritos con Mayor Carga de EDA (2000 - 2024)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Casos Totales Reportados")
    axes[0].set_ylabel("")

    # Gráfico 2: Curva de Concentración de Pareto
    pct_distritos = (pdf.index + 1) / len(pdf) * 100
    axes[1].plot(pct_distritos, pdf["pct_acumulado"], color="#d62728", lw=2.5, label="Distribución Real EDA")
    axes[1].plot([0, 100], [0, 100], color="gray", linestyle="--", label="Distribución Equitativa (1:1)")
    
    # Marcador del 5%
    idx_5pct = int(len(pdf) * 0.05)
    val_5pct = pdf.loc[idx_5pct, "pct_acumulado"]
    axes[1].scatter([5], [val_5pct], color="black", s=60, zorder=5)
    axes[1].annotate(f"5% distritos = {val_5pct:.1f}% casos", xy=(5, val_5pct), xytext=(15, val_5pct - 10),
                     arrowprops=dict(arrowstyle="->", color="black", lw=1.5), fontweight="bold")

    axes[1].set_title("Curva de Concentración Espacial de EDA (Principio de Pareto)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Porcentaje Acumulado de Distritos (%)")
    axes[1].set_ylabel("Porcentaje Acumulado de Casos (%)")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    output_path = "reports/figures/05_disparidad_espacial_pareto.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Gráfico de disparidad espacial guardado exitosamente en: {output_path}")

if __name__ == "__main__":
    generate_disparity_plots()