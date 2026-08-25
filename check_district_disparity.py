import polars as pl

pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(10)
pl.Config.set_tbl_width_chars(160)

# Cargar dataset semanal procesado
df = pl.read_parquet("data/processed/df_master_weekly.parquet")

# Consolidar casos totales por distrito
distritos = df.group_by(["ubigeo", "departamento", "provincia", "distrito"]).agg([
    (pl.col("ep_men5") + pl.col("ep_may5")).sum().alias("total_casos_historicos"),
    pl.col("ep_men5").sum().alias("casos_men5"),
    pl.col("ep_may5").sum().alias("casos_may5"),
    pl.col("hosp_men5").sum().alias("hosp_men5")
]).sort("total_casos_historicos", descending=True)

# Calcular porcentaje acumulado del total nacional
total_nacional = distritos["total_casos_historicos"].sum()
distritos = distritos.with_columns([
    ((pl.col("total_casos_historicos") / total_nacional) * 100).alias("pct_del_total"),
    (((pl.col("total_casos_historicos") / total_nacional) * 100).cum_sum()).alias("pct_acumulado")
])

print("\n" + "="*110)
print(" TOP 15 DISTRITOS CON MAYOR CARGA EPIDEMIOLÓGICA DE EDA EN PERÚ (2000 - 2024)")
print("="*110)
print(distritos.head(15))
print("="*110)

print("\n" + "="*110)
print(" 15 DISTRITOS CON MENOR REGISTRO (COLA INFERIOR / RURALES)")
print("="*110)
print(distritos.tail(15))
print("="*110)

# Análisis de Concentración (Principio de Pareto)
n_distritos = distritos.height
top_5pct = int(n_distritos * 0.05)
pct_concentrado = distritos.head(top_5pct)["pct_del_total"].sum()

print(f"\n--- ANÁLISIS DE CONCENTRACIÓN GEOGRÁFICA ---")
print(f" Total de distritos evaluados        : {n_distritos}")
print(f" El 5% de distritos ({top_5pct} distritos) concentra el: {pct_concentrado:.2f}% de todos los casos del país.")
print("-" * 110 + "\n")