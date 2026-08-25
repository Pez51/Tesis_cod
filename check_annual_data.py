import polars as pl

# Configurar Polars para mostrar todas las filas y columnas completas
pl.Config.set_tbl_rows(35)
pl.Config.set_tbl_cols(10)
pl.Config.set_tbl_width_chars(160)

# Cargar parquet procesado
df = pl.read_parquet("data/processed/df_master_weekly.parquet")

# Agregación anual considerando menores (<5) y mayores (>=5)
resumen = df.group_by("ano").agg([
    pl.col("ep_men5").sum().alias("casos_men5"),
    pl.col("ep_may5").sum().alias("casos_may5"),
    (pl.col("ep_men5") + pl.col("ep_may5")).sum().alias("total_casos_global"),
    (pl.col("hosp_men5") + pl.col("hosp_may5")).sum().alias("total_hospitalizados"),
    (pl.col("def_men5") + pl.col("def_may5")).sum().alias("total_defunciones")
]).sort("ano")

print("\n" + "="*100)
print(" CONSOLIDADO HISTÓRICO ANUAL DE EDA EN PERÚ (2000 - 2024) - TODOS LOS GRUPOS DE EDAD")
print("="*100)
print(resumen)
print("="*100)

# Gran total de los 25 años
totales_historicos = resumen.select([
    pl.col("casos_men5").sum().alias("sum_casos_men5"),
    pl.col("casos_may5").sum().alias("sum_casos_may5"),
    pl.col("total_casos_global").sum().alias("gran_total_casos"),
    pl.col("total_hospitalizados").sum().alias("gran_total_hosp"),
    pl.col("total_defunciones").sum().alias("gran_total_def")
])

print("\n--- ACUMULADO HISTÓRICO TOTAL (2000 - 2024) ---")
print(totales_historicos)
print("-" * 100 + "\n")