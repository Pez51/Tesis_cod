import os
import polars as pl
import numpy as np
from src.utils import load_config, save_pickle

def run_etl():
    config = load_config()
    raw_path = config["paths"]["raw_csv"]
    processed_dir = config["paths"]["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)

    print("[ETL 1/5] Leyendo dataset crudo...")
    
    column_names = [
        "departamento", "provincia", "distrito", "ano", "semana",
        "sub_reg_nt", "ubigeo", "episodios_men5", "hospitalizados_men5",
        "defunciones_men5", "episodios_may5", "hospitalizados_may5", "defunciones_may5"
    ]

    df_raw = pl.read_csv(
        raw_path,
        has_header=False,
        new_columns=column_names,
        separator=";",
        encoding="utf8-lossy",
        infer_schema_length=10000,
        ignore_errors=True
    )
    
    total_raw_records = df_raw.height

    print("[ETL 2/5] Casteando tipos y aplicando filtros de consistencia...")
    df_clean = df_raw.with_columns([
        pl.col("ano").cast(pl.Int32, strict=False),
        pl.col("semana").cast(pl.Int32, strict=False),
        pl.col("ubigeo").cast(pl.Utf8, strict=False).str.strip_chars().str.zfill(6),
        pl.col("departamento").cast(pl.Utf8, strict=False).str.strip_chars(),
        pl.col("provincia").cast(pl.Utf8, strict=False).str.strip_chars(),
        pl.col("distrito").cast(pl.Utf8, strict=False).str.strip_chars(),
        pl.col("episodios_men5").cast(pl.Float32, strict=False).fill_null(0.0),
        pl.col("hospitalizados_men5").cast(pl.Float32, strict=False).fill_null(0.0),
        pl.col("defunciones_men5").cast(pl.Float32, strict=False).fill_null(0.0),
        pl.col("episodios_may5").cast(pl.Float32, strict=False).fill_null(0.0),
        pl.col("hospitalizados_may5").cast(pl.Float32, strict=False).fill_null(0.0),
        pl.col("defunciones_may5").cast(pl.Float32, strict=False).fill_null(0.0),
    ])

    start_y = config["data_processing"]["start_year"]
    end_y = config["data_processing"]["end_year"]

    # Filtro de registros válidos
    valid_df = df_clean.filter(
        (pl.col("ano") >= start_y) & (pl.col("ano") <= end_y) &
        (pl.col("semana") >= 1) & (pl.col("semana") <= 53) &
        (pl.col("ubigeo").is_not_null()) &
        (pl.col("ubigeo").str.len_chars() == 6)
    )

    total_valid_records = valid_df.height
    dropped_records = total_raw_records - total_valid_records
    retention_rate = (total_valid_records / total_raw_records) * 100

    print("[ETL 3/5] Agregando múltiples postas a nivel Distrito-Semana...")
    agg_df = valid_df.group_by(["ubigeo", "departamento", "provincia", "distrito", "ano", "semana"]).agg([
        pl.col("episodios_men5").sum().alias("ep_men5"),
        pl.col("hospitalizados_men5").sum().alias("hosp_men5"),
        pl.col("defunciones_men5").sum().alias("def_men5"),
        pl.col("episodios_may5").sum().alias("ep_may5"),
        pl.col("hospitalizados_may5").sum().alias("hosp_may5"),
        pl.col("defunciones_may5").sum().alias("def_may5"),
    ])

    aggregated_records = agg_df.height

    # Construir grilla temporal continua
    time_index = []
    for y in range(start_y, end_y + 1):
        max_sem = 53 if y in [2004, 2009, 2015, 2020] else 52
        for s in range(1, max_sem + 1):
            time_index.append((y, s))

    time_to_idx = {t: i for i, t in enumerate(time_index)}
    T = len(time_index)

    unique_districts = agg_df.select(
        ["ubigeo", "departamento", "provincia", "distrito"]
    ).unique("ubigeo").sort("ubigeo")

    valid_ubigeos = unique_districts["ubigeo"].to_list()
    ubigeo_to_idx = {u: i for i, u in enumerate(valid_ubigeos)}
    N = len(valid_ubigeos)

    features = config["data_processing"]["features"]
    F = len(features)

    # Tensor (T, N, F)
    data_tensor = np.zeros((T, N, F), dtype=np.float32)
    pdf = agg_df.to_pandas()

    for row in pdf.itertuples():
        t_i = time_to_idx.get((row.ano, row.semana))
        n_i = ubigeo_to_idx.get(row.ubigeo)
        if t_i is not None and n_i is not None:
            data_tensor[t_i, n_i, :] = [
                row.ep_men5, row.hosp_men5, row.def_men5,
                row.ep_may5, row.hosp_may5, row.def_may5
            ]

    print("[ETL 4/5] Generando matriz binaria de brotes (Percentil)...")
    target_idx = config["model_params"]["target_idx"]
    target_series = data_tensor[:, :, target_idx]
    
    outbreak_matrix = np.zeros((T, N), dtype=np.int64)
    pct = config["data_processing"]["outbreak_percentile"]
    
    for n in range(N):
        node_vals = target_series[:, n]
        non_zero = node_vals[node_vals > 0]
        threshold = np.percentile(non_zero, pct) if len(non_zero) > 10 else 1.0
        outbreak_matrix[:, n] = (node_vals >= threshold).astype(np.int64)

    print("[ETL 5/5] Exportando artefactos a 'data/processed/'...")
    np.save(os.path.join(processed_dir, "tensor_eda_TNF.npy"), data_tensor)
    np.save(os.path.join(processed_dir, "targets_outbreak.npy"), outbreak_matrix)
    agg_df.write_parquet(os.path.join(processed_dir, "df_master_weekly.parquet"))

    metadata = {
        "ubigeos": valid_ubigeos,
        "ubigeo_to_idx": ubigeo_to_idx,
        "time_index": time_index,
        "features": features,
        "districts_catalog": unique_districts.to_dicts()
    }
    save_pickle(metadata, os.path.join(processed_dir, "metadata.pkl"))

    # Reporte de Auditoría
    print("\n" + "-"*45)
    print(" REPORTE DE AUDITORÍA Y LIMPIEZA DE DATOS")
    print("-"*45)
    print(f" Registros totales leidos (CSV)     : {total_raw_records:,}")
    print(f" Registros válidos retenidos       : {total_valid_records:,}")
    print(f" Registros descartados (anomalías) : {dropped_records:,}")
    print(f" Tasa de retención de datos         : {retention_rate:.2f}%")
    print(f" Registros agregados (Distrito-Sem): {aggregated_records:,}")
    print(f" Distritos únicos incorporados (N) : {N:,}")
    print(f" Pasos temporales / semanas (T)    : {T:,}")
    print(f" Forma final del Tensor (T x N x F) : {data_tensor.shape}")
    print("-"*45 + "\n")

if __name__ == "__main__":
    run_etl()