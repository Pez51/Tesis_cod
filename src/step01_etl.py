import os
import polars as pl
import numpy as np
from src.utils import load_config, save_pickle

def run_etl():
    config = load_config()
    raw_path = config["paths"]["raw_csv"]
    processed_dir = config["paths"]["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)

    print("[ETL 1/5] Cargando dataset bruto con Polars...")
    df = pl.read_csv(
        raw_path,
        schema_overrides={
            "ano": pl.Int32,
            "semana": pl.Int32,
            "ubigeo": pl.Utf8,
            "departamento": pl.Utf8,
            "provincia": pl.Utf8,
            "distrito": pl.Utf8,
            "episodios_men5": pl.Float32,
            "hospitalizados_men5": pl.Float32,
            "defunciones_men5": pl.Float32,
            "episodios_may5": pl.Float32,
            "hospitalizados_may5": pl.Float32,
            "defunciones_may5": pl.Float32
        },
        ignore_errors=True
    )

    print("[ETL 2/5] Estandarizando y filtrando rango temporal...")
    # Limpieza de UBIGEO a 6 caracteres
    df = df.with_columns(
        pl.col("ubigeo").str.strip_chars().str.zfill(6)
    )

    # Filtrar rango temporal 2000-2024
    start_y = config["data_processing"]["start_year"]
    end_y = config["data_processing"]["end_year"]
    df = df.filter(
        (pl.col("ano") >= start_y) & (pl.col("ano") <= end_y) &
        (pl.col("semana") >= 1) & (pl.col("semana") <= 53)
    )

    print("[ETL 3/5] Agregando registros por distrito, año y semana...")
    agg_df = df.group_by(["ubigeo", "departamento", "provincia", "distrito", "ano", "semana"]).agg([
        pl.col("episodios_men5").sum().fill_null(0).alias("ep_men5"),
        pl.col("hospitalizados_men5").sum().fill_null(0).alias("hosp_men5"),
        pl.col("defunciones_men5").sum().fill_null(0).alias("def_men5"),
        pl.col("episodios_may5").sum().fill_null(0).alias("ep_may5"),
        pl.col("hospitalizados_may5").sum().fill_null(0).alias("hosp_may5"),
        pl.col("defunciones_may5").sum().fill_null(0).alias("def_may5"),
    ])

    # Construir grilla temporal completa (Semanas 2000 - 2024)
    time_index = []
    for y in range(start_y, end_y + 1):
        max_sem = 53 if y in [2004, 2009, 2015, 2020] else 52
        for s in range(1, max_sem + 1):
            time_index.append((y, s))

    time_to_idx = {t: i for i, t in enumerate(time_index)}
    T = len(time_index)

    # Filtrar distritos activos según consistencia
    unique_districts = agg_df.select(["ubigeo", "departamento", "provincia", "distrito"]).unique("ubigeo").sort("ubigeo")
    valid_ubigeos = unique_districts["ubigeo"].to_list()
    ubigeo_to_idx = {u: i for i, u in enumerate(valid_ubigeos)}
    N = len(valid_ubigeos)

    features = config["data_processing"]["features"]
    F = len(features)

    print(f"-> Dimensiones: {T} pasos temporales, {N} distritos, {F} variables.")

    # Generación de Tensores
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

    print("[ETL 4/5] Generando matriz binaria de brotes (Clasificación)...")
    # Umbral por nodo para detectar brotes (percentil sobre casos históricos > 0)
    target_idx = config["model_params"]["target_idx"]
    target_series = data_tensor[:, :, target_idx]
    
    outbreak_matrix = np.zeros((T, N), dtype=np.int64)
    pct = config["data_processing"]["outbreak_percentile"]
    for n in range(N):
        node_vals = target_series[:, n]
        non_zero = node_vals[node_vals > 0]
        threshold = np.percentile(non_zero, pct) if len(non_zero) > 10 else 1.0
        outbreak_matrix[:, n] = (node_vals >= threshold).astype(np.int64)

    print("[ETL 5/5] Exportando artefactos procesados...")
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
    print("ETL finalizado con éxito en 'data/processed/'.")

if __name__ == "__main__":
    run_etl()