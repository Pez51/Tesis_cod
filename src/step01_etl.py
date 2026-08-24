import os
import polars as pl
import numpy as np
from src.utils import load_config, save_pickle

def run_etl():
    config = load_config()
    raw_path = config["paths"]["raw_csv"]
    processed_dir = config["paths"]["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)

    print("[ETL 1/5] Cargando dataset crudo...")
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

    print("[ETL 2/5] Casteando tipos y limpiando UBIGEOS...")
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

    valid_df = df_clean.filter(
        (pl.col("ano") >= start_y) & (pl.col("ano") <= end_y) &
        (pl.col("semana") >= 1) & (pl.col("semana") <= 53) &
        (pl.col("ubigeo").is_not_null()) &
        (pl.col("ubigeo").str.len_chars() == 6)
    )

    print("[ETL 3/5] Agregando registros por Distrito-Semana...")
    agg_df = valid_df.group_by(["ubigeo", "departamento", "provincia", "distrito", "ano", "semana"]).agg([
        pl.col("episodios_men5").sum().alias("ep_men5"),
        pl.col("hospitalizados_men5").sum().alias("hosp_men5"),
        pl.col("defunciones_men5").sum().alias("def_men5"),
        pl.col("episodios_may5").sum().alias("ep_may5"),
        pl.col("hospitalizados_may5").sum().alias("hosp_may5"),
        pl.col("defunciones_may5").sum().alias("def_may5"),
    ])

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

    # 1. Tensor base (T, N, 6)
    base_tensor = np.zeros((T, N, 6), dtype=np.float32)
    pdf = agg_df.to_pandas()

    for row in pdf.itertuples():
        t_i = time_to_idx.get((row.ano, row.semana))
        n_i = ubigeo_to_idx.get(row.ubigeo)
        if t_i is not None and n_i is not None:
            base_tensor[t_i, n_i, :] = [
                row.ep_men5, row.hosp_men5, row.def_men5,
                row.ep_may5, row.hosp_may5, row.def_may5
            ]

    print("[ETL 4/5] Generando ingenieria de caracteristicas (10 variables)...")
    data_tensor = np.zeros((T, N, F), dtype=np.float32)
    data_tensor[:, :, 0:6] = base_tensor

    # Feature 6: Delta casos (Velocidad de contagio)
    delta_cases = np.zeros((T, N), dtype=np.float32)
    delta_cases[1:] = base_tensor[1:, :, 0] - base_tensor[:-1, :, 0]
    data_tensor[:, :, 6] = delta_cases

    # Feature 7: Ratio de hospitalización
    data_tensor[:, :, 7] = base_tensor[:, :, 1] / (base_tensor[:, :, 0] + 1.0)

    # Features 8 y 9: Seno y Coseno de la semana epidemiológica
    for t_i, (y, s) in enumerate(time_index):
        data_tensor[t_i, :, 8] = np.sin(2 * np.pi * s / 53.0)
        data_tensor[t_i, :, 9] = np.cos(2 * np.pi * s / 53.0)

    # Matriz de Brotes
    target_series = data_tensor[:, :, 0]
    outbreak_matrix = np.zeros((T, N), dtype=np.int64)
    pct = config["data_processing"]["outbreak_percentile"]
    
    for n in range(N):
        node_vals = target_series[:, n]
        non_zero = node_vals[node_vals > 0]
        threshold = np.percentile(non_zero, pct) if len(non_zero) > 10 else 1.0
        outbreak_matrix[:, n] = (node_vals >= threshold).astype(np.int64)

    print("[ETL 5/5] Exportando matrices procesadas...")
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
    print(f"ETL finalizado con exito: Tensor final {data_tensor.shape} ({F} características).")

if __name__ == "__main__":
    run_etl()