import os
import polars as pl
import numpy as np
from src.utils import load_config, save_pickle

def run_etl():
    config = load_config()
    raw_path = config["paths"]["raw_csv"]
    processed_dir = config["paths"]["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)

    print("[ETL 1/5] Cargando dataset crudo del MINSA...")
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

    print("[ETL 2/5] Depurando y extrayendo código provincial (UBIGEO 4 dígitos)...")
    df_clean = df_raw.with_columns([
        pl.col("ano").cast(pl.Int32, strict=False),
        pl.col("semana").cast(pl.Int32, strict=False),
        pl.col("ubigeo").cast(pl.Utf8, strict=False).str.strip_chars().str.zfill(6),
        pl.col("departamento").cast(pl.Utf8, strict=False).str.strip_chars(),
        pl.col("provincia").cast(pl.Utf8, strict=False).str.strip_chars(),
        pl.col("episodios_men5").cast(pl.Float32, strict=False).fill_null(0.0),
        pl.col("hospitalizados_men5").cast(pl.Float32, strict=False).fill_null(0.0),
        pl.col("defunciones_men5").cast(pl.Float32, strict=False).fill_null(0.0),
        pl.col("episodios_may5").cast(pl.Float32, strict=False).fill_null(0.0),
        pl.col("hospitalizados_may5").cast(pl.Float32, strict=False).fill_null(0.0),
        pl.col("defunciones_may5").cast(pl.Float32, strict=False).fill_null(0.0),
    ]).with_columns([
        pl.col("ubigeo").str.slice(0, 4).alias("ubigeo_prov")
    ])

    start_y = config["data_processing"]["start_year"]
    end_y = config["data_processing"]["end_year"]

    valid_df = df_clean.filter(
        (pl.col("ano") >= start_y) & (pl.col("ano") <= end_y) &
        (pl.col("semana") >= 1) & (pl.col("semana") <= 53) &
        (pl.col("ubigeo_prov").is_not_null()) &
        (pl.col("ubigeo_prov").str.len_chars() == 4)
    )

    print("[ETL 3/5] Agregando registros a nivel Provincia-Semana...")
    agg_df = valid_df.group_by(["ubigeo_prov", "departamento", "provincia", "ano", "semana"]).agg([
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

    unique_provinces = agg_df.select(
        ["ubigeo_prov", "departamento", "provincia"]
    ).unique("ubigeo_prov").sort("ubigeo_prov")

    valid_ubigeos = unique_provinces["ubigeo_prov"].to_list()
    ubigeo_to_idx = {u: i for i, u in enumerate(valid_ubigeos)}
    N = len(valid_ubigeos)

    features = [
        "ep_men5", "hosp_men5", "def_men5", "ep_may5", "hosp_may5", "def_may5",
        "delta_ep_men5", "accel_ep_men5", "ratio_hosp_men5", "ratio_def_men5",
        "roll_mean_4", "roll_std_4", "surge_ratio", "sem_sin", "sem_cos",
        "lag_departamental_ep", "lag_2_ep"
    ]
    F = len(features)

    base_tensor = np.zeros((T, N, 6), dtype=np.float32)
    pdf = agg_df.to_pandas()

    for row in pdf.itertuples():
        t_i = time_to_idx.get((row.ano, row.semana))
        n_i = ubigeo_to_idx.get(row.ubigeo_prov)
        if t_i is not None and n_i is not None:
            base_tensor[t_i, n_i, :] = [
                row.ep_men5, row.hosp_men5, row.def_men5,
                row.ep_may5, row.hosp_may5, row.def_may5
            ]

    print(f"[ETL 4/5] Generando tensor de {F} variables epidemiológicas...")
    data_tensor = np.zeros((T, N, F), dtype=np.float32)
    data_tensor[:, :, 0:6] = base_tensor

    # Feature 6: Delta casos (Velocidad de propagación)
    delta_cases = np.zeros((T, N), dtype=np.float32)
    delta_cases[1:] = base_tensor[1:, :, 0] - base_tensor[:-1, :, 0]
    data_tensor[:, :, 6] = delta_cases

    # Feature 7: Aceleración epidémica (Segunda derivada: cambio en la velocidad)
    accel_cases = np.zeros((T, N), dtype=np.float32)
    accel_cases[1:] = delta_cases[1:] - delta_cases[:-1]
    data_tensor[:, :, 7] = accel_cases

    # Feature 8: Ratio de hospitalización
    data_tensor[:, :, 8] = base_tensor[:, :, 1] / (base_tensor[:, :, 0] + 1.0)

    # Feature 9: Tasa de letalidad provincial
    data_tensor[:, :, 9] = base_tensor[:, :, 2] / (base_tensor[:, :, 0] + 1.0)

    # Feature 10 y 11: Media y Desviación móvil de 4 semanas
    roll_mean = np.zeros((T, N), dtype=np.float32)
    roll_std = np.zeros((T, N), dtype=np.float32)
    for t in range(T):
        start_t = max(0, t - 4)
        roll_mean[t] = np.mean(base_tensor[start_t : t + 1, :, 0], axis=0)
        roll_std[t] = np.std(base_tensor[start_t : t + 1, :, 0], axis=0)
    data_tensor[:, :, 10] = roll_mean
    data_tensor[:, :, 11] = roll_std

    # Feature 12: Ratio de aceleración sobre la media (Surge ratio)
    data_tensor[:, :, 12] = base_tensor[:, :, 0] / (roll_mean + 1.0)

    # Features 13 y 14: Estacionalidad sen/cos
    for t_i, (y, s) in enumerate(time_index):
        data_tensor[t_i, :, 13] = np.sin(2 * np.pi * s / 53.0)
        data_tensor[t_i, :, 14] = np.cos(2 * np.pi * s / 53.0)

    # Feature 15: Retardo departamental regional (t-1)
    dep_dict = {}
    for idx, u in enumerate(valid_ubigeos):
        dep_code = u[:2]
        dep_dict.setdefault(dep_code, []).append(idx)

    spatial_lag = np.zeros((T, N), dtype=np.float32)
    for dep, indices in dep_dict.items():
        if len(indices) > 0:
            dep_cases = np.mean(base_tensor[:, indices, 0], axis=1, keepdims=True)
            spatial_lag[1:, indices] = dep_cases[:-1]
    data_tensor[:, :, 15] = spatial_lag

    # Feature 16: Retardo autorregresivo directo de 2 semanas (t-2)
    lag_2 = np.zeros((T, N), dtype=np.float32)
    lag_2[2:] = base_tensor[:-2, :, 0]
    data_tensor[:, :, 16] = lag_2

    # Canal Endémico Dinámico Semanal (Bortman / MINSA)
    outbreak_matrix = np.zeros((T, N), dtype=np.int64)
    endemic_channel_q3 = np.zeros((53, N), dtype=np.float32)
    target_series = data_tensor[:, :, 0]

    for s in range(1, 54):
        time_indices_s = [t_i for t_i, (y, sem) in enumerate(time_index) if sem == s]
        if len(time_indices_s) > 0:
            cases_s = target_series[time_indices_s, :]
            endemic_channel_q3[s - 1, :] = np.percentile(cases_s, 75, axis=0)

    for t_i, (y, s) in enumerate(time_index):
        q3_thresh = endemic_channel_q3[s - 1, :]
        actual_cases = target_series[t_i, :]
        is_outbreak = (actual_cases > q3_thresh) & (actual_cases >= 5.0)
        outbreak_matrix[t_i, :] = is_outbreak.astype(np.int64)

    district_thresholds = np.mean(endemic_channel_q3, axis=0)

    print("[ETL 5/5] Exportando tensores enriquecidos...")
    np.save(os.path.join(processed_dir, "tensor_eda_TNF.npy"), data_tensor)
    np.save(os.path.join(processed_dir, "targets_outbreak.npy"), outbreak_matrix)
    np.save(os.path.join(processed_dir, "endemic_channel_q3.npy"), endemic_channel_q3)
    np.save(os.path.join(processed_dir, "district_thresholds.npy"), district_thresholds)
    agg_df.write_parquet(os.path.join(processed_dir, "df_master_weekly.parquet"))

    metadata = {
        "ubigeos": valid_ubigeos,
        "ubigeo_to_idx": ubigeo_to_idx,
        "time_index": time_index,
        "features": features,
        "provinces_catalog": unique_provinces.to_dicts(),
        "endemic_channel_q3": endemic_channel_q3,
        "district_thresholds": district_thresholds
    }
    save_pickle(metadata, os.path.join(processed_dir, "metadata.pkl"))
    print(f"ETL Provincial completado: Tensor {data_tensor.shape} ({N} provincias, {T} semanas, {F} variables).")

if __name__ == "__main__":
    run_etl()