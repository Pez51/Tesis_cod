import os
import torch
import numpy as np
import geopandas as gpd
from scipy.spatial.distance import cdist
from src.utils import load_config, load_pickle

def build_graph():
    config = load_config()
    geo_path = config["paths"]["geo_file"]
    processed_dir = config["paths"]["processed_dir"]
    metadata_path = os.path.join(processed_dir, "metadata.pkl")

    print("[GRAFO 1/3] Cargando cartografía y catálogo provincial...")
    metadata = load_pickle(metadata_path)
    valid_ubigeos = metadata["ubigeos"]
    ubigeo_to_idx = metadata["ubigeo_to_idx"]
    N = len(valid_ubigeos)
    valid_set = set(valid_ubigeos)

    gdf_dist = gpd.read_file(geo_path)
    print(f" -> Columnas disponibles en la cartografía: {list(gdf_dist.columns)}")

    # 1. Búsqueda por nombres estándar
    prov_col = None
    common_names = [
        "ubigeo", "ubigeo_prov", "iddist", "idprov", "cod_dist", "cod_prov",
        "coddist", "codprov", "id_dist", "id_prov", "cod_distrito", "cod_provincia"
    ]
    for col in gdf_dist.columns:
        if col.lower() in common_names:
            prov_col = col
            break

    # 2. Búsqueda por descomposición INEI (CCDD + CCPP)
    cols_upper = {c.upper(): c for c in gdf_dist.columns}
    if prov_col is None and "CCDD" in cols_upper and "CCPP" in cols_upper:
        c_dep = cols_upper["CCDD"]
        c_prov = cols_upper["CCPP"]
        gdf_dist["ubigeo_prov"] = (
            gdf_dist[c_dep].astype(str).str.strip().str.zfill(2) +
            gdf_dist[c_prov].astype(str).str.strip().str.zfill(2)
        )
        prov_col = "ubigeo_prov"

    # 3. Detección heurística por coincidencia de valores contra valid_ubigeos
    if prov_col is None or prov_col != "ubigeo_prov":
        best_col = None
        best_matches = 0
        use_prefix = True

        for col in gdf_dist.columns:
            if col == "geometry":
                continue
            sample = gdf_dist[col].dropna().astype(str).str.strip()
            # Caso A: Códigos distritales de 6 dígitos -> tomar primeros 4
            matches_pfx = sample.str.zfill(6).str[:4].isin(valid_set).sum()
            # Caso B: Códigos provinciales directos de 4 dígitos
            matches_dir = sample.str.zfill(4).isin(valid_set).sum()

            max_m = max(matches_pfx, matches_dir)
            if max_m > best_matches:
                best_matches = max_m
                best_col = col
                use_prefix = (matches_pfx >= matches_dir)

        if best_matches > 0 and best_col is not None:
            print(f" -> Columna identificada por coincidencia demográfica: '{best_col}' ({best_matches} registros coincidentes)")
            if use_prefix:
                gdf_dist["ubigeo_prov"] = gdf_dist[best_col].astype(str).str.strip().str.zfill(6).str[:4]
            else:
                gdf_dist["ubigeo_prov"] = gdf_dist[best_col].astype(str).str.strip().str.zfill(4)
        else:
            raise ValueError(
                f"No se pudo identificar el código geográfico en las columnas: {list(gdf_dist.columns)}"
            )
    elif prov_col != "ubigeo_prov":
        raw_vals = gdf_dist[prov_col].astype(str).str.strip()
        if raw_vals.str.len().max() >= 6:
            gdf_dist["ubigeo_prov"] = raw_vals.str.zfill(6).str[:4]
        else:
            gdf_dist["ubigeo_prov"] = raw_vals.str.zfill(4)

    print("[GRAFO 2/3] Disolviendo polígonos a nivel provincial (196 nodos)...")
    gdf_prov = gdf_dist.dissolve(by="ubigeo_prov").reset_index()
    gdf_prov = gdf_prov[gdf_prov["ubigeo_prov"].isin(valid_set)].copy()

    # Mapeo de índices
    gdf_prov["node_idx"] = gdf_prov["ubigeo_prov"].map(ubigeo_to_idx)
    gdf_prov = gdf_prov.dropna(subset=["node_idx"]).copy()
    gdf_prov["node_idx"] = gdf_prov["node_idx"].astype(int)

    centroids = gdf_prov.geometry.centroid
    node_coords = np.zeros((N, 2), dtype=np.float32)
    for idx, geom_idx in zip(gdf_prov["node_idx"], centroids.index):
        node_coords[idx] = [centroids[geom_idx].x, centroids[geom_idx].y]

    print("[GRAFO 3/3] Construyendo matriz de adyacencia provincial (fronteras Queen y k-NN)...")
    edges_src = []
    edges_dst = []

    prov_list = gdf_prov.to_dict('records')
    for i_rec in prov_list:
        idx_i = i_rec["node_idx"]
        poly_i = i_rec["geometry"]
        for j_rec in prov_list:
            idx_j = j_rec["node_idx"]
            if idx_i != idx_j and poly_i.touches(j_rec["geometry"]):
                edges_src.append(idx_i)
                edges_dst.append(idx_j)

    # Conectividad de respaldo para provincias sin frontera terrestre inmediata
    dist_mat = cdist(node_coords, node_coords)
    np.fill_diagonal(dist_mat, np.inf)
    adj_set = set(zip(edges_src, edges_dst))

    for i in range(N):
        has_edges = any(src == i for src in edges_src)
        if not has_edges:
            closest_2 = np.argsort(dist_mat[i])[:2]
            for j in closest_2:
                edges_src.extend([i, int(j)])
                edges_dst.extend([int(j), i])
                adj_set.add((i, int(j)))
                adj_set.add((int(j), i))

    # Auto-bucles (self-loops) para asegurar la persistencia temporal local
    for i in range(N):
        edges_src.append(i)
        edges_dst.append(i)

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_index = torch.unique(edge_index, dim=1)

    graph_data = {
        "edge_index": edge_index,
        "num_nodes": N,
        "coords": node_coords,
        "ubigeos": valid_ubigeos
    }

    out_path = os.path.join(processed_dir, "graph_topology.pt")
    torch.save(graph_data, out_path)
    print(f"Topología provincial completada: {N} nodos y {edge_index.size(1)} aristas. Archivo guardado en {out_path}.")

if __name__ == "__main__":
    build_graph()