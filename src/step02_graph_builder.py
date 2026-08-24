import os
import urllib.request
import pickle
import numpy as np
import geopandas as gpd
import torch
from libpysal.weights import Queen
from scipy.spatial.distance import cdist
from src.utils import load_config

GEO_URL = "https://raw.githubusercontent.com/juaneladio/peru-geojson/master/peru_distrital_simple.geojson"

def download_geojson_if_missing(geo_path):
    os.makedirs(os.path.dirname(geo_path), exist_ok=True)
    if not os.path.exists(geo_path):
        print(f"[GEO] Descargando cartografía distrital...")
        try:
            urllib.request.urlretrieve(GEO_URL, geo_path)
            print(f"[GEO] Archivo guardado en: {geo_path}")
        except Exception as e:
            raise RuntimeError(f"Error al descargar la cartografía: {e}")
    else:
        print(f"[GEO] Cartografía encontrada en: {geo_path}")

def build_spatial_graph():
    config = load_config()
    geo_path = config["paths"]["geo_file"]
    processed_dir = config["paths"]["processed_dir"]

    # 1. Cartografía
    download_geojson_if_missing(geo_path)

    # 2. Metadatos
    metadata_path = os.path.join(processed_dir, "metadata.pkl")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError("No se encontró 'metadata.pkl'. Ejecuta primero 'step01_etl.py'.")
    
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    valid_ubigeos = metadata["ubigeos"]
    ubigeo_to_idx = metadata["ubigeo_to_idx"]
    num_nodes = len(valid_ubigeos)

    print(f"[GRAFO 1/4] Leyendo y reparando geometrías del GeoJSON...")
    gdf = gpd.read_file(geo_path)

    ubigeo_col = None
    for col in ["UBIGEO", "ubigeo", "IDDIST", "iddist"]:
        if col in gdf.columns:
            ubigeo_col = col
            break

    if ubigeo_col is None:
        raise KeyError("No se encontró la columna de UBIGEO en el GeoJSON.")

    gdf["ubigeo_clean"] = gdf[ubigeo_col].astype(str).str.strip().str.zfill(6)
    gdf = gdf[gdf.geometry.notnull() & (~gdf.geometry.is_empty)].copy()
    gdf["geometry"] = gdf["geometry"].buffer(0)

    gdf_filtered = gdf[gdf["ubigeo_clean"].isin(valid_ubigeos)].copy()
    gdf_filtered["node_idx"] = gdf_filtered["ubigeo_clean"].map(ubigeo_to_idx)
    gdf_filtered = gdf_filtered.drop_duplicates(subset=["node_idx"]).set_index("node_idx").sort_index()

    print(f"[GRAFO 2/4] Calculando matriz de contigüidad espacial (Queen)...")
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    edges_src = []
    edges_dst = []

    try:
        w_queen = Queen.from_dataframe(gdf_filtered, use_index=True)
        for i, neighbors in w_queen.neighbors.items():
            for j in neighbors:
                adj_matrix[i, j] = 1.0
                edges_src.append(int(i))
                edges_dst.append(int(j))
    except Exception:
        pass

    # Respaldo por centroides en proyección métrica
    gdf_projected = gdf_filtered.to_crs(epsg=32718)
    centroids = gdf_projected.geometry.centroid
    coords = np.array([[geom.x, geom.y] for geom in centroids])
    indices = gdf_filtered.index.to_numpy()

    if len(coords) > 1:
        dist_matrix = cdist(coords, coords)
        for idx_pos, i in enumerate(indices):
            if adj_matrix[i].sum() <= 0:
                nearest_positions = np.argsort(dist_matrix[idx_pos])[1:4]
                for n_pos in nearest_positions:
                    j = indices[n_pos]
                    adj_matrix[i, j] = 1.0
                    adj_matrix[j, i] = 1.0
                    edges_src.extend([int(i), int(j)])
                    edges_dst.extend([int(j), int(i)])

    # Autobucles
    for i in range(num_nodes):
        adj_matrix[i, i] = 1.0
        edges_src.append(i)
        edges_dst.append(i)

    print(f"[GRAFO 3/4] Generando tensores de PyTorch Geometric...")
    edge_pairs = set(zip(edges_src, edges_dst))
    clean_src = [pair[0] for pair in edge_pairs]
    clean_dst = [pair[1] for pair in edge_pairs]

    edge_index = torch.tensor([clean_src, clean_dst], dtype=torch.long)
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32)

    print(f"[GRAFO 4/4] Exportando topología a 'data/processed/'...")
    np.save(os.path.join(processed_dir, "adj_matrix.npy"), adj_matrix)
    np.save(os.path.join(processed_dir, "ubigeos.npy"), np.array(valid_ubigeos))

    torch.save({
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "num_nodes": num_nodes
    }, os.path.join(processed_dir, "graph_topology.pt"))

    print(f"Topología completada: {num_nodes} nodos y {edge_index.size(1)} aristas.")

if __name__ == "__main__":
    build_spatial_graph()