import os
import sys
import json
import urllib.request
import unicodedata

# Asegurar reconocimiento de la raíz del proyecto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.collections import PatchCollection
from matplotlib. patches import Polygon as MplPolygon
from scipy.stats import norm

from src.utils import load_config

# URLs públicas de la cartografía oficial del Perú (GeoJSON provincial y departamental)
URL_GEOJSON_PROVINCIAS = "https://raw.githubusercontent.com/juaneladio/peru-geojson/master/peru_provincial_simple.geojson"
URL_GEOJSON_DEPARTAMENTOS = "https://raw.githubusercontent.com/juaneladio/peru-geojson/master/peru_departamental_simple.geojson"

# Coordenadas exactas para provincias creadas recientemente o casos especiales
SPECIAL_PROV_COORDS = {
    "PUTUMAYO": (-72.6700, -2.4500),
    "DATEM DEL MARANON": (-76.6500, -4.4500),
    "LAURICOCHA": (-76.7000, -10.2000),
    "YAROWILCA": (-76.6000, -9.8000),
    "CANDARAVE": (-70.2500, -17.2600),
    "JORGE BASADRE": (-70.7300, -17.5800),
    "CALLAO": (-77.1181, -12.0566),
    "PROV CONST DEL CALLAO": (-77.1181, -12.0566),
}

DEPT_COORDS_FALLBACK = {
    "TUMBES": (-80.4515, -3.5669), "PIURA": (-80.6328, -5.1945),
    "LAMBAYEQUE": (-79.8409, -6.7714), "CAJAMARCA": (-78.5128, -7.1617),
    "LA LIBERTAD": (-79.0300, -8.1160), "ANCASH": (-77.5288, -9.5261),
    "LIMA": (-77.0428, -12.0464), "CALLAO": (-77.1181, -12.0566),
    "HUANUCO": (-76.2422, -9.9306), "PASCO": (-76.2567, -10.6675),
    "JUNIN": (-75.2049, -12.0651), "HUANCAVELICA": (-74.9764, -12.7862),
    "ICA": (-75.7286, -14.0678), "AYACUCHO": (-74.2236, -13.1631),
    "APURIMAC": (-72.8814, -13.6339), "CUSCO": (-71.9675, -13.5320),
    "AREQUIPA": (-71.5375, -16.4090), "MOQUEGUA": (-70.9357, -17.1938),
    "TACNA": (-70.2463, -18.0066), "PUNO": (-70.0219, -15.8402),
    "AMAZONAS": (-77.8690, -6.2317), "SAN MARTIN": (-76.3656, -6.4869),
    "LORETO": (-73.2516, -3.7437), "UCAYALI": (-74.5539, -8.3791),
    "MADRE DE DIOS": (-69.1891, -12.5933)
}


def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.upper().strip()
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    for ch in [".", ",", "-", "_"]:
        text = text.replace(ch, " ")
    return " ".join(text.split())


# -------------------------------------------------------------------------
# 1. CÁLCULO DEL ÍNDICE DE MORAN GLOBAL (I) CON PERMUTACIÓN DE MONTE CARLO
# -------------------------------------------------------------------------
def calcular_indice_moran_global(x, W, num_permutaciones=999, seed=42):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    z = x - np.mean(x)
    denom = np.sum(z ** 2)
    S0 = np.sum(W)

    if denom == 0 or S0 == 0:
        return 0.0, -1.0 / (N - 1), 0.0, 1.0

    num_obs = z @ (W @ z)
    I_obs = (N / S0) * (num_obs / denom)

    I_perm = np.zeros(num_permutaciones, dtype=np.float64)
    for p in range(num_permutaciones):
        z_p = rng.permutation(z)
        I_perm[p] = (N / S0) * ((z_p @ (W @ z_p)) / denom)

    E_I = -1.0 / (N - 1)
    std_perm = np.std(I_perm) + 1e-9
    z_score = (I_obs - np.mean(I_perm)) / std_perm
    p_val_twosided = 2.0 * (1.0 - norm.cdf(abs(z_score)))

    return float(I_obs), float(E_I), float(z_score), float(p_val_twosided)


def ejecutar_auditoria_moran(df_red, df_corredores, edge_index, num_nodes, reports_dir):
    print("\n" + "=" * 78)
    print(" [OE2 & OE3] AUTOCORRELACIÓN ESPACIAL GLOBAL DE MORAN (999 PERMUTACIONES)")
    print("=" * 78)

    W = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    for u, v in zip(src, dst):
        if u != v:
            W[u, v] = 1.0

    row_sums = W.sum(axis=1, keepdims=True)
    W = np.divide(W, row_sums, out=np.zeros_like(W), where=row_sums > 0)

    atencion_nodo = df_corredores.groupby("nodo_origen_idx")["peso_atencion_alfa"].mean().to_dict()
    df_red["atencion_gat_media"] = df_red["node_idx"].map(lambda idx: atencion_nodo.get(idx, 0.5))

    variables_evaluar = [
        ("Frecuencia de Brotes (% Semanas > Q3)", df_red["pct_semanas_brote"].values),
        ("Incidencia Promedio Semanal (Log-Casos)", np.log1p(df_red["casos_semanales_promedio"].values)),
        ("Centralidad de Intermediación (Betweenness)", df_red["intermediacion_betweenness"].values),
        ("Intensidad de Atención Espacial GAT (Alfa)", df_red["atencion_gat_media"].values),
    ]

    resultados_moran = []
    for nombre_var, valores in variables_evaluar:
        I_obs, E_I, z_sc, p_val = calcular_indice_moran_global(valores, W)
        sig = "Agrupamiento Significativo (p < 0.05)" if p_val < 0.05 else "Patrón Aleatorio"
        print(f" -> {nombre_var:<43} | Moran's I = {I_obs:+.4f} | Z = {z_sc:+.2f} | p = {p_val:.4e} [{sig}]")
        resultados_moran.append({
            "Variable_Espacial": nombre_var,
            "Moran_I": round(I_obs, 4),
            "Esperado_EI": round(E_I, 4),
            "Z_Score": round(z_sc, 4),
            "p_value": p_val,
            "Interpretacion": sig
        })

    df_moran = pd.DataFrame(resultados_moran)
    out_csv = os.path.join(reports_dir, "tabla_autocorrelacion_espacial_moran.csv")
    df_moran.to_csv(out_csv, index=False)
    print(f"\n[OK] Tabla de Índice de Moran exportada en: {out_csv}")
    return df_moran


# -------------------------------------------------------------------------
# 2. CARGA O DESCARGA AUTOMÁTICA DE CARTOGRAFÍA GEOJSON DEL PERÚ
# -------------------------------------------------------------------------
def descargar_o_cargar_geojson(url, local_path):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if not os.path.exists(local_path):
        print(f"[CARTOGRAFÍA] Descargando mapa oficial del Perú: {os.path.basename(local_path)}...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as response, open(local_path, "wb") as out_file:
            out_file.write(response.read())
    with open(local_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extraer_anillos_y_centroide(geometry):
    """Extrae listas de polígonos (lon, lat) y calcula el centroide geométrico del anillo principal."""
    if not geometry:
        return [], None
    gtype = geometry.get("type", "")
    coords = geometry.get("coordinates", [])
    rings = []

    if gtype == "Polygon":
        if len(coords) > 0:
            rings.append(np.array(coords[0], dtype=np.float64))
    elif gtype == "MultiPolygon":
        for poly in coords:
            if len(poly) > 0 and len(poly[0]) > 0:
                rings.append(np.array(poly[0], dtype=np.float64))

    if not rings:
        return [], None

    # Centroide del anillo exterior de mayor tamaño
    largest_ring = max(rings, key=lambda r: len(r))
    centroid_lon = float(np.mean(largest_ring[:, 0]))
    centroid_lat = float(np.mean(largest_ring[:, 1]))
    return rings, (centroid_lon, centroid_lat)


def construir_cartografia_peru(df_red):
    ext_dir = os.path.join("data", "external")
    prov_geojson_path = os.path.join(ext_dir, "peru_provincial_simple.geojson")
    dep_geojson_path = os.path.join(ext_dir, "peru_departamental_simple.geojson")

    prov_data = None
    dep_data = None
    try:
        prov_data = descargar_o_cargar_geojson(URL_GEOJSON_PROVINCIAS, prov_geojson_path)
        dep_data = descargar_o_cargar_geojson(URL_GEOJSON_DEPARTAMENTOS, dep_geojson_path)
        print("[CARTOGRAFÍA] Polígonos provinciales y departamentales del Perú cargados exitosamente.")
    except Exception as e:
        print(f"[AVISO] No se pudo descargar GeoJSON ({e}). Verifique conexión a internet.")

    # Diccionarios de emparejamiento desde el GeoJSON provincial
    ubigeo_to_centroid = {}
    name_to_centroid = {}
    prov_polygons_list = []  # Lista de tuplas: (rings, ubigeo_4d, prov_norm, dep_norm)

    if prov_data and "features" in prov_data:
        for feat in prov_data["features"]:
            props = feat.get("properties", {})
            rings, centroid = extraer_anillos_y_centroide(feat.get("geometry"))
            if not rings or centroid is None:
                continue

            # Extraer código Ubigeo provincial (4 dígitos)
            ubi_raw = (
                props.get("FIRST_IDPR")
                or props.get("IDPROV")
                or props.get("UBIGEO")
                or props.get("COD_PROV")
                or ""
            )
            ubi_4d = str(ubi_raw).strip().zfill(4)[:4] if ubi_raw else ""

            # Extraer nombres de provincia y departamento
            p_name = normalize_text(props.get("NOMBPROV") or props.get("NOMPROV") or props.get("PROVINCIA") or "")
            d_name = normalize_text(props.get("NOMBDEP") or props.get("NOMDEP") or props.get("DEPARTAMEN") or "")

            if ubi_4d:
                ubigeo_to_centroid[ubi_4d] = centroid
            if p_name:
                name_to_centroid[p_name] = centroid
                if d_name:
                    name_to_centroid[f"{d_name}_{p_name}"] = centroid

            prov_polygons_list.append((rings, ubi_4d, p_name, d_name))

    # Extraer anillos de los 25 departamentos para trazar fronteras departamentales nítidas
    dep_rings_list = []
    if dep_data and "features" in dep_data:
        for feat in dep_data["features"]:
            rings, _ = extraer_anillos_y_centroide(feat.get("geometry"))
            dep_rings_list.extend(rings)

    # Asignar coordenadas exactas a cada uno de los 196 nodos del grafo
    coords_nodos = {}
    matched_count = 0
    rng = np.random.default_rng(42)

    for _, row in df_red.iterrows():
        idx = int(row["node_idx"])
        u_str = str(row["ubigeo"]).strip().zfill(4)[:4]
        p_norm = normalize_text(str(row["provincia"]))
        d_norm = normalize_text(str(row["departamento"]))

        if u_str in ubigeo_to_centroid:
            coords_nodos[idx] = ubigeo_to_centroid[u_str]
            matched_count += 1
        elif f"{d_norm}_{p_norm}" in name_to_centroid:
            coords_nodos[idx] = name_to_centroid[f"{d_norm}_{p_norm}"]
            matched_count += 1
        elif p_norm in name_to_centroid:
            coords_nodos[idx] = name_to_centroid[p_norm]
            matched_count += 1
        elif p_norm in SPECIAL_PROV_COORDS:
            coords_nodos[idx] = SPECIAL_PROV_COORDS[p_norm]
            matched_count += 1
        else:
            # Coincidencia parcial por subcadena
            found_sub = False
            for k_name, c_val in name_to_centroid.items():
                if p_norm and (p_norm in k_name or k_name in p_norm):
                    coords_nodos[idx] = c_val
                    matched_count += 1
                    found_sub = True
                    break
            if not found_sub:
                lon_b, lat_b = DEPT_COORDS_FALLBACK.get(d_norm, (-75.0, -10.0))
                coords_nodos[idx] = (lon_b + rng.uniform(-0.25, 0.25), lat_b + rng.uniform(-0.25, 0.25))

    print(f"[CARTOGRAFÍA] Provincias georreferenciadas sobre polígonos reales: {matched_count}/{len(df_red)}")
    return prov_polygons_list, dep_rings_list, coords_nodos


# -------------------------------------------------------------------------
# 3. RENDERIZADO DE LA FIGURA 11 CON EL MAPA DEL PERÚ EN EL FONDO
# -------------------------------------------------------------------------
def generar_mapa_cartografico_peru(df_red, df_corredores, edge_index, prov_polygons, dep_rings, coords, figures_dir):
    print("\n[MAPA GEOESPACIAL] Renderizando Figura 11 con fondo cartográfico del Perú (300 DPI)...")
    fig, axes = plt.subplots(1, 2, figsize=(19, 11.5))
    ax1, ax2 = axes[0], axes[1]

    # Mapear tasa de brote por ubigeo y nombre de provincia para colorear el mapa coroplético de fondo
    brote_by_ubigeo = {}
    brote_by_prov = {}
    for _, row in df_red.iterrows():
        u_str = str(row["ubigeo"]).strip().zfill(4)[:4]
        p_norm = normalize_text(str(row["provincia"]))
        val = float(row["pct_semanas_brote"])
        brote_by_ubigeo[u_str] = val
        brote_by_prov[p_norm] = val

    vmin_brote = float(df_red["pct_semanas_brote"].min())
    vmax_brote = float(df_red["pct_semanas_brote"].max())
    cmap_choropleth = plt.cm.YlOrRd

    # 1. DIBUJAR POLÍGONOS PROVINCIALES DEL PERÚ EN AMBOS PANELES
    if prov_polygons:
        patches_ax1 = []
        colors_ax1 = []
        patches_ax2 = []

        for rings, ubi_4d, p_name, _ in prov_polygons:
            val_brote = brote_by_ubigeo.get(ubi_4d, brote_by_prov.get(p_name, np.mean(df_red["pct_semanas_brote"])))
            norm_val = np.clip((val_brote - vmin_brote) / (vmax_brote - vmin_brote + 1e-6), 0.0, 1.0)

            for ring in rings:
                poly1 = MplPolygon(ring, closed=True)
                patches_ax1.append(poly1)
                colors_ax1.append(cmap_choropleth(norm_val * 0.55 + 0.05))  # Tono suave coroplético en el fondo

                poly2 = MplPolygon(ring, closed=True)
                patches_ax2.append(poly2)

        # Colección coroplética suave en Panel A
        pc1 = PatchCollection(
            patches_ax1, facecolors=colors_ax1, edgecolors="#bcaaa4", linewidths=0.35, alpha=0.55, zorder=1
        )
        ax1.add_collection(pc1)

        # Colección cartográfica limpia en Panel B (territorio peruano en tono crema/gris claro)
        pc2 = PatchCollection(
            patches_ax2, facecolors="#f5f3ef", edgecolors="#cbd5e1", linewidths=0.40, alpha=0.95, zorder=1
        )
        ax2.add_collection(pc2)

    # 2. DIBUJAR LÍMITES DEPARTAMENTALES MÁS MARCADOS PARA DIMENSIONAR LAS REGIONES DEL PERÚ
    if dep_rings:
        for ring in dep_rings:
            ax1.plot(ring[:, 0], ring[:, 1], color="#64748b", linewidth=0.85, alpha=0.75, zorder=2)
            ax2.plot(ring[:, 0], ring[:, 1], color="#64748b", linewidth=0.90, alpha=0.80, zorder=2)

    # Configurar estética de ejes y océano
    for ax in [ax1, ax2]:
        ax.set_facecolor("#dceefb")  # Color océano Pacífico / entorno
        ax.set_xlim(-81.8, -68.2)
        ax.set_ylim(-18.7, 0.3)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Longitud Geográfica (°O)", fontsize=10.5, fontweight="bold")
        ax.set_ylabel("Latitud Geográfica (°S)", fontsize=10.5, fontweight="bold")
        ax.grid(True, linestyle="--", color="#94a3b8", alpha=0.35)
        ax.text(-80.8, -15.5, "OCÉANO\nPACÍFICO", fontsize=9.5, fontweight="bold", color="#3b82f6", alpha=0.55, ha="center", style="italic")

    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()

    # 3. DIBUJAR ARISTAS DE LA RED QUEEN (196 PROVINCIAS)
    for u, v in zip(src, dst):
        if u < v and u in coords and v in coords:
            x_vals = [coords[u][0], coords[v][0]]
            y_vals = [coords[u][1], coords[v][1]]
            ax1.plot(x_vals, y_vals, color="#475569", alpha=0.32, linewidth=0.65, zorder=3)
            ax2.plot(x_vals, y_vals, color="#94a3b8", alpha=0.30, linewidth=0.55, zorder=3)

    # ---------------------------------------------------------------------
    # PANEL A: TOPOLOGÍA DE RED PROVINCIAL Y NODOS PUENTE (OE3)
    # ---------------------------------------------------------------------
    lons = [coords[int(i)][0] for i in df_red["node_idx"]]
    lats = [coords[int(i)][1] for i in df_red["node_idx"]]
    brotes_pct = df_red["pct_semanas_brote"].values
    bet_cent = df_red["intermediacion_betweenness"].values

    sizes = 30 + (bet_cent / (np.max(bet_cent) + 1e-6)) * 290

    sc1 = ax1.scatter(
        lons, lats, c=brotes_pct, s=sizes, cmap="YlOrRd",
        vmin=vmin_brote, vmax=vmax_brote,
        edgecolors="#1e293b", linewidth=0.7, alpha=0.92, zorder=4
    )
    cbar1 = plt.colorbar(sc1, ax=ax1, fraction=0.038, pad=0.025)
    cbar1.set_label("% de Semanas en Brote Epidémico (> Q3)", fontsize=10, fontweight="bold")

    # Etiquetar el Top 6 de Nodos Puente (Betweenness Centrality) con desplazamientos limpios
    top_bridges = df_red.sort_values(by="intermediacion_betweenness", ascending=False).head(6)
    offsets_bridges = [(1.1, 0.4), (1.2, -0.5), (1.1, -0.7), (1.1, 0.5), (1.0, 0.35), (1.1, -0.4)]
    for rank_b, (_, row) in enumerate(top_bridges.iterrows()):
        idx = int(row["node_idx"])
        x, y = coords[idx]
        dx, dy = offsets_bridges[rank_b % len(offsets_bridges)]
        ax1.annotate(
            f"{row['provincia']} ({row['departamento']})\n[Betweenness = {row['intermediacion_betweenness']:.2f}]",
            xy=(x, y), xytext=(x + dx, y + dy),
            fontsize=7.8, fontweight="bold", color="#1e293b",
            bbox=dict(boxstyle="round,pad=0.25", fc="#fef9c3", ec="#d97706", lw=1.0, alpha=0.95),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.15", color="#b45309", lw=1.2),
            zorder=6
        )

    ax1.set_title(
        "A) Mapa de Vulnerabilidad Provincial y Nodos Puente en el Perú (OE3)\n"
        "[Polígonos/Color: % Semanas en Brote | Tamaño Nodal: Centralidad de Intermediación]",
        fontsize=11.5, fontweight="bold", pad=10
    )

    # ---------------------------------------------------------------------
    # PANEL B: CORREDORES CRÍTICOS DE ATENCIÓN ESPACIAL GAT (OE2)
    # ---------------------------------------------------------------------
    ax2.scatter(lons, lats, c="#334155", s=24, edgecolors="white", linewidth=0.4, alpha=0.70, zorder=4)

    # Seleccionar los Top 25 corredores únicos
    pares_vistos = set()
    corredores_unicos = []
    for _, row in df_corredores.iterrows():
        u, v = int(row["nodo_origen_idx"]), int(row["nodo_destino_idx"])
        par = tuple(sorted((u, v)))
        if par not in pares_vistos:
            pares_vistos.add(par)
            corredores_unicos.append(row)
        if len(corredores_unicos) >= 25:
            break

    offsets_corr = [(-3.8, -0.6), (1.1, -0.8), (1.0, 0.65), (1.0, -0.8), (0.8, -0.9)]
    for rank, row in enumerate(corredores_unicos):
        u, v = int(row["nodo_origen_idx"]), int(row["nodo_destino_idx"])
        if u in coords and v in coords:
            x1, y1 = coords[u]
            x2, y2 = coords[v]
            es_inter = bool(row["es_interdepartamental"])
            color_linea = "#dc2626" if es_inter else "#2563eb"
            grosor = 2.4 if rank < 8 else 1.5

            ax2.annotate(
                "", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.12", color=color_linea, lw=grosor, alpha=0.92),
                zorder=5
            )
            ax2.scatter([x1, x2], [y1, y2], c=color_linea, s=58, edgecolors="white", linewidth=0.8, zorder=6)

            if rank < 5:
                xm, ym = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                dx, dy = offsets_corr[rank]
                origen_corto = str(row["provincia_origen"]).split("(")[0].strip()
                dest_corto = str(row["provincia_destino"]).split("(")[0].strip()
                ax2.annotate(
                    f"#{rank+1}: {origen_corto} ↔ {dest_corto}\n(α = {row['peso_atencion_alfa']:.3f})",
                    xy=(xm, ym), xytext=(xm + dx, ym + dy),
                    fontsize=7.8, fontweight="bold", color="#0f172a",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color_linea, lw=1.1, alpha=0.95),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color=color_linea, lw=1.0),
                    zorder=7
                )

    leyenda_inter = mlines.Line2D([], [], color="#dc2626", lw=2.5, marker="o", label="Corredor GAT Interdepartamental (Alta Transmisión)")
    leyenda_intra = mlines.Line2D([], [], color="#2563eb", lw=2.2, marker="o", label="Corredor GAT Intradepartamental")
    leyenda_dep = mlines.Line2D([], [], color="#64748b", lw=1.2, label="Límites Departamentales del Perú")
    leyenda_grafo = mlines.Line2D([], [], color="#94a3b8", lw=0.8, label="Aristas de Vecindad Queen (196 Provincias)")
    ax2.legend(handles=[leyenda_inter, leyenda_intra, leyenda_dep, leyenda_grafo], loc="lower left", frameon=True, facecolor="white", framealpha=0.95, fontsize=8.8)

    ax2.set_title(
        "B) Corredores Críticos de Propagación Aprendidos por GAT sobre el Perú (OE2)\n"
        "[Vectores Dirigidos según Coeficiente de Atención Espacial α_ij]",
        fontsize=11.5, fontweight="bold", pad=10
    )

    plt.tight_layout()
    out_fig = os.path.join(figures_dir, "11_mapa_cartografico_red_corredores_peru.png")
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"[OK] Figura 11 con fondo cartográfico del Perú exportada en: {out_fig}")


# -------------------------------------------------------------------------
# 4. FUNCIÓN PRINCIPAL
# -------------------------------------------------------------------------
def main():
    config = load_config()
    processed_dir = config["paths"]["processed_dir"]
    reports_dir = "reports"
    figures_dir = os.path.join(reports_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    red_csv = os.path.join(reports_dir, "tabla_metricas_red_oe3.csv")
    corr_csv = os.path.join(reports_dir, "tabla_corredores_atencion_oe2.csv")

    if not os.path.exists(red_csv) or not os.path.exists(corr_csv):
        raise FileNotFoundError("Ejecuta primero: python -m src.step07_spatial_explainability_network_metrics")

    df_red = pd.read_csv(red_csv)
    df_corredores = pd.read_csv(corr_csv)

    graph_data = torch.load(os.path.join(processed_dir, "graph_topology.pt"), weights_only=False)
    edge_index = graph_data["edge_index"]
    num_nodes = graph_data["num_nodes"]

    # 1. Auditoría de Índice de Moran Global
    ejecutar_auditoria_moran(df_red, df_corredores, edge_index, num_nodes, reports_dir)

    # 2. Construir cartografía oficial del Perú y renderizar Figura 11
    prov_polygons, dep_rings, coords = construir_cartografia_peru(df_red)
    generar_mapa_cartografico_peru(df_red, df_corredores, edge_index, prov_polygons, dep_rings, coords, figures_dir)

    print("\n" + "=" * 78)
    print(" MAPA CARTOGRÁFICO DEL PERÚ Y AUDITORÍA MORAN COMPLETADOS AL 100%")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()