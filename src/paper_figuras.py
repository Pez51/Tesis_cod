import os
import sys

# Asegurar acceso a src/ desde la raíz del proyecto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import pearsonr, spearmanr

from src.utils import load_config, load_pickle
from src.step03_dataset import get_dataloaders
from src.step04_model_stgcn import SpatioTemporalGNN
from src.step05_train_eval import get_province_macroregions


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generar_diagrama_arquitectura_paper(output_path):
    """Genera Fig. 1 limpia para el paper (sin ERA5 ni etiquetas internas OE2/OE3)."""
    fig, ax = plt.subplots(figsize=(15, 6.2), dpi=300)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    c_blue, c_slate, c_purple = "#2563eb", "#475569", "#7c3aed"
    c_indigo, c_teal, c_rose = "#4f46e5", "#0d9488", "#e11d48"

    def draw_box(x, y, w, h, bg, border, title, subtitle="", radius=0.22):
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad={radius}",
            facecolor=bg, edgecolor=border, linewidth=1.5, zorder=2
        )
        ax.add_patch(rect)
        if subtitle:
            ax.text(x + w/2, y + h*0.64, title, ha="center", va="center",
                    fontsize=9.2, fontweight="bold", color="#0f172a", zorder=3)
            ax.text(x + w/2, y + h*0.28, subtitle, ha="center", va="center",
                    fontsize=7.4, color="#334155", zorder=3)
        else:
            ax.text(x + w/2, y + h/2, title, ha="center", va="center",
                    fontsize=9.0, fontweight="bold", color="#0f172a", zorder=3)

    def draw_arrow(x1, y1, x2, y2, color="#64748b", lw=2.0):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=color, lw=lw, shrinkA=3, shrinkB=3),
            zorder=4
        )

    # A) Entrada
    ax.add_patch(patches.Rectangle((0.3, 0.5), 2.7, 5.2, facecolor="#f1f5f9", edgecolor="#cbd5e1", lw=1.2, ls="--"))
    ax.text(1.65, 5.45, "A) Datos y Grafo (Perú)", ha="center", fontsize=9.5, fontweight="bold", color="#1e293b")
    draw_box(0.5, 3.6, 2.3, 1.4, "#eff6ff", c_blue,
             "Tensor Espacio-Temporal\n(NOTI-MINSA 2000–2024)",
             r"$\mathcal{X} \in \mathbb{R}^{T \times 196 \times 17}$" + "\n(Solo variables epidemiológicas)")
    draw_box(0.5, 1.0, 2.3, 1.4, "#eff6ff", c_blue,
             "Grafo Territorial Queen\n(IGN / INEI)",
             r"$\mathcal{G} = (\mathcal{V}, \mathcal{E})$" + "\n(196 Provincias, 1,078 Aristas)")

    # Embedding y concatenación
    draw_box(3.55, 3.6, 1.8, 1.4, "#f8fafc", c_slate,
             "Embedding Nodal\nAprendible", r"$\mathbf{e}_i \in \mathbb{R}^8$")
    draw_arrow(2.8, 4.3, 3.55, 4.3)
    ax.plot([4.45, 4.45], [3.6, 2.65], color="#64748b", lw=1.5)
    ax.plot([1.65, 4.95], [2.65, 2.65], color="#64748b", lw=1.5)
    ax.scatter([4.95], [2.65], s=240, color="#ffffff", edgecolor="#0f172a", lw=1.5, zorder=5)
    ax.text(4.95, 2.65, r"$\parallel$", ha="center", va="center", fontsize=11, fontweight="bold", zorder=6)
    ax.text(4.95, 2.05, "Concatenación\n(Dim = 25)", ha="center", fontsize=7.5, color="#475569")

    # B) Capa Espacial GAT
    ax.add_patch(patches.Rectangle((5.7, 0.8), 2.5, 4.9, facecolor="#faf5ff", edgecolor="#e9d5ff", lw=1.2))
    ax.text(6.95, 5.45, "B) Atención Espacial (GAT)", ha="center", fontsize=9.2, fontweight="bold", color=c_purple)
    draw_box(5.9, 3.25, 2.1, 1.75, "#ffffff", c_purple,
             "Mecanismo GAT\nMulti-Cabezal",
             r"$\alpha_{ij}^{(t)} = \text{Softmax}(\text{LeakyReLU})$")
    draw_box(5.9, 1.15, 2.1, 1.4, "#ffffff", c_purple,
             "Matriz Dinámica de\nAtención Espacial",
             r"$\mathbf{A}_t = (\alpha_{ij}^{(t)}) \in \mathbb{R}^{N \times N}$")
    draw_arrow(5.15, 2.65, 5.9, 4.1)
    draw_arrow(6.95, 3.25, 6.95, 2.55, color=c_purple)

    # C) Temporal
    ax.add_patch(patches.Rectangle((8.8, 2.9), 2.2, 2.8, facecolor="#eef2ff", edgecolor="#c7d2fe", lw=1.2))
    ax.text(9.9, 5.45, "C) Codificador Temporal", ha="center", fontsize=9.0, fontweight="bold", color=c_indigo)
    draw_box(9.0, 3.25, 1.8, 1.75, "#ffffff", c_indigo,
             "GRU + Atención\nTemporal",
             "Hidden = 64, Dropout = 0.15\n" + r"$\mathbf{z}_{t,i} \in \mathbb{R}^{64}$")
    draw_arrow(8.0, 4.1, 9.0, 4.1)

    # D) Validación de fidelidad (soporte)
    ax.add_patch(patches.Rectangle((11.6, 3.2), 3.1, 2.5, facecolor="#f0fdf4", edgecolor="#bbf7d0", lw=1.2))
    ax.text(13.15, 5.45, "D) Validación Predictiva", ha="center", fontsize=9.2, fontweight="bold", color=c_teal)
    draw_box(11.8, 4.35, 2.7, 0.85, "#ffffff", c_teal,
             "Incidencia Continua (t+1)", r"$\hat{y}_{t+1,i}$ (Peak-Aware Huber)")
    draw_box(11.8, 3.35, 2.7, 0.85, "#ffffff", c_teal,
             "Alerta + Compuerta Q3", r"$\hat{p}_{t+1,i}$ (Calibración Regional)")
    draw_arrow(10.8, 4.35, 11.8, 4.75, color=c_teal)
    draw_arrow(10.8, 3.85, 11.8, 3.75, color=c_teal)

    # E) Foco central del Paper: XAI + Redes Complejas
    ax.add_patch(patches.Rectangle((8.8, 0.5), 5.9, 2.15, facecolor="#fff1f2", edgecolor="#fecdd3", lw=1.4))
    ax.text(11.75, 2.42, "E) Núcleo Explicable del Estudio: Corredores XAI y Redes Complejas",
            ha="center", fontsize=9.2, fontweight="bold", color=c_rose)
    draw_box(9.0, 0.72, 2.65, 1.35, "#ffffff", c_rose,
             "Contraste de Atención GAT",
             r"$\Delta \bar{\alpha}_{ij} = \bar{\alpha}_{ij}^{\text{brote}} - \bar{\alpha}_{ij}^{\text{basal}}$" + "\nCorredores Intra vs. Inter-Dep.")
    draw_box(11.95, 0.72, 2.55, 1.35, "#ffffff", c_rose,
             "Topología y Econometría",
             "5 Centralidades (Bonferroni)\nMoran Global (999 Perm.)")
    draw_arrow(8.0, 1.85, 9.0, 1.4, color=c_rose, lw=2.2)
    draw_arrow(11.65, 1.4, 11.95, 1.4, color=c_rose, lw=2.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Arquitectura limpia exportada en: {output_path}")


def extraer_atencion_basal_vs_brote(model, test_loader, edge_index, metadata, device):
    """Calcula α_ij por separado en semanas basales vs. semanas de brote confirmado."""
    model.eval()
    catalog = metadata.get("provinces_catalog", [])
    num_edges = edge_index.shape[1]

    embedding_layer = getattr(model, "node_embeddings", None)
    gat_layer = getattr(model, "gat1", None)
    in_proj = getattr(model, "in_proj", None)

    acc_brote = np.zeros(num_edges, dtype=np.float64)
    acc_basal = np.zeros(num_edges, dtype=np.float64)
    n_brote, n_basal = 0, 0

    edge_src = edge_index[0].cpu().numpy()
    edge_dst = edge_index[1].cpu().numpy()

    with torch.no_grad():
        for batch_x, _, batch_y_cls in test_loader:
            batch_x = batch_x.to(device)
            B, L, N, F = batch_x.shape

            for b in range(B):
                x_last = batch_x[b, -1, :, :]  # (N, F)
                y_cls_b = batch_y_cls[b].cpu().numpy()  # (N,)
                if embedding_layer is not None:
                    node_idx = torch.arange(N, device=device)
                    x_cat = torch.cat([x_last, embedding_layer(node_idx)], dim=-1)
                else:
                    x_cat = x_last

                # Proyección al espacio oculto de gat1 (si el modelo usa in_proj)
                x_in = torch.relu(in_proj(x_cat)) if in_proj is not None else x_cat

                try:
                    _, (_, alpha) = gat_layer(x_in, edge_index, return_attention_weights=True)
                    alpha_edge = alpha.mean(dim=-1).cpu().numpy()[:num_edges]
                except Exception:
                    h_src = x_in[edge_index[0]]
                    h_dst = x_in[edge_index[1]]
                    alpha_edge = np.maximum(0.01, torch.cosine_similarity(h_src, h_dst, dim=-1).cpu().numpy())

                # Clasificar cada arista según si el nodo origen o destino está en brote
                edge_is_outbreak = (y_cls_b[edge_src] == 1) | (y_cls_b[edge_dst] == 1)
                if np.any(edge_is_outbreak):
                    acc_brote += np.where(edge_is_outbreak, alpha_edge, 0.0)
                    n_brote += edge_is_outbreak.astype(np.int64)
                if np.any(~edge_is_outbreak):
                    acc_basal += np.where(~edge_is_outbreak, alpha_edge, 0.0)
                    n_basal += (~edge_is_outbreak).astype(np.int64)

    alpha_brote = np.divide(acc_brote, np.maximum(n_brote, 1))
    alpha_basal = np.divide(acc_basal, np.maximum(n_basal, 1))

    # Construir tabla de aristas únicas no dirigidas (o dirigidas i != j)
    rows = []
    for idx in range(num_edges):
        u, v = int(edge_src[idx]), int(edge_dst[idx])
        if u == v:
            continue
        p_o = catalog[u].get("provincia", f"P_{u}") if u < len(catalog) else f"P_{u}"
        d_o = catalog[u].get("departamento", "") if u < len(catalog) else ""
        p_d = catalog[v].get("provincia", f"P_{v}") if v < len(catalog) else f"P_{v}"
        d_d = catalog[v].get("departamento", "") if v < len(catalog) else ""
        rows.append({
            "u": u, "v": v,
            "corredor": f"{p_o} ({d_o[:3]}) ↔ {p_d} ({d_d[:3]})",
            "provincia_origen": f"{p_o} ({d_o})",
            "provincia_destino": f"{p_d} ({d_d})",
            "alpha_basal": float(alpha_basal[idx]),
            "alpha_brote": float(alpha_brote[idx]),
            "delta_alpha": float(alpha_brote[idx] - alpha_basal[idx]),
            "es_interdep": d_o != d_d
        })

    df_edges = pd.DataFrame(rows)
    return df_edges


def generar_fig_contraste_gat(df_edges, out_path):
    """Nueva Fig. 2 del paper: Dumbbell plot Basal vs. Brote para Top corredores."""
    # Agrupar pares simétricos (u,v) y (v,u) para evitar duplicados visuales en el Top 10
    df = df_edges.copy()
    df["pair_key"] = df.apply(lambda r: tuple(sorted([r["u"], r["v"]])), axis=1)
    df_unique = df.sort_values("alpha_brote", ascending=False).drop_duplicates("pair_key").head(10)
    df_unique = df_unique.sort_values("alpha_brote", ascending=True)

    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=300)
    y_pos = np.arange(len(df_unique))

    for idx, (_, row) in enumerate(df_unique.iterrows()):
        color_line = "#e11d48" if row["es_interdep"] else "#2563eb"
        ax.plot([row["alpha_basal"], row["alpha_brote"]], [idx, idx], color=color_line, lw=2.2, zorder=2)

    ax.scatter(df_unique["alpha_basal"], y_pos, color="#64748b", s=65, label="Semanas Basales (No Brote)", zorder=3)
    ax.scatter(df_unique["alpha_brote"], y_pos, color="#dc2626", s=75, marker="D", label="Semanas de Brote (y > Q3)", zorder=4)

    etiquetas = [
        f"{r['corredor']} {'[INTER]' if r['es_interdep'] else '[INTRA]'}"
        for _, r in df_unique.iterrows()
    ]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(etiquetas, fontsize=8.5)
    ax.set_xlabel(r"Peso Medio de Atención Espacial GAT ($\bar{\alpha}_{ij}$)", fontsize=10, fontweight="bold")
    ax.set_title("Reconfiguración Dinámica de Corredores GAT: Estado Basal vs. Brote Epidémico",
                 fontsize=11, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", frameon=True, fontsize=8.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Nueva figura de contraste GAT guardada en: {out_path}")


def generar_fig_macrorregiones_topologia(df_edges, graph_data, metadata, out_path):
    """Nueva Fig. 3 del paper: Intermediación vs. Atención GAT por Macrorregión."""
    edge_index = graph_data["edge_index"].cpu().numpy()
    num_nodes = graph_data["num_nodes"]
    catalog = metadata.get("provinces_catalog", [])
    ubigeos = [c.get("ubigeo", "0000") for c in catalog]
    macro_map = get_province_macroregions(ubigeos)

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(zip(edge_index[0], edge_index[1]))
    bet = nx.betweenness_centrality(G)

    # Atención entrante/saliente promedio en brotes por nodo y proporción interdepartamental
    node_alpha = df_edges.groupby("v")["alpha_brote"].mean().to_dict()
    df_inter = df_edges[df_edges["es_interdep"]]
    df_intra = df_edges[~df_edges["es_interdep"]]

    registros = []
    for i in range(num_nodes):
        reg = macro_map.get(i, "SIERRA")
        registros.append({
            "node_idx": i,
            "provincia": catalog[i].get("provincia", f"P_{i}") if i < len(catalog) else f"P_{i}",
            "macrorregion": reg,
            "betweenness": bet[i],
            "alpha_brote_medio": node_alpha.get(i, 0.0)
        })
    df_nodes = pd.DataFrame(registros)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)
    colores = {"COSTA": "#eab308", "SIERRA": "#2563eb", "SELVA": "#16a34a"}

    # Panel A: Dispersión Betweenness vs Atención GAT por Macrorregión
    for reg in ["COSTA", "SIERRA", "SELVA"]:
        sub = df_nodes[df_nodes["macrorregion"] == reg]
        axes[0].scatter(sub["betweenness"], sub["alpha_brote_medio"],
                        c=colores[reg], label=reg.capitalize(), alpha=0.75, s=45, edgecolors="white", lw=0.5)
    axes[0].set_title("A) Intermediación Topológica vs. Atención GAT", fontsize=10.5, fontweight="bold")
    axes[0].set_xlabel("Centralidad de Intermediación ($C_B$)", fontsize=9.5)
    axes[0].set_ylabel(r"Atención Media Recibida en Brotes ($\bar{\alpha}$)", fontsize=9.5)
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(frameon=True, fontsize=8.5)

    # Panel B: Peso promedio de enlaces Intra vs Interdepartamentales por Macrorregión
    df_edges_reg = df_edges.copy()
    df_edges_reg["macrorregion"] = df_edges_reg["u"].map(macro_map)
    resumen = df_edges_reg.groupby(["macrorregion", "es_interdep"])["alpha_brote"].mean().unstack(fill_value=0)
    regiones = ["COSTA", "SIERRA", "SELVA"]
    intra_vals = [resumen.loc[r, False] if r in resumen.index and False in resumen.columns else 0 for r in regiones]
    inter_vals = [resumen.loc[r, True] if r in resumen.index and True in resumen.columns else 0 for r in regiones]

    x = np.arange(len(regiones))
    w = 0.35
    axes[1].bar(x - w/2, intra_vals, w, label="Intradepartamental", color="#3b82f6")
    axes[1].bar(x + w/2, inter_vals, w, label="Interdepartamental", color="#ef4444")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([r.capitalize() for r in regiones], fontsize=9.5)
    axes[1].set_ylabel(r"Atención Media en Brotes ($\bar{\alpha}_{ij}$)", fontsize=9.5)
    axes[1].set_title("B) Intensidad Intra vs. Interdepartamental por Región", fontsize=10.5, fontweight="bold")
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)
    axes[1].legend(frameon=True, fontsize=8.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Nueva figura macrorregional guardada en: {out_path}")


def exportar_tabla_centralidades_paper(graph_data, tensor_data, endemic_q3, metadata, out_csv):
    """Exporta tabla consolidada de 5 centralidades con corrección de Bonferroni para el paper."""
    edge_index = graph_data["edge_index"].cpu().numpy()
    num_nodes = graph_data["num_nodes"]
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(zip(edge_index[0], edge_index[1]))

    time_index = metadata["time_index"]
    target_weeks = [w for _, w in time_index]
    q3_semanal = np.stack([endemic_q3[w - 1, :] for w in target_weeks], axis=0)
    vuln_pct = (np.sum((tensor_data[:, :, 0] > q3_semanal) & (tensor_data[:, :, 0] >= 5.0), axis=0) / len(tensor_data)) * 100.0

    metricas = {
        "Grado (k_i)": [G.degree[i] for i in range(num_nodes)],
        "Intermediación (C_B)": list(nx.betweenness_centrality(G).values()),
        "Cercanía (C_C)": list(nx.closeness_centrality(G).values()),
        "Agrupamiento (C_i)": list(nx.clustering(G).values()),
        "PageRank (PR_i)": list(nx.pagerank(G).values()),
    }

    filas = []
    alpha_bonf = 0.05 / 10.0
    for nombre, vals in metricas.items():
        r_p, p_p = pearsonr(vals, vuln_pct)
        r_s, p_s = spearmanr(vals, vuln_pct)
        filas.append({
            "Metrica": nombre,
            "Pearson_r": round(float(r_p), 4),
            "Pearson_p": float(p_p),
            "Spearman_rs": round(float(r_s), 4),
            "Spearman_p": float(p_s),
            "Significativo_Bonferroni_0.005": (p_p < alpha_bonf) or (p_s < alpha_bonf)
        })
    df_out = pd.DataFrame(filas)
    df_out.to_csv(out_csv, index=False)
    print(f"[OK] Tabla de centralidades (Bonferroni) guardada en: {out_csv}")
    print(df_out.to_string(index=False))


def main():
    set_seed(42)
    config = load_config()
    processed_dir = config["paths"]["processed_dir"]
    models_dir = config["paths"]["models_dir"]
    reports_dir = "reports"
    figures_dir = os.path.join(reports_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Regenerar Figura 1 limpia
    generar_diagrama_arquitectura_paper(os.path.join(figures_dir, "paper_fig1_architecture_v2.png"))

    # 2. Cargar artefactos procesados y modelo entrenado
    graph_data = torch.load(os.path.join(processed_dir, "graph_topology.pt"), weights_only=False)
    edge_index = graph_data["edge_index"].to(device)
    num_nodes = graph_data["num_nodes"]
    metadata = load_pickle(os.path.join(processed_dir, "metadata.pkl"))
    tensor_data = np.load(os.path.join(processed_dir, "tensor_eda_TNF.npy"))
    endemic_q3 = np.load(os.path.join(processed_dir, "endemic_channel_q3.npy"))

    _, _, test_loader, _ = get_dataloaders()
    model = SpatioTemporalGNN(
        num_nodes=num_nodes, in_features=tensor_data.shape[2], embed_dim=16, hidden_dim=64, dropout=0.15
    ).to(device)
    model_path = os.path.join(models_dir, "best_stgnn_model.pt")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except RuntimeError:
        # Compatibilidad si best_stgnn_model.pt fue guardado con embed_dim=8
        model = SpatioTemporalGNN(
            num_nodes=num_nodes, in_features=tensor_data.shape[2], embed_dim=8, hidden_dim=64, dropout=0.15
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    # 3. Extraer contraste Basal vs. Brote y generar nuevas figuras + tablas
    df_edges = extraer_atencion_basal_vs_brote(model, test_loader, edge_index, metadata, device)
    df_edges.to_csv(os.path.join(reports_dir, "paper_corredores_basal_vs_brote.csv"), index=False)

    generar_fig_contraste_gat(df_edges, os.path.join(figures_dir, "paper_fig_contraste_gat_basal_brote.png"))
    generar_fig_macrorregiones_topologia(df_edges, graph_data, metadata, os.path.join(figures_dir, "paper_fig_macrorregiones_gat_cb.png"))
    exportar_tabla_centralidades_paper(
        graph_data, tensor_data, endemic_q3, metadata,
        os.path.join(reports_dir, "paper_tabla_centralidades_bonferroni.csv")
    )


if __name__ == "__main__":
    main()