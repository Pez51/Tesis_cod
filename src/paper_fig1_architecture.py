import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generar_diagrama_arquitectura(output_path="reports/figures/paper_fig1_architecture.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Dimensiones estándar para figura a dos columnas IEEE (ancho ~14-16 in)
    fig, ax = plt.subplots(figsize=(15, 6.5), dpi=300)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Paleta de colores científicos (Estilo IEEE/Nature)
    c_blue   = "#2563eb" # Entradas / Tensores
    c_slate  = "#475569" # Embeddings
    c_purple = "#7c3aed" # Capa Espacial GAT
    c_indigo = "#4f46e5" # Capa Temporal GRU
    c_teal   = "#0d9488" # Cabezales de predicción
    c_rose   = "#e11d48" # Módulo XAI / Corredores
    c_bg     = "#f8fafc" # Cajas de fondo

    # Helper para cajas con bordes redondeados
    def draw_box(x, y, w, h, bg_color, border_color, title, subtitle="", radius=0.25):
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad={radius}",
            facecolor=bg_color, edgecolor=border_color, linewidth=1.5, zorder=2
        )
        ax.add_patch(rect)
        if subtitle:
            ax.text(x + w/2, y + h*0.62, title, ha="center", va="center",
                    fontsize=9.5, fontweight="bold", color="#0f172a", zorder=3)
            ax.text(x + w/2, y + h*0.30, subtitle, ha="center", va="center",
                    fontsize=7.5, color="#334155", zorder=3)
        else:
            ax.text(x + w/2, y + h/2, title, ha="center", va="center",
                    fontsize=9.0, fontweight="bold", color="#0f172a", zorder=3)

    def draw_arrow(x1, y1, x2, y2, color="#64748b", lw=2.0, style="->"):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=3, shrinkB=3),
            zorder=4
        )

    # -------------------------------------------------------------
    # 1. BLOQUE ENTRADA: DATASET & GRAFO TERRITORIAL
    # -------------------------------------------------------------
    rect_input = patches.Rectangle((0.3, 0.6), 2.7, 5.3, facecolor="#f1f5f9", edgecolor="#cbd5e1", lw=1.2, linestyle="--")
    ax.add_patch(rect_input)
    ax.text(1.65, 5.65, "A) Input Space (Peru)", ha="center", fontsize=9.5, fontweight="bold", color="#1e293b")

    draw_box(0.5, 3.8, 2.3, 1.4, "#eff6ff", c_blue, 
             "Historical Spatiotemporal\nFeature Tensor", 
             r"$\mathcal{X} \in \mathbb{R}^{L \times N \times F}$" + "\n($L=8, N=196, F=17$)")
    
    draw_box(0.5, 1.1, 2.3, 1.4, "#eff6ff", c_blue, 
             "Queen Contiguity\nGraph Topology", 
             r"$\mathcal{G} = (\mathcal{V}, \mathcal{E})$" + "\n(196 Nodes, 1,078 Edges)")

    # -------------------------------------------------------------
    # 2. BLOQUE: NODE EMBEDDING
    # -------------------------------------------------------------
    draw_box(3.6, 3.8, 1.8, 1.4, "#f8fafc", c_slate,
             "Learned Static\nNode Embeddings",
             r"$\mathbf{e}_i \in \mathbb{R}^8$")

    draw_arrow(2.8, 4.5, 3.6, 4.5)
    
    # Concatenación
    ax.plot([4.5, 4.5], [3.8, 2.8], color="#64748b", lw=1.5)
    ax.plot([1.65, 5.0], [2.8, 2.8], color="#64748b", lw=1.5)
    ax.scatter([5.0], [2.8], s=250, color="#ffffff", edgecolor="#0f172a", lw=1.5, zorder=5)
    ax.text(5.0, 2.8, r"$\parallel$", ha="center", va="center", fontsize=12, fontweight="bold", color="#0f172a", zorder=6)
    ax.text(5.0, 2.2, "Feature Concat\n(Dim = 25)", ha="center", fontsize=7.5, color="#475569")

    # -------------------------------------------------------------
    # 3. BLOQUE ESPACIAL: GAT CON ATENCIÓN DINÁMICA
    # -------------------------------------------------------------
    rect_gat = patches.Rectangle((5.8, 1.0), 2.5, 4.9, facecolor="#faf5ff", edgecolor="#e9d5ff", lw=1.2)
    ax.add_patch(rect_gat)
    ax.text(7.05, 5.65, "B) Spatial GAT Layer", ha="center", fontsize=9.5, fontweight="bold", color=c_purple)

    draw_box(6.0, 3.4, 2.1, 1.8, "#ffffff", c_purple,
             "Graph Attention\nMechanism",
             r"$\alpha_{ij} = \text{Softmax}(\text{LeakyReLU})$" + "\n" + r"$\mathbf{h}_i = \sum_{j \in \mathcal{N}_i} \alpha_{ij} \mathbf{W} \mathbf{x}_j$")

    draw_box(6.0, 1.3, 2.1, 1.4, "#ffffff", c_purple,
             "Dynamic Spatial\nWeights Matrix",
             r"$\mathbf{A}_t = (\alpha_{ij}) \in \mathbb{R}^{N \times N}$")

    draw_arrow(5.2, 2.8, 6.0, 4.3)
    draw_arrow(7.05, 3.4, 7.05, 2.7, color=c_purple)

    # -------------------------------------------------------------
    # 4. BLOQUE TEMPORAL: GRU MULTI-LAYER
    # -------------------------------------------------------------
    rect_gru = patches.Rectangle((8.9, 1.0), 2.1, 4.9, facecolor="#eef2ff", edgecolor="#c7d2fe", lw=1.2)
    ax.add_patch(rect_gru)
    ax.text(9.95, 5.65, "C) Temporal Recurrence", ha="center", fontsize=9.5, fontweight="bold", color=c_indigo)

    draw_box(9.1, 2.5, 1.7, 2.4, "#ffffff", c_indigo,
             "Gated Recurrent\nUnit (GRU)",
             "2 Layers (Hidden=64)\n" + r"$\mathbf{z}_{t,i} \in \mathbb{R}^{64}$" + "\nDropout = 0.15")

    draw_arrow(8.1, 4.3, 9.1, 3.7)

    # -------------------------------------------------------------
    # 5. BLOQUE MULTITAREA: SALIDAS PREDICTIVAS
    # -------------------------------------------------------------
    rect_out = patches.Rectangle((11.6, 3.2), 3.1, 2.7, facecolor="#f0fdf4", edgecolor="#bbf7d0", lw=1.2)
    ax.add_patch(rect_out)
    ax.text(13.15, 5.65, "D) Multi-Task Predictions", ha="center", fontsize=9.5, fontweight="bold", color=c_teal)

    draw_box(11.8, 4.5, 2.7, 0.9, "#ffffff", c_teal,
             "Continuous Incidence Head", r"$\hat{y}_{t+1,i} \in \mathbb{R}^+$ (Huber Loss)")

    draw_box(11.8, 3.4, 2.7, 0.9, "#ffffff", c_teal,
             "Outbreak Classifier Head", r"$\hat{p}_{t+1,i} \in [0, 1]$ (Focal BCE)")

    draw_arrow(10.8, 4.2, 11.8, 4.9, color=c_teal)
    draw_arrow(10.8, 3.2, 11.8, 3.8, color=c_teal)

    # -------------------------------------------------------------
    # 6. BLOQUE XAI: EXPLICABILIDAD ESPACIAL Y CORREDORES
    # -------------------------------------------------------------
    rect_xai = patches.Rectangle((11.6, 0.6), 3.1, 2.3, facecolor="#fff1f2", edgecolor="#fecdd3", lw=1.2)
    ax.add_patch(rect_xai)
    ax.text(13.15, 2.65, "E) Spatial XAI & Network Theory", ha="center", fontsize=9.0, fontweight="bold", color=c_rose)

    draw_box(11.8, 0.8, 2.7, 1.5, "#ffffff", c_rose,
             "Introspection Protocol",
             r"$\bar{\alpha}_{ij} = \frac{1}{|\mathcal{T}|} \sum \alpha_{ij}^{(t)}$ for $y > Q_3$" + "\n" +
             "Betweenness Centrality ($C_B$)\nMoran's I Autocorrelation ($p < 0.001$)")

    # Flecha desde matriz GAT hacia XAI
    draw_arrow(8.1, 2.0, 11.8, 1.55, color=c_rose, lw=2.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Figura 1 del Paper exportada exitosamente en: {output_path}")

if __name__ == "__main__":
    generar_diagrama_arquitectura()