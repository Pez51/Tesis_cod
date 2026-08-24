import sys
from src.step01_etl import run_etl
from src.step02_graph_builder import build_spatial_graph

def main():
    print("="*55)
    print(" PIPELINE ST-GNN: MODELADO EPIDEMIOLÓGICO DE EDA")
    print("="*55)

    # FASE 1: Limpieza y Creación de Tensores
    print("\n[FASE 1] Extracción, Transformación y Carga (ETL)...")
    try:
        run_etl()
    except Exception as e:
        print(f"Error en Fase 1: {e}")
        sys.exit(1)

    # FASE 2: Descarga de Cartografía y Construcción del Grafo
    print("\n[FASE 2] Descarga Cartográfica y Construcción de Topología...")
    try:
        build_spatial_graph()
    except Exception as e:
        print(f"Error en Fase 2: {e}")
        sys.exit(1)

    print("\n" + "="*55)
    print(" FASES 1 Y 2 COMPLETADAS EXITOSAMENTE")
    print(" Artefactos generados en 'data/processed/':")
    print("  - df_master_weekly.parquet")
    print("  - tensor_eda_TNF.npy")
    print("  - targets_outbreak.npy")
    print("  - adj_matrix.npy")
    print("  - graph_topology.pt")
    print("="*55)

if __name__ == "__main__":
    main()