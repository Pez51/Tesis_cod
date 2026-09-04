import os
import sys
from src.utils import load_config
from src.step01_etl import run_etl
from src.step02_graph_builder import build_spatial_graph
from src.step05_train_eval import train_and_evaluate

def main():
    print("="*55)
    print(" PIPELINE ST-GNN: MODELADO EPIDEMIOLOGICO DE EDA")
    print("="*55)

    config = load_config()
    processed_dir = config["paths"]["processed_dir"]

    # FASE 1: ETL
    if not os.path.exists(os.path.join(processed_dir, "tensor_eda_TNF.npy")):
        print("\n[FASE 1] Ejecutando ETL...")
        run_etl()
    else:
        print("\n[FASE 1] Datos procesados ya existentes. Omitiendo ETL.")

    # FASE 2: Topología del Grafo
    if not os.path.exists(os.path.join(processed_dir, "graph_topology.pt")):
        print("\n[FASE 2] Construyendo topologia del grafo...")
        build_spatial_graph()
    else:
        print("\n[FASE 2] Topologia existente. Omitiendo construccion.")

    # FASE 3, 4 y 5: Entrenamiento y Evaluación
    print("\n[FASE 3, 4 y 5] Iniciando entrenamiento y evaluacion del modelo...")
    try:
        train_and_evaluate()
    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()