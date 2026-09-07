import os
import sys
from src.utils import load_config
from src.step01_etl import run_etl
from src.step02_graph_builder import build_graph
from src.step05_train_eval import train_and_evaluate

def main():
    print("=" * 75)
    print(" PIPELINE ST-GNN PROVINCIAL: VIGILANCIA EPIDEMIOLÓGICA DE EDA (PERÚ)")
    print("=" * 75)

    config = load_config()
    processed_dir = config["paths"]["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(config["paths"]["models_dir"], exist_ok=True)
    os.makedirs(os.path.join("reports", "figures"), exist_ok=True)

    tensor_path = os.path.join(processed_dir, "tensor_eda_TNF.npy")
    targets_path = os.path.join(processed_dir, "targets_outbreak.npy")
    graph_path = os.path.join(processed_dir, "graph_topology.pt")

    # FASE 1: Ingesta, Agregación Provincial y Canal Endémico Dinámico
    if not os.path.exists(tensor_path) or not os.path.exists(targets_path):
        print("\n[FASE 1] Ejecutando ETL Provincial (196 provincias)...")
        run_etl()
    else:
        print("\n[FASE 1] Tensor provincial y canal endémico ya existentes. Omitiendo ETL.")

    # FASE 2: Modelado de Topología Espacial (Fronteras y Adyacencia Queen)
    if not os.path.exists(graph_path):
        print("\n[FASE 2] Construyendo topología provincial continua...")
        build_graph()
    else:
        print("\n[FASE 2] Topología de grafo existente. Omitiendo construcción.")

    # FASE 3, 4 y 5: Carga de Datos, ST-GNN, Entrenamiento y Reporte
    print("\n[FASE 3, 4 y 5] Iniciando entrenamiento y evaluación predictiva...")
    train_and_evaluate()

    print("\n" + "=" * 75)
    print(" PIPELINE PROVINCIAL COMPLETADO SATISFACTORIAMENTE")
    print("=" * 75)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Ejecución interrumpida manualmente por el usuario.")
        sys.exit(0)