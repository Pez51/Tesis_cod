import os
import sys
from src.step01_etl import run_etl

def main():
    print("="*50)
    print(" PIPELINE ST-GNN: PRECAUCIÓN DE EDA EN PERÚ")
    print("="*50)

    # 1. Fase de ETL y Limpieza
    print("\n[FASE 1] Iniciando Extracción, Transformación y Carga...")
    try:
        run_etl()
    except Exception as e:
        print(f"Error durante el ETL: {e}")
        sys.exit(1)
        
    print("\nFase 1 completada. Revisa la carpeta 'data/processed/' para ver los tensores generados.")
    
    # Aquí iremos agregando las siguientes fases conforme avancemos:
    # 2. Generación de Grafo (step02_graph_builder.py)
    # 3. Entrenamiento (step05_train_eval.py)

if __name__ == "__main__":
    main()