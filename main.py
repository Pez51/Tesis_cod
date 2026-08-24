import os
import sys
import torch
from src.utils import load_config
from src.step01_etl import run_etl
from src.step02_graph_builder import build_spatial_graph
from src.step03_dataset import get_dataloaders
from src.step04_model_stgcn import SpatioTemporalGNN

def main():
    print("="*55)
    print(" PIPELINE ST-GNN: MODELADO EPIDEMIOLÓGICO DE EDA")
    print("="*55)

    config = load_config()
    processed_dir = config["paths"]["processed_dir"]

    # FASE 1: ETL
    if not os.path.exists(os.path.join(processed_dir, "tensor_eda_TNF.npy")):
        print("\n[FASE 1] Ejecutando ETL...")
        run_etl()
    else:
        print("\n[FASE 1] Datos procesados ya existentes. Omitiendo ETL.")

    # FASE 2: Grafo
    if not os.path.exists(os.path.join(processed_dir, "graph_topology.pt")):
        print("\n[FASE 2] Construyendo topología del grafo...")
        build_spatial_graph()
    else:
        print("\n[FASE 2] Topología existente. Omitiendo construcción.")

    # FASE 3: DataLoaders
    print("\n[FASE 3] Configurando DataLoaders con partición temporal...")
    train_loader, val_loader, test_loader, scalers = get_dataloaders()

    # FASE 4: Instanciación del Modelo y Test con GPU
    print("\n[FASE 4] Inicializando arquitectura ST-GNN...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Dispositivo de cómputo: {device}")

    # Cargar topología
    graph_data = torch.load(os.path.join(processed_dir, "graph_topology.pt"))
    edge_index = graph_data["edge_index"].to(device)

    num_features = len(config["data_processing"]["features"])
    model = SpatioTemporalGNN(in_features=num_features, hidden_dim=64).to(device)

    # Verificación de flujo hacia adelante (Forward pass) con un batch de prueba
    for batch_x, batch_y_reg, batch_y_cls in train_loader:
        batch_x = batch_x.to(device)
        pred_reg, pred_cls = model(batch_x, edge_index)
        
        print("\n--- Verificación Exitosa de Tensores ---")
        print(f" Entrada Batch X         : {batch_x.shape} (B, P, N, F)")
        print(f" Salida Predicción Casos : {pred_reg.shape} (B, N)")
        print(f" Salida Alertas Brotes   : {pred_cls.shape} (B, N)")
        print(f" Dispositivo Activo      : {pred_reg.device}")
        break

    print("\n" + "="*55)
    print(" FASES 1 A 4 INTEGRADAS Y LISTAS PARA EL ENTRENAMIENTO")
    print("="*55)

if __name__ == "__main__":
    main()