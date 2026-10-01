import os
import numpy as np
from src.utils import load_config, load_pickle

config = load_config()
processed_dir = config["paths"]["processed_dir"]
metadata = load_pickle(os.path.join(processed_dir, "metadata.pkl"))

print("Claves disponibles en metadata.pkl:", list(metadata.keys()))
if "feature_names" in metadata:
    print("\nNombres reales guardados en ETL:")
    for idx, name in enumerate(metadata["feature_names"]):
        print(f"  Columna {idx:02d}: {name}")
else:
    print("\n[ALERTA] 'feature_names' no se guardó en metadata.pkl.")
    print("Revisar la función donde se construye tensor_eda_TNF en step01_etl.py")