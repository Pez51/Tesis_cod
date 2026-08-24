import os
import yaml
import pickle
from datetime import datetime

def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_pickle(obj, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

def save_experiment_report(content_dict, report_path="data/processed/reporte_ultimo_experimento.txt"):
    """Sobrescribe el reporte con los parámetros y métricas del último entrenamiento ejecutado."""
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 65 + "\n")
        f.write(" REPORTE DE EVALUACIÓN - MODELO ST-GNN (EDA PERÚ)\n")
        f.write(f" Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 65 + "\n\n")
        for section, metrics in content_dict.items():
            f.write(f"[{section}]\n")
            f.write("-" * 45 + "\n")
            for k, v in metrics.items():
                f.write(f"  * {k:<32}: {v}\n")
            f.write("\n")