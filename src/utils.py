import os
import yaml
import pickle

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