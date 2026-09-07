import numpy as np
from src.utils import load_config, load_pickle
from src.step05_train_eval import get_province_macroregions
from src.step03_dataset import get_dataloaders

config = load_config()
processed_dir = config["paths"]["processed_dir"]

metadata = load_pickle(f"{processed_dir}/metadata.pkl")
targets = np.load(f"{processed_dir}/targets_outbreak.npy") # (1304, 196)
ubigeos = metadata["ubigeos"]
reg_map = get_province_macroregions(ubigeos)

_, val_loader, test_loader, _ = get_dataloaders()
num_val = len(val_loader.dataset)
num_test = len(test_loader.dataset)
T_total = len(targets)
train_T = int(T_total * config["model_params"]["train_split"])
val_T = int(T_total * config["model_params"]["val_split"])
seq_len = config["model_params"]["seq_len"]

val_targets = targets[train_T + seq_len : train_T + seq_len + num_val, :]
test_targets = targets[train_T + val_T + seq_len : train_T + val_T + seq_len + num_test, :]

regions = np.array([reg_map[i] for i in range(len(ubigeos))])

print("=" * 70)
print(f"{'MACRORREGIÓN':<12} | {'PROVINCIAS':<10} | {'BROTES VAL (%)':<18} | {'BROTES TEST (%)':<18}")
print("=" * 70)

for reg in ["COSTA", "SIERRA", "SELVA"]:
    mask = (regions == reg)
    n_provs = np.sum(mask)
    
    val_reg = val_targets[:, mask]
    test_reg = test_targets[:, mask]
    
    val_b = np.sum(val_reg == 1)
    val_total = val_reg.size
    val_pct = (val_b / val_total) * 100
    
    test_b = np.sum(test_reg == 1)
    test_total = test_reg.size
    test_pct = (test_b / test_total) * 100
    
    print(f"{reg:<12} | {n_provs:<10} | {val_b:>5}/{val_total:<5} ({val_pct:4.1f}%) | {test_b:>5}/{test_total:<5} ({test_pct:4.1f}%)")

print("=" * 70)