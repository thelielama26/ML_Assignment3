"""Create the ONE stratified 45k/5k split. Run once; never regenerate after results exist."""
import sys, json, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.datasets import CIFAR10
from src.common.utils import load_config

cfg = load_config("configs/base.yaml")
seed = cfg["seed"]
split_path = Path(cfg["split_path"])
split_path.parent.mkdir(parents=True, exist_ok=True)

ds = CIFAR10(root=cfg["data_root"], train=True, download=True)
y = np.array(ds.targets)
idx = np.arange(len(y))
train_idx, val_idx = train_test_split(
    idx, test_size=5000, stratify=y, random_state=seed, shuffle=True)
train_idx, val_idx = np.sort(train_idx), np.sort(val_idx)

assert len(train_idx) == 45000 and len(val_idx) == 5000
assert len(set(train_idx) & set(val_idx)) == 0

np.savez(split_path, train_idx=train_idx, val_idx=val_idx, seed=seed)
meta = {
    "roll_number": cfg["roll_number"],
    "seed": seed,
    "n_train": int(len(train_idx)),
    "n_val": int(len(val_idx)),
    "train_class_counts": np.bincount(y[train_idx]).tolist(),
    "val_class_counts": np.bincount(y[val_idx]).tolist(),
    "sha256_train_idx": hashlib.sha256(train_idx.tobytes()).hexdigest(),
    "sha256_val_idx": hashlib.sha256(val_idx.tobytes()).hexdigest(),
}
with open(split_path.with_suffix(".json"), "w") as f:
    json.dump(meta, f, indent=2)
print(json.dumps(meta, indent=2))
