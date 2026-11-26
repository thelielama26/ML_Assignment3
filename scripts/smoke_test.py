import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch, torch.nn as nn
from src.common.utils import load_config, set_seed, count_params
from src.common.data import get_data
from src.common.train import fit, evaluate

cfg = load_config("configs/base.yaml")
set_seed(cfg["seed"])
print("seed:", cfg["seed"], "| device:", cfg["device"])

d = get_data(cfg)
print("train/val/test sizes:",
      len(d["train"].dataset), len(d["val"].dataset), len(d["test"].dataset))
print("val class counts:", torch.bincount(d["val"].dataset.y).tolist())
print("mean/std:", d["mean"], d["std"])

model = nn.Sequential(nn.Flatten(), nn.Linear(3072, 10))   # throwaway model, plumbing check only
print("params:", count_params(model))
fit(model, d["train"], d["val"], epochs=2, lr=1e-3, device=cfg["device"],
    log_dir="logs/smoke", run_name="smoke")
print("OK")
