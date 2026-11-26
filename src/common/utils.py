import random
import numpy as np
import torch
import yaml


def roll_to_seed(roll_number: str) -> int:
    """All numeric characters of the roll number, in order, as one integer."""
    digits = "".join(c for c in roll_number if c.isdigit())
    assert digits, "roll number has no digits"
    return int(digits)


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["seed"] = roll_to_seed(str(cfg["roll_number"]))
    if cfg.get("device", "cpu") == "cuda" and not torch.cuda.is_available():
        cfg["device"] = "cpu"
    return cfg


def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_params(model, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters()
               if p.requires_grad or not trainable_only)
