import argparse, csv, json
from pathlib import Path
import torch
from src.common.utils import load_config, set_seed, count_params
from src.common.data import get_data
from src.common.train import fit
from src.q1.models import q1a_specs
from src.q2.transforms import get_transform

q1h = json.load(open("results/tables/q1a_hparams.json"))   # SAME budget as Q1a (fairness)
p = argparse.ArgumentParser()
p.add_argument("--conditions", nargs="+", default=["perm", "gray", "patch8"])
p.add_argument("--models", nargs="+", default=["cnn", "mlp_matched"])
p.add_argument("--epochs", type=int, default=q1h["epochs"])
p.add_argument("--lr", type=float, default=q1h["lr"])
p.add_argument("--optimizer", default=q1h["optimizer"])
p.add_argument("--n_seeds", type=int, default=q1h["n_seeds"])
a = p.parse_args()

cfg = load_config("configs/base.yaml")
dev = cfg["device"]
specs = q1a_specs()
Path("results/tables").mkdir(parents=True, exist_ok=True)
json.dump(vars(a), open("results/tables/q2_hparams.json", "w"), indent=2)
params = {m: count_params(specs[m]()) for m in specs if m in a.models}
with open("results/tables/q2_models.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["model", "params"])
    for m, c in params.items():
        w.writerow([m, c])
print("params:", params)

for cond in a.conditions:
    tf = get_transform(cond, cfg["seed"])      # fixed transform, identical for train/val/test
    d = get_data(cfg, transform=tf)
    rows = []
    for s in range(a.n_seeds):
        seed = cfg["seed"] + s
        for m in a.models:
            set_seed(seed)
            model = specs[m]()
            d["train"].generator.manual_seed(seed)
            h = fit(model, d["train"], d["val"], a.epochs, a.lr, dev, optimizer=a.optimizer,
                    select_by="val_loss", log_dir=f"logs/q2/{cond}", run_name=f"{m}_s{s}")
            be = h["best_epoch"] - 1
            torch.save(model.state_dict(), f"checkpoints/q2/{cond}_{m}_s{s}.pt")
            rows.append(dict(condition=cond, model=m, seed=seed, params=params[m],
                             best_epoch=h["best_epoch"], train_acc=h["train_acc"][be],
                             val_acc=h["val_acc"][be], val_loss=h["val_loss"][be]))
    with open(f"results/tables/q2_val_{cond}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for m in a.models:
        v = torch.tensor([r["val_acc"] for r in rows if r["model"] == m])
        print(f"[{cond}] {m:12s} val_acc {v.mean():.4f} +- {v.std():.4f}")
