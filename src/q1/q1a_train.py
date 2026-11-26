"""Q1a: CNN vs matched MLP vs large MLP. Uses train/val ONLY (test untouched)."""
import argparse, csv, json
from pathlib import Path
import torch
from src.common.utils import load_config, set_seed, count_params
from src.common.data import get_data
from src.common.train import fit
from src.q1.models import q1a_specs

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=40)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--optimizer", default="adam")
p.add_argument("--n_seeds", type=int, default=3)
p.add_argument("--tag", default="q1a")
a = p.parse_args()

cfg = load_config("configs/base.yaml")
dev = cfg["device"]
specs = q1a_specs()
d = get_data(cfg)
Path("results/tables").mkdir(parents=True, exist_ok=True)
Path(f"checkpoints/{a.tag}").mkdir(parents=True, exist_ok=True)

params = {n: count_params(b()) for n, b in specs.items()}
with open(f"results/tables/{a.tag}_models.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["model", "params"])
    for n, c in params.items():
        w.writerow([n, c])
json.dump(vars(a), open(f"results/tables/{a.tag}_hparams.json", "w"), indent=2)
print("parameter counts:", params)

rows = []
for s in range(a.n_seeds):
    seed = cfg["seed"] + s
    for name, build in specs.items():
        set_seed(seed)
        model = build()
        d["train"].generator.manual_seed(seed)
        run = f"{name}_s{s}"
        h = fit(model, d["train"], d["val"], a.epochs, a.lr, dev,
                optimizer=a.optimizer, select_by="val_loss",
                log_dir=f"logs/q1/{a.tag}", run_name=run)
        be = h["best_epoch"] - 1
        torch.save(model.state_dict(), f"checkpoints/{a.tag}/{run}.pt")
        rows.append(dict(model=name, seed=seed, params=params[name],
                         best_epoch=h["best_epoch"], train_acc=h["train_acc"][be],
                         val_acc=h["val_acc"][be], val_loss=h["val_loss"][be]))

with open(f"results/tables/{a.tag}_val.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

print("\nmodel | val_acc mean +- std | train_acc mean")
for name in specs:
    r = [x for x in rows if x["model"] == name]
    va = torch.tensor([x["val_acc"] for x in r]); ta = torch.tensor([x["train_acc"] for x in r])
    print(f"{name:12s} {va.mean():.4f} +- {va.std():.4f} | {ta.mean():.4f}")
