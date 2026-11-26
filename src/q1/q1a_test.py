import argparse, csv, json
import torch
from src.common.utils import load_config, count_params
from src.common.data import get_data
from src.common.train import evaluate
from src.q1.models import q1a_specs

p = argparse.ArgumentParser()
p.add_argument("--tag", default="q1a")
p.add_argument("--n_seeds", type=int, default=3)
a = p.parse_args()

cfg = load_config("configs/base.yaml")
dev = cfg["device"]
d = get_data(cfg)
rows = []
for s in range(a.n_seeds):
    for name, build in q1a_specs().items():
        m = build().to(dev)
        m.load_state_dict(torch.load(f"checkpoints/{a.tag}/{name}_s{s}.pt", map_location=dev))
        loss, acc = evaluate(m, d["test"], dev)
        rows.append(dict(model=name, seed=cfg["seed"] + s, params=count_params(m),
                         test_loss=loss, test_acc=acc))
with open(f"results/tables/{a.tag}_test.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("model | params | TEST acc mean +- std")
for name in q1a_specs():
    r = [x for x in rows if x["model"] == name]
    t = torch.tensor([x["test_acc"] for x in r])
    print(f"{name:12s} {r[0]['params']:>10d} | {t.mean():.4f} +- {t.std():.4f}")
