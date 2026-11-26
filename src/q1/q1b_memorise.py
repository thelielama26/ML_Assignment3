"""Q1b: large MLP + randomly corrupted training labels. Train/val only."""
import argparse, csv, json
from pathlib import Path
from collections import defaultdict
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from src.common.utils import load_config, set_seed, count_params
from src.common.data import get_data
from src.common.train import evaluate
from src.q1.models import make_mlp

p = argparse.ArgumentParser()
p.add_argument("--noise", nargs="+", type=float, default=[0.0, 0.4, 1.0])
p.add_argument("--hidden", nargs="+", type=int, default=[2048, 2048, 2048])
p.add_argument("--epochs", type=int, default=150)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--thresh", type=float, default=0.5,
               help="memorisation 'starts' when wrong-label fit first exceeds this")
a = p.parse_args()

cfg = load_config("configs/base.yaml")
dev = cfg["device"]
d = get_data(cfg)
ds = d["train"].dataset
y_orig = ds.y.clone()
n = len(y_orig)
eval_loader = DataLoader(ds, batch_size=2048, shuffle=False, num_workers=cfg["num_workers"])
Path("logs/q1/q1b").mkdir(parents=True, exist_ok=True)
Path("results/tables").mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def train_logits(model):
    model.eval()
    return torch.cat([model(x.to(dev)) for x, _ in eval_loader])


def frac(pred, target, mask):
    return (pred[mask] == target[mask]).float().mean().item() if mask.any() else float("nan")


summary = []
for noise in a.noise:
    set_seed(cfg["seed"])
    g = torch.Generator().manual_seed(cfg["seed"] + 1)
    sel = torch.rand(n, generator=g) < noise                  # examples chosen for corruption
    y_new = torch.randint(0, 10, (n,), generator=g)           # random labels
    y_asg = torch.where(sel, y_new, y_orig)
    mism = sel & (y_asg != y_orig)                            # corrupted AND actually wrong
    ds.y = y_asg                                              # loader now trains on noisy labels
    d["train"].generator.manual_seed(cfg["seed"])

    model = make_mlp(a.hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)       # no wd / dropout / augmentation
    print(f"\n=== noise={noise} | params={count_params(model)} | "
          f"corrupted-wrong examples={int(mism.sum())} ===")
    yA, yT, mm, cl = y_asg.to(dev), y_orig.to(dev), mism.to(dev), (~sel).to(dev)
    H = defaultdict(list)

    for ep in range(1, a.epochs + 1):
        model.train()
        for x, y in d["train"]:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(x), y).backward()
            opt.step()
        logits = train_logits(model)
        pred = logits.argmax(1)
        vl, va = evaluate(model, d["val"], dev)               # val labels are clean
        rec = dict(epoch=ep,
                   train_loss=F.cross_entropy(logits, yA).item(),
                   train_acc_assigned=(pred == yA).float().mean().item(),
                   acc_clean=frac(pred, yA, cl),              # uncorrupted examples
                   fit_wrong=frac(pred, yA, mm),              # memorisation of wrong labels
                   true_on_mism=frac(pred, yT, mm),           # structure: predicts TRUE label of corrupted ex.
                   val_loss=vl, val_acc=va)
        for k, v in rec.items():
            H[k].append(v)
        if ep % 5 == 0 or ep == 1:
            print(f"ep {ep:3d} | val_acc {va:.3f} | clean {rec['acc_clean']:.3f} | "
                  f"fit_wrong {rec['fit_wrong']:.3f} | true_on_mism {rec['true_on_mism']:.3f}", flush=True)

    json.dump(H, open(f"logs/q1/q1b/noise_{noise}.json", "w"))
    va_ = torch.tensor(H["val_acc"])
    fw = torch.tensor(H["fit_wrong"])
    peak = int(va_.argmax()) + 1
    mem_ep = next((i + 1 for i, v in enumerate(H["fit_wrong"]) if v == v and v >= a.thresh), None)
    summary.append(dict(noise=noise, params=count_params(model), val_acc_peak=va_.max().item(),
                        val_acc_peak_epoch=peak,
                        val_loss_min_epoch=int(torch.tensor(H["val_loss"]).argmin()) + 1,
                        memorisation_start_epoch=mem_ep,
                        val_acc_at_memorisation_start=(H["val_acc"][mem_ep - 1] if mem_ep else None),
                        true_on_mism_at_peak=H["true_on_mism"][peak - 1],
                        final_val_acc=H["val_acc"][-1], final_fit_wrong=H["fit_wrong"][-1]))
    print(summary[-1])

with open("results/tables/q1b_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
json.dump(vars(a), open("results/tables/q1b_hparams.json", "w"), indent=2)
