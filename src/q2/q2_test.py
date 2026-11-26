import argparse, csv
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from src.common.utils import load_config, count_params
from src.common.data import get_data, CLASSES
from src.q1.models import q1a_specs
from src.q2.transforms import get_transform

p = argparse.ArgumentParser()
p.add_argument("--conditions", nargs="+", default=["clean", "perm", "gray", "patch8"])
p.add_argument("--models", nargs="+", default=["cnn", "mlp_matched"])
p.add_argument("--n_seeds", type=int, default=3)
a = p.parse_args()

cfg = load_config("configs/base.yaml")
dev = cfg["device"]
specs = q1a_specs()
Path("results/q2").mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def predict(model, loader):
    model.eval(); L, Y = [], []
    for x, y in loader:
        L.append(model(x.to(dev)).cpu()); Y.append(y)
    return torch.cat(L), torch.cat(Y)


def ece(conf, correct, bins=15):
    edges, e = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


rows, cls_rows, agree_rows = [], [], []
for cond in a.conditions:
    d = get_data(cfg, transform=get_transform(cond, cfg["seed"]))   # SAME transform as training
    for s in range(a.n_seeds):
        preds = {}
        for m in a.models:
            model = specs[m]().to(dev)
            ck = (f"checkpoints/q1a/{m}_s{s}.pt" if cond == "clean"   # clean = reused Q1a model
                  else f"checkpoints/q2/{cond}_{m}_s{s}.pt")
            model.load_state_dict(torch.load(ck, map_location=dev))
            logits, y = predict(model, d["test"])
            prob = F.softmax(logits, 1)
            conf, pred = prob.max(1)
            conf, pred, y_np = conf.numpy(), pred.numpy(), y.numpy()
            correct = pred == y_np
            nll = F.cross_entropy(logits, y).item()
            np.savez(f"results/q2/test_{cond}_{m}_s{s}.npz", y=y_np, pred=pred, conf=conf)
            rows.append(dict(condition=cond, model=m, seed=cfg["seed"] + s, params=count_params(model),
                             test_acc=correct.mean(), test_nll=nll, mean_conf=conf.mean(),
                             conf_correct=conf[correct].mean() if correct.any() else np.nan,
                             conf_wrong=conf[~correct].mean() if (~correct).any() else np.nan,
                             ece=ece(conf, correct)))
            cls_rows.append(dict(condition=cond, model=m, seed=cfg["seed"] + s,
                                 **{CLASSES[k]: correct[y_np == k].mean() for k in range(10)}))
            preds[m] = (pred, y_np)
        if "cnn" in preds and "mlp_matched" in preds:
            pc, y_np = preds["cnn"]; pm, _ = preds["mlp_matched"]
            cc, cm = pc == y_np, pm == y_np
            agree_rows.append(dict(condition=cond, seed=cfg["seed"] + s,
                                   both_correct=(cc & cm).mean(), cnn_only=(cc & ~cm).mean(),
                                   mlp_only=(~cc & cm).mean(), both_wrong=(~cc & ~cm).mean()))


def write(path, rs):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rs[0])); w.writeheader(); w.writerows(rs)


write("results/tables/q2_test.csv", rows)
write("results/tables/q2_class_acc.csv", cls_rows)
if agree_rows:
    write("results/tables/q2_agreement.csv", agree_rows)

print("condition | model | params | TEST acc | ECE | conf(correct) | conf(wrong)   [mean over seeds]")
summ = []
for cond in a.conditions:
    for m in a.models:
        r = [x for x in rows if x["condition"] == cond and x["model"] == m]
        g = lambda k: np.mean([x[k] for x in r])
        sd = np.std([x["test_acc"] for x in r], ddof=1) if len(r) > 1 else 0.0
        summ.append(dict(condition=cond, model=m, params=r[0]["params"], test_acc_mean=g("test_acc"),
                         test_acc_std=sd, ece=g("ece"), conf_correct=g("conf_correct"),
                         conf_wrong=g("conf_wrong")))
        print(f"{cond:7s} {m:12s} {r[0]['params']:>8d} {g('test_acc'):.4f}+-{sd:.4f} "
              f"{g('ece'):.3f} {g('conf_correct'):.3f} {g('conf_wrong'):.3f}")
write("results/tables/q2_test_summary.csv", summ)
