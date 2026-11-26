import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

out = Path("results/figures"); out.mkdir(parents=True, exist_ok=True)
df = pd.read_csv("results/tables/q2_test.csv")
conds = [c for c in ["clean", "perm", "gray", "patch8"] if c in set(df.condition)]
models = ["cnn", "mlp_matched"]
g = df.groupby(["condition", "model"]).agg(acc=("test_acc", "mean"), sd=("test_acc", "std"),
                                           cc=("conf_correct", "mean"), cw=("conf_wrong", "mean")).reset_index()

fig, ax = plt.subplots(figsize=(7, 4)); w = 0.38
for i, m in enumerate(models):
    s = g[g.model == m].set_index("condition").loc[conds]
    ax.bar(np.arange(len(conds)) + (i - .5) * w, s.acc, w, yerr=s.sd, capsize=3, label=m)
ax.set_xticks(range(len(conds))); ax.set_xticklabels(conds); ax.set_ylabel("test accuracy"); ax.legend()
fig.tight_layout(); fig.savefig(out / "q2_accuracy.pdf"); fig.savefig(out / "q2_accuracy.png", dpi=150)

fig, ax = plt.subplots(len(models), len(conds), figsize=(3.2 * len(conds), 3.2 * len(models)), squeeze=False)
for r, m in enumerate(models):
    for c, cd in enumerate(conds):
        z = np.load(f"results/q2/test_{cd}_{m}_s0.npz"); conf = z["conf"]; corr = z["pred"] == z["y"]
        e = np.linspace(0, 1, 11); xs, ys = [], []
        for lo, hi in zip(e[:-1], e[1:]):
            k = (conf > lo) & (conf <= hi)
            if k.sum() > 0:
                xs.append(conf[k].mean()); ys.append(corr[k].mean())
        ax[r, c].plot([0, 1], [0, 1], "k:"); ax[r, c].plot(xs, ys, "o-")
        ax[r, c].set_title(f"{m} | {cd}", fontsize=8); ax[r, c].set_xlabel("confidence"); ax[r, c].set_ylabel("accuracy")
fig.tight_layout(); fig.savefig(out / "q2_reliability.pdf"); fig.savefig(out / "q2_reliability.png", dpi=150)

fig, ax = plt.subplots(1, len(models), figsize=(5.5 * len(models), 3.8), squeeze=False)
for i, m in enumerate(models):
    s = g[g.model == m].set_index("condition").loc[conds]; x = np.arange(len(conds))
    ax[0, i].bar(x - .2, s.cc, .4, label="correct"); ax[0, i].bar(x + .2, s.cw, .4, label="wrong")
    ax[0, i].set_xticks(x); ax[0, i].set_xticklabels(conds); ax[0, i].set_title(m); ax[0, i].set_ylabel("mean confidence")
ax[0, 0].legend(); fig.tight_layout(); fig.savefig(out / "q2_confidence.pdf"); fig.savefig(out / "q2_confidence.png", dpi=150)
print("saved Q2 figures")
