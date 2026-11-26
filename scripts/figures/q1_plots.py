import json, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = Path("results/figures"); out.mkdir(parents=True, exist_ok=True)

# ---- Q1a: mean +- std over seeds ----
runs = {}
for f in sorted(glob.glob("logs/q1/q1a/*_s*.json")):
    runs.setdefault(Path(f).stem.rsplit("_s", 1)[0], []).append(json.load(open(f)))
fig, ax = plt.subplots(2, 2, figsize=(10, 7))
for name, hs in runs.items():
    for a_, key in zip(ax.ravel(), ["train_loss", "val_loss", "train_acc", "val_acc"]):
        arr = np.array([h[key] for h in hs]); m, s = arr.mean(0), arr.std(0)
        e = np.arange(1, len(m) + 1)
        a_.plot(e, m, label=name); a_.fill_between(e, m - s, m + s, alpha=0.2)
        a_.set_title(key); a_.set_xlabel("epoch")
ax[0, 0].legend(); fig.tight_layout()
fig.savefig(out / "q1a_curves.pdf"); fig.savefig(out / "q1a_curves.png", dpi=150)

# ---- Q1b: one panel per noise level ----
files = sorted(glob.glob("logs/q1/q1b/noise_*.json"), key=lambda f: float(Path(f).stem.split("_")[1]))
if files:
    fig, ax = plt.subplots(1, len(files), figsize=(5.2 * len(files), 4), squeeze=False)
    for a_, f in zip(ax[0], files):
        H = json.load(open(f)); e = H["epoch"]
        a_.plot(e, H["val_acc"], label="val acc (clean)", lw=2)
        a_.plot(e, H["true_on_mism"], label="corrupted ex.: true-label acc")
        a_.plot(e, H["fit_wrong"], label="corrupted ex.: wrong-label fit")
        a_.plot(e, H["acc_clean"], label="uncorrupted train ex. acc", ls="--")
        a_.axvline(int(np.argmax(H["val_acc"])) + 1, color="k", ls=":", lw=1)
        a_.axhline(0.1, color="gray", lw=0.6)
        a_.set_title(f"label noise = {Path(f).stem.split('_')[1]}"); a_.set_xlabel("epoch"); a_.set_ylim(0, 1.02)
    ax[0, 0].legend(fontsize=7); fig.tight_layout()
    fig.savefig(out / "q1b_curves.pdf"); fig.savefig(out / "q1b_curves.png", dpi=150)
print("saved figures to", out)
