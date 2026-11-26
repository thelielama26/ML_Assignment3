import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, pandas as pd, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.common.utils import load_config
from src.q3.ae import ConvAE, load_u8, fixed_noisy, recon

cfg = load_config("configs/base.yaml"); dev = cfg["device"]
out = Path("results/figures"); out.mkdir(parents=True, exist_ok=True)
models = pd.read_csv("results/tables/q3a_models.csv")

# training curves
fig, ax = plt.subplots(figsize=(5.5, 4))
for d in models.bottleneck:
    H = json.load(open(f"logs/q3/ae_d{d}.json")); ax.plot(H["epoch"], H["val_mse"], label=f"d={d}")
ax.set_xlabel("epoch"); ax.set_ylabel("validation MSE"); ax.set_yscale("log"); ax.legend()
fig.tight_layout(); fig.savefig(out / "q3a_val_curves.png", dpi=150); fig.savefig(out / "q3a_val_curves.pdf")

# reconstruction examples (first 8 test images, fixed test noise)
hp = json.load(open("results/tables/q3b_hparams.json")); d, sigma = hp["bottleneck"], hp["sigma"]
D = load_u8(cfg); clean = (D["xte"].float() / 255)[:8]
noisy = fixed_noisy(D["xte"].float() / 255, sigma, cfg["seed"] + 202)[:8]
ae = ConvAE(d).to(dev); ae.load_state_dict(torch.load(f"checkpoints/q3/ae_d{d}.pt", map_location=dev))
dae = ConvAE(d).to(dev); dae.load_state_dict(torch.load(f"checkpoints/q3/dae_d{d}.pt", map_location=dev))
cols = [("clean", clean), ("corrupted", noisy), ("clean-AE(corrupted)", recon(ae, noisy, dev)),
        ("DAE(corrupted)", recon(dae, noisy, dev))]
fig, ax = plt.subplots(8, 4, figsize=(6, 12))
for c, (t, x) in enumerate(cols):
    for r in range(8):
        ax[r, c].imshow(x[r].permute(1, 2, 0).clamp(0, 1).numpy()); ax[r, c].axis("off")
        if r == 0: ax[r, c].set_title(t, fontsize=7)
fig.tight_layout(); fig.savefig(out / "q3b_examples.png", dpi=150); fig.savefig(out / "q3b_examples.pdf")

# reconstruction vs class information
pr = pd.read_csv("results/tables/q3c_probe.csv"); a = pr[pr.representation.str.startswith("clean_AE")]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
for k, lab in (("linear_test_acc", "linear probe"), ("knn_test_acc", "kNN probe")):
    ax[0].plot(a.d, a[k], "o-", label=lab)
ax[0].set_xscale("log", base=2); ax[0].set_xlabel("bottleneck d"); ax[0].set_ylabel("test accuracy"); ax[0].legend()
ax[1].plot(a.d, a.test_recon_mse, "o-"); ax[1].set_xscale("log", base=2); ax[1].set_yscale("log")
ax[1].set_xlabel("bottleneck d"); ax[1].set_ylabel("test reconstruction MSE")
fig.tight_layout(); fig.savefig(out / "q3c_recon_vs_class.png", dpi=150); fig.savefig(out / "q3c_recon_vs_class.pdf")
print("saved Q3 figures")
