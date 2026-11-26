"""Q3b: denoising AE (same arch as selected clean AE). Train/val only; prints validation pipeline results."""
import argparse, json
import torch
from src.common.utils import load_config, set_seed
from src.q3.ae import (ConvAE, load_u8, train_ae, fixed_noisy, load_classifier,
                       pipeline_rows, write_csv)

p = argparse.ArgumentParser()
p.add_argument("--sigma", type=float, default=0.2)
p.add_argument("--epochs", type=int, default=40)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--bs", type=int, default=128)
a = p.parse_args()

cfg = load_config("configs/base.yaml")
dev = cfg["device"]
d = json.load(open("results/tables/q3a_selected.json"))["bottleneck"]
D = load_u8(cfg)
xva = D["xva"].float() / 255
val_noisy = fixed_noisy(xva, a.sigma, cfg["seed"] + 101)       # fixed val corruption

set_seed(cfg["seed"])                                          # same init seed as the clean AE
dae = ConvAE(d)
H = train_ae(dae, D["xtr"], val_noisy, xva, a.epochs, a.lr, a.bs, a.sigma, dev, cfg["seed"],
             "logs/q3", f"dae_d{d}")
torch.save(dae.state_dict(), f"checkpoints/q3/dae_d{d}.pt")
json.dump({"sigma": a.sigma, "bottleneck": d, "epochs": a.epochs, "lr": a.lr, "bs": a.bs},
          open("results/tables/q3b_hparams.json", "w"), indent=2)

ae = ConvAE(d).to(dev); ae.load_state_dict(torch.load(f"checkpoints/q3/ae_d{d}.pt", map_location=dev))
rows = pipeline_rows(ae, dae.to(dev), load_classifier(dev), xva, val_noisy, D["yva"],
                     D["mean"], D["std"], dev)
write_csv("results/tables/q3b_val.csv", rows)
print("\nVALIDATION pipeline (sigma=%.2f, bottleneck=%d)" % (a.sigma, d))
for r in rows:
    print(f"{r['input']:26s} mse {r['mse_vs_clean']:.5f} | acc {r['acc']:.4f} | conf {r['mean_conf']:.3f}")
