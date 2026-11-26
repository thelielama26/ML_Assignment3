"""Q3a: three clean conv AEs, differing ONLY in bottleneck size. Train/val only."""
import argparse, json
import torch
from src.common.utils import load_config, set_seed, count_params
from src.q3.ae import ConvAE, load_u8, train_ae, write_csv

p = argparse.ArgumentParser()
p.add_argument("--bottlenecks", nargs="+", type=int, default=[16, 64, 256])
p.add_argument("--epochs", type=int, default=40)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--bs", type=int, default=128)
a = p.parse_args()

cfg = load_config("configs/base.yaml")
dev = cfg["device"]
D = load_u8(cfg)
xva = D["xva"].float() / 255
json.dump(vars(a), open("results/tables/q3a_hparams.json", "w"), indent=2)

rows = []
for d in a.bottlenecks:
    set_seed(cfg["seed"])
    model = ConvAE(d)
    H = train_ae(model, D["xtr"], xva, xva, a.epochs, a.lr, a.bs, None, dev, cfg["seed"],
                 "logs/q3", f"ae_d{d}")
    torch.save(model.state_dict(), f"checkpoints/q3/ae_d{d}.pt")
    be = H["best_epoch"] - 1
    rows.append(dict(bottleneck=d, params=count_params(model), best_epoch=H["best_epoch"],
                     train_mse_running=H["train_mse"][be], val_mse=H["val_mse"][be]))
write_csv("results/tables/q3a_models.csv", rows)
sel = min(rows, key=lambda r: r["val_mse"])
json.dump({"bottleneck": sel["bottleneck"]}, open("results/tables/q3a_selected.json", "w"))
for r in rows:
    print(r)
print("SELECTED (lowest validation MSE):", sel["bottleneck"])
