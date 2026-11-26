import json
import torch
from src.common.utils import load_config
from src.q3.ae import ConvAE, load_u8, fixed_noisy, load_classifier, pipeline_rows, write_csv

cfg = load_config("configs/base.yaml")
dev = cfg["device"]
hp = json.load(open("results/tables/q3b_hparams.json"))
d, sigma = hp["bottleneck"], hp["sigma"]
D = load_u8(cfg)
clean = D["xte"].float() / 255
noisy = fixed_noisy(clean, sigma, cfg["seed"] + 202)            # fixed TEST corruption

ae = ConvAE(d).to(dev); ae.load_state_dict(torch.load(f"checkpoints/q3/ae_d{d}.pt", map_location=dev))
dae = ConvAE(d).to(dev); dae.load_state_dict(torch.load(f"checkpoints/q3/dae_d{d}.pt", map_location=dev))
rows = pipeline_rows(ae, dae, load_classifier(dev), clean, noisy, D["yte"], D["mean"], D["std"], dev)
write_csv("results/tables/q3b_test.csv", rows)

print(f"TEST (sigma={sigma}, bottleneck={d})")
for r in rows:
    print(f"{r['input']:26s} mse {r['mse_vs_clean']:.5f} | psnr {r['psnr_db']:.2f} | "
          f"acc {r['acc']:.4f} | conf {r['mean_conf']:.3f} | nll {r['nll']:.3f}")
acc = {r["input"]: r["acc"] for r in rows}
ref, bad = acc["clean images (reference)"], acc["corrupted"]
for k in ["clean-AE(corrupted)", "DAE(corrupted)"]:
    print(f"fraction of lost accuracy recovered by {k}: {(acc[k] - bad) / (ref - bad):.3f}")
print("selected clean AE test MSE:", [r["mse_vs_clean"] for r in rows if r["input"] == "clean-AE(clean)"][0])
