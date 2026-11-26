"""Q3c: frozen encoders -> linear + kNN probes. Hyper-parameters chosen on VAL; test evaluated once."""
import json
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from src.common.utils import load_config, set_seed
from src.q3.ae import ConvAE, load_u8, encode_all, recon, mse, write_csv

cfg = load_config("configs/base.yaml")
dev = cfg["device"]
D = load_u8(cfg)
ytr, yva, yte = D["ytr"].numpy(), D["yva"].numpy(), D["yte"].numpy()
models = pd.read_csv("results/tables/q3a_models.csv")
d_sel = json.load(open("results/tables/q3a_selected.json"))["bottleneck"]
test_clean = D["xte"].float() / 255

reps = []   # (name, d, model, has_decoder)
for _, r in models.iterrows():
    d = int(r["bottleneck"])
    m = ConvAE(d).to(dev); m.load_state_dict(torch.load(f"checkpoints/q3/ae_d{d}.pt", map_location=dev))
    reps.append((f"clean_AE_d{d}", d, m, True, r["val_mse"]))
m = ConvAE(d_sel).to(dev); m.load_state_dict(torch.load(f"checkpoints/q3/dae_d{d_sel}.pt", map_location=dev))
reps.append((f"DAE_d{d_sel} (clean input)", d_sel, m, True, float("nan")))
set_seed(cfg["seed"])
reps.append((f"random_untrained_d{d_sel}", d_sel, ConvAE(d_sel).to(dev), False, float("nan")))

rows = []
for name, d, m, has_dec, val_mse in reps:
    Ftr, Fva, Fte = [encode_all(m, D[k], dev).numpy() for k in ("xtr", "xva", "xte")]
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-8
    Ftr, Fva, Fte = [(f - mu) / sd for f in (Ftr, Fva, Fte)]
    best_lin = max(((LogisticRegression(C=C, max_iter=300).fit(Ftr, ytr), C) for C in (0.01, 0.1, 1.0)),
                   key=lambda t: t[0].score(Fva, yva))
    best_knn = max(((KNeighborsClassifier(k, n_jobs=-1).fit(Ftr, ytr), k) for k in (5, 20)),
                   key=lambda t: t[0].score(Fva, yva))
    rows.append(dict(representation=name, d=d,
                     val_recon_mse=val_mse,
                     test_recon_mse=mse(recon(m, test_clean, dev), test_clean) if has_dec else float("nan"),
                     linear_C=best_lin[1], linear_val_acc=best_lin[0].score(Fva, yva),
                     linear_test_acc=best_lin[0].score(Fte, yte),
                     knn_k=best_knn[1], knn_val_acc=best_knn[0].score(Fva, yva),
                     knn_test_acc=best_knn[0].score(Fte, yte)))
    print(rows[-1], flush=True)
write_csv("results/tables/q3c_probe.csv", rows)

ae_rows = [r for r in rows if r["representation"].startswith("clean_AE")]
mse_ = [r["test_recon_mse"] for r in ae_rows]
for k in ("linear_test_acc", "knn_test_acc"):
    rho = spearmanr(mse_, [r[k] for r in ae_rows])[0]
    print(f"Spearman(test recon MSE, {k}) over the {len(ae_rows)} clean AEs: {rho:.2f} "
          "(rho=-1 => perfect agreement: lower error <-> more class info; n=3 is tiny)")
