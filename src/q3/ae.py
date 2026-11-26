import csv, json
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from src.common.data import get_data
from src.q1.models import q1a_specs


class ConvAE(nn.Module):
    """Input/output: (N,3,32,32) pixels in [0,1]. Only the two bottleneck Linear layers depend on d."""
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
            nn.Flatten(), nn.Linear(2048, d))
        self.dec = nn.Sequential(
            nn.Linear(d, 2048), nn.ReLU(), nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid())

    def encode(self, x):
        return self.enc(x)

    def forward(self, x):
        return self.dec(self.enc(x))


def load_u8(cfg):
    d = get_data(cfg)
    g = lambda k: d[k].dataset
    return dict(xtr=g("train").x, ytr=g("train").y, xva=g("val").x, yva=g("val").y,
                xte=g("test").x, yte=g("test").y, mean=d["mean"], std=d["std"])


def add_noise(pix, sigma, gen=None):
    noise = torch.randn(pix.shape, generator=gen) if gen is not None else torch.randn_like(pix)
    return (pix + sigma * noise.to(pix.device)).clamp(0, 1)


def fixed_noisy(pix, sigma, seed):
    """Deterministic corruption (same noise every call) for val/test."""
    return add_noise(pix.cpu(), sigma, torch.Generator().manual_seed(seed))


@torch.no_grad()
def recon(model, x, dev, bs=1000):
    model.eval()
    return torch.cat([model(x[i:i + bs].to(dev)).cpu() for i in range(0, len(x), bs)])


@torch.no_grad()
def encode_all(model, u8, dev, bs=1000):
    model.eval()
    return torch.cat([model.encode(u8[i:i + bs].to(dev).float() / 255).cpu()
                      for i in range(0, len(u8), bs)])


def mse(a, b):
    return ((a - b) ** 2).mean().item()


def train_ae(model, xtr_u8, val_in, val_tgt, epochs, lr, bs, sigma, dev, seed, log_dir, run_name):
    """sigma=None -> clean AE (input=target). sigma>0 -> denoising AE (fresh noise each batch).
    Model selection by best validation MSE. No labels used."""
    model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xtr = xtr_u8.to(dev)
    N = len(xtr)
    g = torch.Generator().manual_seed(seed)
    H = {"epoch": [], "train_mse": [], "val_mse": []}
    best, best_state, best_ep = None, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(N, generator=g).to(dev)
        tot = 0.0
        for i in range(0, N, bs):
            tgt = xtr[perm[i:i + bs]].float() / 255
            inp = add_noise(tgt, sigma) if sigma else tgt
            loss = F.mse_loss(model(inp), tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(tgt)
        vm = mse(recon(model, val_in, dev), val_tgt)
        H["epoch"].append(ep); H["train_mse"].append(tot / N); H["val_mse"].append(vm)
        print(f"[{run_name}] ep {ep:03d} | train(running) {tot / N:.5f} | val {vm:.5f}", flush=True)
        if best is None or vm < best:
            best, best_ep = vm, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    H["best_epoch"] = best_ep
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    json.dump(H, open(f"{log_dir}/{run_name}.json", "w"))
    return H


def load_classifier(dev):
    m = q1a_specs()["cnn"]().to(dev)
    m.load_state_dict(torch.load("checkpoints/q1a/cnn_s0.pt", map_location=dev))
    return m.eval()


@torch.no_grad()
def classify(clf, pix, y, mean, std, dev, bs=1000):
    mean_t = torch.tensor(mean, device=dev).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=dev).view(1, 3, 1, 1)
    clf.eval()
    logits = torch.cat([clf((pix[i:i + bs].to(dev) - mean_t) / std_t).cpu() for i in range(0, len(pix), bs)])
    conf, pred = F.softmax(logits, 1).max(1)
    return dict(acc=(pred == y).float().mean().item(), mean_conf=conf.mean().item(),
                nll=F.cross_entropy(logits, y).item())


def pipeline_rows(ae, dae, clf, clean, noisy, y, mean, std, dev):
    items = [("clean images (reference)", clean), ("corrupted", noisy),
             ("clean-AE(corrupted)", recon(ae, noisy, dev)), ("DAE(corrupted)", recon(dae, noisy, dev)),
             ("clean-AE(clean)", recon(ae, clean, dev)), ("DAE(clean)", recon(dae, clean, dev))]
    rows = []
    for name, pix in items:
        e = mse(pix, clean)
        rows.append(dict(input=name, mse_vs_clean=e,
                         psnr_db=float("nan") if e == 0 else -10 * np.log10(e),
                         **classify(clf, pix, y, mean, std, dev)))
    return rows


def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
