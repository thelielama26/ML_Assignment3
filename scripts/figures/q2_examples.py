import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))
import torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.common.utils import load_config
from src.common.data import _load_arrays
from src.q2.transforms import get_transform

cfg = load_config("configs/base.yaml")
x, y = _load_arrays(cfg["data_root"], train=False)
conds = ["clean", "perm", "gray", "patch8"]
fig, ax = plt.subplots(4, len(conds), figsize=(2.2 * len(conds), 9))
for r in range(4):
    img = torch.from_numpy(x[r]).float() / 255
    for c, cd in enumerate(conds):
        tf = get_transform(cd, cfg["seed"])
        a = (tf(img) if tf else img).permute(1, 2, 0).numpy()
        ax[r, c].imshow(a.clip(0, 1)); ax[r, c].axis("off")
        if r == 0: ax[r, c].set_title(cd)
fig.tight_layout(); Path("results/figures").mkdir(parents=True, exist_ok=True)
fig.savefig("results/figures/q2_examples.png", dpi=150); fig.savefig("results/figures/q2_examples.pdf")
