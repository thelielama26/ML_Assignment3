import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def _load_arrays(root, train):
    ds = CIFAR10(root=root, train=train, download=True)
    x = np.ascontiguousarray(ds.data.transpose(0, 3, 1, 2))  # N,3,32,32 uint8
    y = np.array(ds.targets, dtype=np.int64)
    return x, y


class ArrayDataset(Dataset):
    """uint8 images -> float [0,1] -> optional transform -> normalise.

    `transform` works on a (3,32,32) float tensor in [0,1], BEFORE normalisation.
    Pass the SAME transform to train/val/test when a question requires it.
    """
    def __init__(self, x, y, mean, std, transform=None):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        img = self.x[i].float() / 255.0
        if self.transform is not None:
            img = self.transform(img)
        return (img - self.mean) / self.std, self.y[i]


def get_data(cfg, transform=None, train_transform=None, batch_size=None,
             shuffle_train=True):
    """Returns dict(train, val, test loaders, mean, std).

    transform       : applied to ALL splits (e.g. a fixed pixel shuffle).
    train_transform : applied to the train split only, after `transform` (e.g. augmentation).
    Mean/std are computed on the 45k training split only.
    """
    split = np.load(cfg["split_path"])
    tr_idx, va_idx = split["train_idx"], split["val_idx"]
    assert int(split["seed"]) == cfg["seed"], "split seed != roll-number seed"

    x_all, y_all = _load_arrays(cfg["data_root"], train=True)
    x_te, y_te = _load_arrays(cfg["data_root"], train=False)

    x_tr, y_tr = x_all[tr_idx], y_all[tr_idx]
    x_va, y_va = x_all[va_idx], y_all[va_idx]

    flat = x_tr.astype(np.float32) / 255.0
    mean = flat.mean(axis=(0, 2, 3)).tolist()
    std = flat.std(axis=(0, 2, 3)).tolist()
    del flat

    def compose(*fs):
        fs = [f for f in fs if f is not None]
        if not fs:
            return None
        def g(img):
            for f in fs:
                img = f(img)
            return img
        return g

    ds_tr = ArrayDataset(x_tr, y_tr, mean, std, compose(transform, train_transform))
    ds_va = ArrayDataset(x_va, y_va, mean, std, compose(transform))
    ds_te = ArrayDataset(x_te, y_te, mean, std, compose(transform))

    bs = batch_size or cfg["batch_size"]
    g = torch.Generator().manual_seed(cfg["seed"])
    kw = dict(num_workers=cfg["num_workers"], pin_memory=cfg["device"] == "cuda")
    return {
        "train": DataLoader(ds_tr, bs, shuffle=shuffle_train, generator=g, **kw),
        "val": DataLoader(ds_va, bs, shuffle=False, **kw),
        "test": DataLoader(ds_te, bs, shuffle=False, **kw),
        "mean": mean, "std": std,
    }
