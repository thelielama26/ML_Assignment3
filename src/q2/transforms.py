"""Transforms act on a (3,32,32) float tensor in [0,1], before normalisation.
Same transform object is applied to train, val and test (see get_data(transform=...))."""
import torch


def pixel_permutation(seed):
    perm = torch.randperm(32 * 32, generator=torch.Generator().manual_seed(seed))
    def f(img):
        return img.reshape(img.shape[0], -1)[:, perm].reshape(img.shape)
    return f


def grayscale():
    w = torch.tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
    def f(img):
        return (img * w).sum(0, keepdim=True).expand(3, -1, -1).contiguous()  # keep 3 channels
    return f


def patch_shuffle(patch, seed):
    n = 32 // patch
    perm = torch.randperm(n * n, generator=torch.Generator().manual_seed(seed))
    def f(img):
        c = img.shape[0]
        p = img.reshape(c, n, patch, n, patch).permute(1, 3, 0, 2, 4).reshape(n * n, c, patch, patch)
        p = p[perm]
        return p.reshape(n, n, c, patch, patch).permute(2, 0, 3, 1, 4).reshape(c, 32, 32)
    return f


def get_transform(name, seed):
    if name == "clean":
        return None
    if name == "perm":
        return pixel_permutation(seed)
    if name == "gray":
        return grayscale()
    if name.startswith("patch"):
        return patch_shuffle(int(name[5:]), seed)
    raise ValueError(name)
