import torch.nn as nn
from src.common.utils import count_params


def make_mlp(hidden, in_dim=3072, n_classes=10):
    layers, d = [nn.Flatten()], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, n_classes))
    return nn.Sequential(*layers)


def make_cnn(channels=(32, 64, 128), n_classes=10):
    layers, c = [], 3
    for ch in channels:
        layers += [nn.Conv2d(c, ch, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
        c = ch
    layers += [nn.Flatten(), nn.Linear(c * 4 * 4, n_classes)]   # 32 -> 4 after 3 pools
    return nn.Sequential(*layers)


def _mlp_params(w, depth, in_dim=3072, n_classes=10):
    return (in_dim * w + w) + (depth - 1) * (w * w + w) + (w * n_classes + n_classes)


def mlp_width_for_params(target, depth=2):
    return min(range(1, 8192), key=lambda w: abs(_mlp_params(w, depth) - target))


def q1a_specs(large_factor=4, depth=2):
    """Same specs everywhere (train + test scripts) so widths never drift."""
    cnn_params = count_params(make_cnn())
    w_m = mlp_width_for_params(cnn_params, depth)
    w_l = mlp_width_for_params(large_factor * cnn_params, depth)
    return {
        "cnn": lambda: make_cnn(),
        "mlp_matched": lambda: make_mlp([w_m] * depth),
        "mlp_large": lambda: make_mlp([w_l] * depth),
    }
