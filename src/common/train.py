import copy, csv, json, time
from pathlib import Path
import torch
import torch.nn.functional as F


# default = classification. For autoencoders pass your own loss_fn / correct_fn=None.
def cls_loss(out, x, y):
    return F.cross_entropy(out, y)


def cls_correct(out, x, y):
    return (out.argmax(1) == y).sum().item()


@torch.no_grad()
def evaluate(model, loader, device, loss_fn=cls_loss, correct_fn=cls_correct):
    """Returns (mean loss, accuracy). accuracy is NaN when correct_fn is None."""
    model.eval()
    tot_loss, tot_correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        b = x.size(0)
        tot_loss += loss_fn(out, x, y).item() * b
        if correct_fn is not None:
            tot_correct += correct_fn(out, x, y)
        n += b
    acc = tot_correct / n if correct_fn is not None else float("nan")
    return tot_loss / n, acc


def make_optimizer(model, name, lr, weight_decay=0.0, momentum=0.9):
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                               weight_decay=weight_decay)
    raise ValueError(name)


def fit(model, train_loader, val_loader, epochs, lr, device,
        optimizer="adam", weight_decay=0.0, momentum=0.9,
        loss_fn=cls_loss, correct_fn=cls_correct,
        select_by="val_loss",           # "val_loss" (lower better) or "val_acc" (higher better)
        restore_best=True,
        eval_mode_train_metrics=True,   # train loss/acc re-measured in eval mode each epoch
        log_dir=None, run_name="run"):
    """Per-epoch history: train_loss, train_acc, val_loss, val_acc, seconds.
    Selection uses VALIDATION only. Never pass a test loader here."""
    model.to(device)
    opt = make_optimizer(model, optimizer, lr, weight_decay, momentum)
    hist = {k: [] for k in ["epoch", "train_loss", "train_acc",
                            "val_loss", "val_acc", "seconds"]}
    best, best_epoch, best_state = None, -1, None

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        run_loss, run_correct, n = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = loss_fn(out, x, y)
            loss.backward()
            opt.step()
            b = x.size(0)
            run_loss += loss.item() * b
            if correct_fn is not None:
                run_correct += correct_fn(out.detach(), x, y)
            n += b

        if eval_mode_train_metrics:
            tr_loss, tr_acc = evaluate(model, train_loader, device, loss_fn, correct_fn)
        else:
            tr_loss = run_loss / n
            tr_acc = run_correct / n if correct_fn is not None else float("nan")
        va_loss, va_acc = evaluate(model, val_loader, device, loss_fn, correct_fn)

        for k, v in zip(hist, [ep, tr_loss, tr_acc, va_loss, va_acc, time.time() - t0]):
            hist[k].append(v)
        print(f"[{run_name}] ep {ep:03d} | train {tr_loss:.4f}/{tr_acc:.4f} "
              f"| val {va_loss:.4f}/{va_acc:.4f}", flush=True)

        score = va_loss if select_by == "val_loss" else -va_acc
        if best is None or score < best:
            best, best_epoch = score, ep
            best_state = copy.deepcopy(model.state_dict())

    hist["best_epoch"] = best_epoch
    if restore_best and best_state is not None:
        model.load_state_dict(best_state)

    if log_dir is not None:
        d = Path(log_dir)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{run_name}.json", "w") as f:
            json.dump(hist, f, indent=2)
        keys = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "seconds"]
        with open(d / f"{run_name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(keys)
            w.writerows(zip(*[hist[k] for k in keys]))
    return hist
